import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
from dataclasses import dataclass
from typing import Dict, List, Union, Any
import yaml
from PIL import Image
import logging
import os
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BlipDataCollator:
    processor: BlipProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        batch = {}

        if 'pixel_values' in features[0]:
            batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])

        if 'input_ids' in features[0]:
            input_ids = [f['input_ids'] for f in features]
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
            )

        if 'attention_mask' in features[0]:
            attention_masks = [f['attention_mask'] for f in features]
            batch['attention_mask'] = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )

        if 'labels' in features[0]:
            labels = [f['labels'] for f in features]
            batch['labels'] = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=-100
            )

        return batch


class BlipPeftTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):

        allowed_keys = {'pixel_values', 'input_ids', 'attention_mask', 'labels'}

        filtered_inputs = {k: v for k, v in inputs.items() if k in allowed_keys}

        logger.debug(f"Passing to model: {list(filtered_inputs.keys())}")

        outputs = model(**filtered_inputs)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        return (loss, outputs) if return_outputs else loss


class ModelFineTuner:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading base model: {self.model_config['base_model']}")

        self.processor = BlipProcessor.from_pretrained(self.model_config['base_model'])
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_config['base_model'],
            torch_dtype=torch.float32
        )

        self.model = prepare_model_for_kbit_training(self.model)
        self.model = self.model.to(self.device)

    def prepare_dataset(self, dataset_path: str) -> Dataset:
        logger.info(f"Preparing dataset from: {dataset_path}")
        dataset = load_dataset('csv', data_files=dataset_path)['train']

        def preprocess_function(example):
            try:
                image = Image.open(example['image_path']).convert('RGB')

                encoding = self.processor(
                    images=image,
                    text=example['description'],
                    padding='max_length',
                    truncation=True,
                    max_length=self.training_config.get('max_length', 77),
                    return_tensors='pt'
                )

                result = {
                    'pixel_values': encoding['pixel_values'].squeeze(),
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze(),
                }
                result['labels'] = result['input_ids'].clone()

                return result

            except Exception as e:
                logger.warning(f"Error processing {example.get('image_path', 'unknown')}: {e}")
                return {}

        processed = dataset.map(
            preprocess_function,
            remove_columns=dataset.column_names,
            desc="Processing dataset"
        )

        processed = processed.filter(lambda x: 'pixel_values' in x and x['pixel_values'] is not None)

        processed.set_format(type='torch', columns=['pixel_values', 'input_ids', 'attention_mask', 'labels'])

        logger.info(f"Dataset ready with {len(processed)} examples")
        return processed

    def setup_lora(self):
        logger.info("Configuring LoRA...")

        target_modules = ["query", "value"]

        lora_config = LoraConfig(
            r=self.model_config['lora_config'].get('r', 8),
            lora_alpha=self.model_config['lora_config'].get('lora_alpha', 16),
            lora_dropout=self.model_config['lora_config'].get('lora_dropout', 0.1),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            modules_to_save=None,
        )

        logger.info(f"Applying LoRA to modules: {target_modules}")
        self.model = get_peft_model(self.model, lora_config)

        def _safe_blip_forward(orig_forward):
            BLIP_ALLOWED = {
                "pixel_values", "input_ids", "attention_mask",
                "labels", "return_dict", "output_attentions",
                "output_hidden_states"
            }

            def wrapper(*args, **kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k in BLIP_ALLOWED}
                return orig_forward(*args, **kwargs)

            return wrapper

        self.model.base_model.forward = _safe_blip_forward(self.model.base_model.forward)

        self.model.print_trainable_parameters()

        '''if hasattr(self.model, "base_model"):
            self.model.base_model.forward = self.model.base_model.forward.__wrapped__'''

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        self.setup_lora()

        os.makedirs(self.model_config['checkpoint_dir'], exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.model_config['checkpoint_dir'],
            num_train_epochs=self.training_config.get('num_epochs', 2),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 8),
            warmup_steps=self.training_config.get('warmup_steps', 10),
            learning_rate=float(self.training_config.get('learning_rate', 5e-5)),
            logging_steps=5,
            save_steps=50,
            eval_steps=25 if eval_dataset else None,
            save_total_limit=2,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            push_to_hub=False,
            remove_unused_columns=False,  # Important!
            label_names=["labels"],
            report_to="none",
            dataloader_num_workers=0,
            fp16=False,
            bf16=False,
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )

        data_collator = BlipDataCollator(processor=self.processor)

        trainer = BlipPeftTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset)}")

        try:
            trainer.train()

            output_dir = os.path.join(self.model_config['checkpoint_dir'], "final_model")
            trainer.save_model(output_dir)
            self.processor.save_pretrained(output_dir)

            logger.info(f"Training complete! Model saved to {output_dir}")

            with open(os.path.join(self.model_config['checkpoint_dir'], "training_complete.txt"), "w") as f:
                f.write("Training completed successfully!")

            return trainer

        except Exception as e:
            logger.error(f"Training failed: {e}")
            emergency_save = os.path.join(self.model_config['checkpoint_dir'], "emergency_checkpoint")
            try:
                trainer.save_model(emergency_save)
                logger.info(f"Emergency checkpoint saved to {emergency_save}")
            except:
                pass
            raise


def simple_finetune_without_lora():
    logger.info("Running simple fine-tuning without LoRA...")

    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for param in model.parameters():
        param.requires_grad = False

    for param in model.text_decoder.bert.encoder.layer[-2:].parameters():
        param.requires_grad = True

    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    logger.info("Simple fine-tuning configured. This approach trains fewer parameters but avoids PEFT issues.")

    return model, processor


if __name__ == "__main__":
    try:
        fine_tuner = ModelFineTuner()

        train_path = "data/myntra_processed/train.csv"
        val_path = "data/myntra_processed/val.csv"

        if os.path.exists(train_path):
            train_dataset = fine_tuner.prepare_dataset(train_path)
            eval_dataset = fine_tuner.prepare_dataset(val_path) if os.path.exists(val_path) else None

            fine_tuner.train(train_dataset, eval_dataset)

            print("\n" + "="*50)
            print("SUCCESS! Fine-tuning completed!")
            print("="*50)
        else:
            logger.error("Dataset not found. Run data preparation first.")

    except Exception as e:
        logger.error(f"PEFT training failed: {e}")
        logger.info("\nTrying simple fine-tuning without LoRA...")

        model, processor = simple_finetune_without_lora()
        logger.info("Use the simple fine-tuning approach if PEFT continues to fail.")