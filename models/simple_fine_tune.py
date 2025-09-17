import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
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
    """Custom data collator for BLIP that only includes required keys"""
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


class SimpleBlipTrainer(Trainer):
    """Simple trainer without PEFT complications"""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        model_inputs = {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs.get('attention_mask'),
            'labels': inputs.get('labels')
        }
        outputs = model(**model_inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


class SimpleModelFineTuner:
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
        self.model = self.model.to(self.device)
        self.setup_selective_training()

    def setup_selective_training(self):
        logger.info("Setting up selective fine-tuning...")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.text_decoder.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        for param in self.model.text_decoder.cls.parameters():
            param.requires_grad = True
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable percentage: {100 * trainable_params / total_params:.4f}%")

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

    def train(self, train_dataset: Dataset, eval_dataset: Dataset = None):
        os.makedirs(self.model_config['checkpoint_dir'], exist_ok=True)
        training_args = TrainingArguments(
            output_dir=self.model_config['checkpoint_dir'],
            num_train_epochs=self.training_config.get('num_epochs', 3),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=self.training_config.get('gradient_accumulation_steps', 4),
            warmup_steps=self.training_config.get('warmup_steps', 100),
            learning_rate=float(self.training_config.get('learning_rate', 2e-5)),
            logging_steps=10,
            save_steps=100,
            eval_steps=50 if eval_dataset else None,
            save_total_limit=3,
            eval_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=bool(eval_dataset),
            metric_for_best_model="eval_loss" if eval_dataset else None,
            push_to_hub=False,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to="none",
            dataloader_num_workers=0,
            fp16=False,
            bf16=False,
            prediction_loss_only=True,
        )
        data_collator = BlipDataCollator(processor=self.processor)
        trainer = SimpleBlipTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        logger.info("Starting simple fine-tuning...")
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
                f.write("Simple fine-tuning completed successfully!")
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


if __name__ == "__main__":
    try:
        logger.info("Starting simple fine-tuning approach (without PEFT)...")
        fine_tuner = SimpleModelFineTuner()
        train_path = "data/myntra_processed/train.csv"
        val_path = "data/myntra_processed/val.csv"
        if os.path.exists(train_path):
            train_dataset = fine_tuner.prepare_dataset(train_path)
            eval_dataset = fine_tuner.prepare_dataset(val_path) if os.path.exists(val_path) else None
            fine_tuner.train(train_dataset, eval_dataset)
            print("\n" + "=" * 50)
            print("SUCCESS! Simple fine-tuning completed!")
            print("=" * 50)
        else:
            logger.error("Dataset not found. Run data preparation first.")
    except Exception as e:
        logger.error(f"Simple fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
