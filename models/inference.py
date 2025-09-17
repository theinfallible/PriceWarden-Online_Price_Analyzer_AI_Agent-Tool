import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel, PeftConfig
import yaml
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageToTextModel:
    def __init__(self, config_path="config/config.yaml", use_finetuned=False):

        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.model_config = self.config['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.processor = BlipProcessor.from_pretrained(self.model_config['base_model'])
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_config['base_model']
        ).to(self.device)

        if use_finetuned:
            self._load_finetuned_model()

    def _load_finetuned_model(self):
        checkpoint_path = self.model_config.get('finetuned_checkpoint', 'data/checkpoints/best_model')

        if os.path.exists(checkpoint_path):
            try:
                self.model = BlipForConditionalGeneration.from_pretrained(checkpoint_path).to(self.device)
                self.processor = BlipProcessor.from_pretrained(checkpoint_path)
                logger.info(f"Loaded fine-tuned model from {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Could not load fine-tuned model from {checkpoint_path}: {e}")
        else:
            logger.warning(f"Checkpoint not found at {checkpoint_path}")

    def generate_search_query(self, image: Image.Image, max_length: int = 50) -> str:

        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                temperature=0.7,
                do_sample=True
            )

        generated_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)

        search_query = self._refine_search_query(generated_text)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return search_query

    def _refine_search_query(self, raw_caption: str) -> str:
        phrases_to_remove = [
            "with a white background",
            "on a white background",
            "a photo of",
            "a close up of",
            "a picture of",
            "an image of",
        ]

        clean_caption = raw_caption.lower()
        for phrase in phrases_to_remove:
            clean_caption = clean_caption.replace(phrase, "")

        stopwords = ['a', 'an', 'the', 'with', 'of', 'in', 'on', 'at', 'is', 'are']
        words = clean_caption.split()

        refined_words = [w for w in words if w not in stopwords or len(w) > 3]

        search_query = ' '.join(refined_words)
        return search_query.strip()