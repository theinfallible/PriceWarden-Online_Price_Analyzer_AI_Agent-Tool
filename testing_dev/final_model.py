#!/usr/bin/env python3
"""
find_and_test_model.py - Finds your fine-tuned model and tests loading it
"""

import os
import json
from pathlib import Path


def find_saved_models():
    print("=" * 60)
    print("SEARCHING FOR SAVED MODELS")
    print("=" * 60)

    search_paths = [
        "data/checkpoints",
        "data",
        "models",
        "checkpoints",
        "."
    ]

    found_models = []

    for base_path in search_paths:
        if not os.path.exists(base_path):
            continue

        for root, dirs, files in os.walk(base_path):
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            model_indicators = [
                "pytorch_model.bin",
                "model.safetensors",
                "adapter_model.bin",
                "adapter_config.json",
                "config.json"
            ]

            for indicator in model_indicators:
                if indicator in files:
                    found_models.append(root)
                    break

    found_models = list(set(found_models))

    if found_models:
        print(f"\nFound {len(found_models)} model checkpoint(s):\n")
        for i, model_path in enumerate(found_models, 1):
            print(f"{i}. {model_path}")

            files = os.listdir(model_path)
            print(f"   Files: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")

            if "adapter_config.json" in files:
                print("   Type: PEFT/LoRA fine-tuned model")
                with open(os.path.join(model_path, "adapter_config.json"), 'r') as f:
                    config = json.load(f)
                    print(f"   Task: {config.get('task_type', 'Unknown')}")
            elif "pytorch_model.bin" in files:
                print("   Type: Standard fine-tuned model")

            model_file = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                print(f"   Model size: {size_mb:.1f} MB")

            print()
    else:
        print("\nNo model checkpoints found!")
        print("Did the fine-tuning complete successfully?")

    return found_models


def test_loading_model(model_path):
    print(f"\nTesting model loading from: {model_path}")
    print("-" * 40)

    try:
        print("Attempting to load as standard BLIP model...")
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model = BlipForConditionalGeneration.from_pretrained(model_path)
        print("Successfully loaded as standard model")

        processor_path = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(processor_path):
            processor = BlipProcessor.from_pretrained(model_path)
            print("Processor also loaded successfully")

        return True

    except Exception as e:
        print(f"Failed to load as standard model: {e}")

    try:
        print("\nAttempting to load as PEFT/LoRA model...")
        from peft import PeftModel, PeftConfig
        from transformers import BlipForConditionalGeneration

        base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = PeftModel.from_pretrained(base_model, model_path)
        print("Successfully loaded as PEFT/LoRA model")

        return True

    except ImportError:
        print("PEFT not installed. Install with: pip install peft")
    except Exception as e:
        print(f"Failed to load as PEFT model: {e}")

    return False


def update_config_file(model_path):
    print("\n" + "=" * 60)
    print("CONFIG UPDATE NEEDED")
    print("=" * 60)

    print("\nUpdate your config/config.yaml with this path:")
    print("-" * 40)
    print(f"""
model:
  base_model: "Salesforce/blip-image-captioning-base"
  checkpoint_dir: "data/checkpoints"
  finetuned_checkpoint: "{model_path}"
""")

    print("\nAnd make sure agent/core.py has:")
    print("-" * 40)
    print("""
class PriceComparisonAgent:
    def __init__(self, config_path="config/config.yaml", use_finetuned=True):
        self.image_model = ImageToTextModel(config_path, use_finetuned=True)
""")


def main():
    print("\nFINE-TUNED MODEL DIAGNOSTIC TOOL\n")

    found_models = find_saved_models()

    if not found_models:
        print("\nNo models found. Please check:")
        print("1. Did fine-tuning complete successfully?")
        print("2. Check the training logs for the save path")
        return

    if len(found_models) == 1:
        model_to_test = found_models[0]
    else:
        print("\nWhich model should we test? (Enter number or press Enter for first):")
        choice = input("> ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(found_models):
            model_to_test = found_models[int(choice) - 1]
        else:
            model_to_test = found_models[0]

    success = test_loading_model(model_to_test)

    if success:
        update_config_file(model_to_test)
        print("\nModel can be loaded successfully!")
        print("Update your config with the path above and restart the app.")
    else:
        print("\nModel loading failed. Check the error messages above.")


if __name__ == "__main__":
    main()
