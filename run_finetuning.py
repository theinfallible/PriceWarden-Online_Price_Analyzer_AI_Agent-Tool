import sys
from pathlib import Path

project_root = Path.cwd()
sys.path.insert(0, str(project_root))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_myntra_data():
    logger.info("Step 1: Preparing Myntra dataset...")

    from scraper_testing.prepare_myntra_data import MyntraDatasetPreparer

    preparer = MyntraDatasetPreparer(base_path="archive (2)/myntradataset")

    train_path, val_path, test_path = preparer.prepare_full_dataset(sample_size=100)

    return train_path, val_path, test_path


def run_fine_tuning(train_path, val_path):
    logger.info("Step 2: Starting fine-tuning...")

    from models.fine_tune import ModelFineTuner
    #from models.simple_fine_tune import SimpleModelFineTuner as ModelFineTuner

    fine_tuner = ModelFineTuner()

    logger.info(f"Loading training data from {train_path}")
    train_dataset = fine_tuner.prepare_dataset(str(train_path))

    logger.info(f"Loading validation data from {val_path}")
    eval_dataset = fine_tuner.prepare_dataset(str(val_path))

    fine_tuner.train(train_dataset, eval_dataset=eval_dataset)

    logger.info("Fine-tuning complete!")


def main():
    print("=" * 60)
    print("  MYNTRA FASHION MODEL FINE-TUNING PIPELINE")
    print("=" * 60)

    try:
        train_path, val_path, test_path = prepare_myntra_data()

        print("\nDataset prepared successfully!")
        print(f"  Train samples: {train_path}")
        print(f"  Validation samples: {val_path}")
        print(f"  Test samples: {test_path}")

        response = input("\nDo you want to start fine-tuning now? (yes/no): ").lower().strip()

        if response in ['yes', 'y']:
            run_fine_tuning(train_path, val_path)

            print("\n" + "=" * 60)
            print("  FINE-TUNING COMPLETE!")
            print("=" * 60)
            print("\nTo use the fine-tuned model:")
            print("1. Update your app.py to use the fine-tuned model")
            print("2. Set use_finetuned=True when initializing the agent")
            print("3. Run: streamlit run app.py")

        else:
            print("\nFine-tuning skipped. You can run it later using:")
            print(f"  python models/fine_tune.py")
            print("\nDatasets have been saved to:")
            print(f"  {train_path}")
            print(f"  {val_path}")
            print(f"  {test_path}")

    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())