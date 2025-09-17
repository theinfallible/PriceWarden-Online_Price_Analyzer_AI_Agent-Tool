"""
Myntra Dataset Preparation Script
Prepares the Myntra fashion dataset for fine-tuning the BLIP model
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyntraDatasetPreparer:
    def __init__(self, base_path="archive (2)/myntradataset"):
        """Initialize with the path to your Myntra dataset."""
        self.base_path = Path(base_path)
        self.images_dir = self.base_path / "images"
        self.styles_csv = self.base_path / "styles.csv"

        # Output directories
        self.output_dir = Path("../data/myntra_processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Dataset path: {self.base_path}")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Styles CSV: {self.styles_csv}")

    def load_and_process_styles(self):
        """Load the styles.csv and create training descriptions."""
        if not self.styles_csv.exists():
            raise FileNotFoundError(f"styles.csv not found at {self.styles_csv}")

        logger.info("Loading styles.csv...")

        # Load with error handling for bad lines
        df = pd.read_csv(self.styles_csv, on_bad_lines='skip')
        logger.info(f"Loaded {len(df)} product entries")

        # Process each row to create training data
        processed_data = []
        missing_images = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing products"):
            try:
                # Construct image path - Myntra uses ID.jpg format
                image_id = row.get('id', '')
                if pd.isna(image_id):
                    continue

                image_path = self.images_dir / f"{int(image_id)}.jpg"

                # Check if image exists
                if not image_path.exists():
                    missing_images += 1
                    continue

                # Create detailed description from available fields
                description_parts = []

                # Add gender if available
                if pd.notna(row.get('gender')):
                    description_parts.append(row['gender'].lower())

                # Add color
                if pd.notna(row.get('baseColour')):
                    description_parts.append(row['baseColour'].lower())

                # Add brand
                if pd.notna(row.get('brandName')):
                    description_parts.append(row['brandName'])

                # Add article type (main product category)
                if pd.notna(row.get('articleType')):
                    description_parts.append(row['articleType'].lower())

                # Add subcategory
                if pd.notna(row.get('subCategory')):
                    description_parts.append(row['subCategory'].lower())

                # Add master category
                if pd.notna(row.get('masterCategory')):
                    description_parts.append(row['masterCategory'].lower())

                # Add season if available
                if pd.notna(row.get('season')):
                    description_parts.append(f"for {row['season'].lower()}")

                # Add year if available
                if pd.notna(row.get('year')):
                    description_parts.append(f"{int(row['year'])} collection")

                # Create the full description
                description = " ".join(description_parts)

                # Alternative: use productDisplayName if available
                if pd.notna(row.get('productDisplayName')):
                    alt_description = row['productDisplayName']
                    # Use the longer, more detailed description
                    if len(alt_description) > len(description):
                        description = alt_description

                processed_data.append({
                    'image_path': str(image_path.absolute()),
                    'description': description,
                    'brand': row.get('brandName', 'Unknown'),
                    'category': row.get('articleType', 'Unknown'),
                    'color': row.get('baseColour', 'Unknown'),
                    'gender': row.get('gender', 'Unisex'),
                    'id': int(image_id)
                })

            except Exception as e:
                logger.debug(f"Error processing row {idx}: {e}")
                continue

        logger.info(f"Successfully processed {len(processed_data)} products")
        logger.info(f"Missing images: {missing_images}")

        return pd.DataFrame(processed_data)

    def validate_images(self, df):
        """Validate that all images in the dataframe are readable."""
        logger.info("Validating images...")
        valid_rows = []
        invalid_count = 0

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            try:
                # Try to open and verify the image
                img = Image.open(row['image_path'])
                img.verify()  # Verify it's a valid image
                valid_rows.append(row)
            except Exception as e:
                logger.debug(f"Invalid image {row['image_path']}: {e}")
                invalid_count += 1

        logger.info(f"Valid images: {len(valid_rows)}")
        logger.info(f"Invalid images: {invalid_count}")

        return pd.DataFrame(valid_rows)

    def create_train_val_test_split(self, df, train_size=0.7, val_size=0.15, test_size=0.15):
        """Split the dataset into train, validation, and test sets."""
        logger.info("Creating train/val/test splits...")

        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=42
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (train_size + val_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=42
        )

        logger.info(f"Train size: {len(train)}")
        logger.info(f"Validation size: {len(val)}")
        logger.info(f"Test size: {len(test)}")

        return train, val, test

    def save_datasets(self, train, val, test):
        """Save the datasets to CSV files."""
        train_path = self.output_dir / "train.csv"
        val_path = self.output_dir / "val.csv"
        test_path = self.output_dir / "test.csv"

        # Select only necessary columns for training
        columns_to_save = ['image_path', 'description']

        train[columns_to_save].to_csv(train_path, index=False)
        val[columns_to_save].to_csv(val_path, index=False)
        test[columns_to_save].to_csv(test_path, index=False)

        logger.info(f"Saved train dataset to {train_path}")
        logger.info(f"Saved validation dataset to {val_path}")
        logger.info(f"Saved test dataset to {test_path}")

        return train_path, val_path, test_path

    def prepare_full_dataset(self, sample_size=None):
        """Main method to prepare the complete dataset."""
        logger.info("=" * 50)
        logger.info("Starting Myntra Dataset Preparation")
        logger.info("=" * 50)

        # Load and process styles
        df = self.load_and_process_styles()

        # Sample if requested (useful for testing)
        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size} products for testing...")
            df = df.sample(n=sample_size, random_state=42)

        # Validate images
        df = self.validate_images(df)

        # Create splits
        train, val, test = self.create_train_val_test_split(df)

        # Save datasets
        train_path, val_path, test_path = self.save_datasets(train, val, test)

        # Print sample entries
        logger.info("\nSample training entries:")
        for idx, row in train.head(3).iterrows():
            logger.info(f"\nSample {idx + 1}:")
            logger.info(f"  Description: {row['description'][:100]}...")

        logger.info("\n" + "=" * 50)
        logger.info("Dataset preparation complete!")
        logger.info("=" * 50)

        return train_path, val_path, test_path


def main():
    """Main function to prepare the Myntra dataset."""

    # Initialize the preparer with your dataset path
    preparer = MyntraDatasetPreparer(base_path="../archive (2)/myntradataset")

    # Prepare full dataset (or use sample_size for testing)
    # For initial testing, you might want to use a smaller sample:
    # train_path, val_path, test_path = preparer.prepare_full_dataset(sample_size=1000)

    # For full dataset:
    train_path, val_path, test_path = preparer.prepare_full_dataset()

    print(f"\nDatasets ready for fine-tuning:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    print(f"  Test: {test_path}")

    print("\nNext steps:")
    print("1. Run the fine-tuning script:")
    print(f"   python models/fine_tune.py")


if __name__ == "__main__":
    main()