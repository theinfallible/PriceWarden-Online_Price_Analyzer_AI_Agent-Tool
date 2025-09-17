"""
Dataset Preparation Script for Fashion Product Images
This script helps prepare your dataset for fine-tuning the BLIP model
"""

import os
import pandas as pd
from pathlib import Path
import json
from PIL import Image
from sklearn.model_selection import train_test_split
import requests
from tqdm import tqdm
import zipfile
import shutil

#class DatasetPreparer:
    def __init__(self, base_dir="data"):
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "dataset"
        self.raw_dir = self.dataset_dir / "raw"
        self.processed_dir = self.dataset_dir / "processed"
        
        # Create directories
        for dir in [self.dataset_dir, self.raw_dir, self.processed_dir]:
            dir.mkdir(parents=True, exist_ok=True)
    
    def download_sample_dataset(self):
        """
        Download a sample fashion dataset for testing
        You should replace this with your actual Kaggle dataset
        """
        print("📥 Preparing sample dataset...")
        
        # Create sample data for testing
        sample_data = [
            {
                "image_name": "nike_airforce_white_001.jpg",
                "description": "Nike Air Force 1 '07 white leather sneakers low top mens shoes",
                "brand": "Nike",
                "category": "Sneakers",
                "color": "White",
                "gender": "Men"
            },
            {
                "image_name": "adidas_ultraboost_black_002.jpg",
                "description": "Adidas Ultraboost 22 black running shoes with boost cushioning technology",
                "brand": "Adidas",
                "category": "Running Shoes",
                "color": "Black",
                "gender": "Unisex"
            },
            {
                "image_name": "puma_suede_blue_003.jpg",
                "description": "Puma Suede Classic blue suede sneakers with white stripes casual shoes",
                "brand": "Puma",
                "category": "Sneakers",
                "color": "Blue",
                "gender": "Unisex"
            },
            {
                "image_name": "levis_denim_jacket_004.jpg",
                "description": "Levi's classic trucker denim jacket blue wash with button closure",
                "brand": "Levi's",
                "category": "Jacket",
                "color": "Blue",
                "gender": "Men"
            },
            {
                "image_name": "zara_floral_dress_005.jpg",
                "description": "Zara floral print midi dress summer collection with v-neck design",
                "brand": "Zara",
                "category": "Dress",
                "color": "Multicolor",
                "gender": "Women"
            }
        ]
        
        # Save as CSV
        df = pd.DataFrame(sample_data)
        df.to_csv(self.raw_dir / "sample_metadata.csv", index=False)
        print(f"✅ Sample metadata saved to {self.raw_dir / 'sample_metadata.csv'}")
        
        return df
    
    def prepare_kaggle_dataset(self, kaggle_zip_path):
        """
        Process actual Kaggle Fashion Product Images dataset
        Download from: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
        """
        print("📦 Extracting Kaggle dataset...")
        
        # Extract zip file
        with zipfile.ZipFile(kaggle_zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.raw_dir)
        
        # Load styles.csv which contains product information
        styles_df = pd.read_csv(self.raw_dir / "styles.csv", error_bad_lines=False)
        
        # Process and create training data
        processed_data = []
        
        for idx, row in tqdm(styles_df.iterrows(), total=len(styles_df), desc="Processing items"):
            try:
                # Construct image path
                image_id = row['id']
                image_path = self.raw_dir / "images" / f"{image_id}.jpg"
                
                if not image_path.exists():
                    continue
                
                # Create detailed description from available fields
                description_parts = []
                
                if pd.notna(row.get('gender')):
                    description_parts.append(row['gender'].lower())
                
                if pd.notna(row.get('masterCategory')):
                    description_parts.append(row['masterCategory'].lower())
                
                if pd.notna(row.get('subCategory')):
                    description_parts.append(row['subCategory'].lower())
                
                if pd.notna(row.get('articleType')):
                    description_parts.append(row['articleType'].lower())
                
                if pd.notna(row.get('baseColour')):
                    description_parts.append(row['baseColour'].lower())
                
                if pd.notna(row.get('season')):
                    description_parts.append(f"{row['season'].lower()} season")
                
                if pd.notna(row.get('productDisplayName')):
                    description_parts.append(row['productDisplayName'])
                
                # Combine into coherent description
                description = " ".join(description_parts)
                
                processed_data.append({
                    'image_path': str(image_path),
                    'description': description,
                    'brand': row.get('brandName', 'Unknown'),
                    'category': row.get('articleType', 'Unknown'),
                    'color': row.get('baseColour', 'Unknown'),
                    'gender': row.get('gender', 'Unisex')
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Create DataFrame
        processed_df = pd.DataFrame(processed_data)
        print(f"✅ Processed {len(processed_df)} items from Kaggle dataset")
        
        return processed_df
    
    def create_training_descriptions(self, df):
        """
        Create enhanced training descriptions for better search query generation
        """
        print("🔤 Enhancing descriptions for training...")
        
        enhanced_descriptions = []
        
        for idx, row in df.iterrows():
            # Create multiple variations of descriptions
            variations = []
            
            # Variation 1: Brand focused
            v1 = f"{row['brand']} {row['category']} {row['color']} {row['gender'].lower()}"
            variations.append(v1)
            
            # Variation 2: Detailed description
            if 'description' in row:
                variations.append(row['description'])
            
            # Variation 3: Search query style
            v3 = f"{row['color']} {row['brand']} {row['category']} for {row['gender'].lower()}"
            variations.append(v3)
            
            # Choose the most detailed one
            enhanced_desc = max(variations, key=len)
            enhanced_descriptions.append(enhanced_desc)
        
        df['enhanced_description'] = enhanced_descriptions
        return df
    
    def split_dataset(self, df, test_size=0.2, val_size=0.1):
        """
        Split dataset into train, validation, and test sets
        """
        print("✂️ Splitting dataset...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['category'] if 'category' in df.columns else None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val, 
            test_size=val_size_adjusted, 
            random_state=42,
            stratify=train_val['category'] if 'category' in train_val.columns else None
        )
        
        # Save splits
        train.to_csv(self.processed_dir / "train.csv", index=False)
        val.to_csv(self.processed_dir / "val.csv", index=False)
        test.to_csv(self.processed_dir / "test.csv", index=False)
        
        print(f"📊 Dataset split complete:")
        print(f"   - Train: {len(train)} samples")
        print(f"   - Validation: {len(val)} samples")
        print(f"   - Test: {len(test)} samples")
        
        return train, val, test
    
    def create_dummy_images(self, df):
        """
        Create dummy images for testing when real dataset is not available
        """
        print("🎨 Creating dummy images for testing...")
        
        images_dir = self.processed_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for idx, row in df.iterrows():
            if 'image_name' in row:
                # Create a simple colored image as placeholder
                img = Image.new('RGB', (224, 224), color='white')
                
                # Add some variety based on color
                if 'color' in row:
                    color_map = {
                        'white': (255, 255, 255),
                        'black': (50, 50, 50),
                        'blue': (100, 100, 200),
                        'red': (200, 100, 100),
                        'multicolor': (150, 150, 150)
                    }
                    color = color_map.get(row['color'].lower(), (200, 200, 200))
                    img = Image.new('RGB', (224, 224), color=color)
                
                # Save image
                image_path = images_dir / row['image_name']
                img.save(image_path)
                
                # Update path in dataframe
                df.at[idx, 'image_path'] = str(image_path)
        
        print(f"✅ Created {len(df)} dummy images")
        return df
    
    def validate_dataset(self, csv_path):
        """
        Validate that all images exist and are readable
        """
        print("🔍 Validating dataset...")
        
        df = pd.read_csv(csv_path)
        valid_rows = []
        invalid_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
            if 'image_path' in row:
                image_path = Path(row['image_path'])
                
                if image_path.exists():
                    try:
                        # Try to open image
                        img = Image.open(image_path)
                        img.verify()
                        valid_rows.append(row)
                    except:
                        invalid_count += 1
                else:
                    invalid_count += 1
            else:
                invalid_count += 1
        
        print(f"✅ Validation complete:")
        print(f"   - Valid samples: {len(valid_rows)}")
        print(f"   - Invalid samples: {invalid_count}")
        
        # Save validated dataset
        valid_df = pd.DataFrame(valid_rows)
        validated_path = csv_path.parent / f"validated_{csv_path.name}"
        valid_df.to_csv(validated_path, index=False)
        
        return valid_df

def main():
    """
    Main function to prepare dataset
    """
    preparer = DatasetPreparer()
    
    print("🚀 Fashion Dataset Preparation Tool")
    print("=" * 50)
    
    # Check if Kaggle dataset exists
    kaggle_zip = Path("fashion-dataset.zip")
    
    if kaggle_zip.exists():
        print("Found Kaggle dataset zip file")
        df = preparer.prepare_kaggle_dataset(kaggle_zip)
    else:
        print("⚠️ Kaggle dataset not found. Creating sample dataset for testing...")
        print("📌 Download the actual dataset from:")
        print("   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
        print("")
        
        # Create sample dataset
        df = preparer.download_sample_dataset()
        df = preparer.create_dummy_images(df)
    
    # Enhance descriptions
    df = preparer.create_training_descriptions(df)
    
    # Split dataset
    train, val, test = preparer.split_dataset(df)
    
    # Validate datasets
    print("\n📋 Validating splits...")
    preparer.validate_dataset(preparer.processed_dir / "train.csv")
    preparer.validate_dataset(preparer.processed_dir / "val.csv")
    preparer.validate_dataset(preparer.processed_dir / "test.csv")
    
    print("\n✨ Dataset preparation complete!")
    print(f"📁 Processed data saved to: {preparer.processed_dir}")
    
    # Print sample entries
    print("\n📝 Sample training entries:")
    train_sample = pd.read_csv(preparer.processed_dir / "train.csv")
    for idx, row in train_sample.head(3).iterrows():
        print(f"\nSample {idx + 1}:")
        print(f"  Description: {row.get('enhanced_description', row.get('description', 'N/A'))[:100]}...")
        if 'brand' in row:
            print(f"  Brand: {row['brand']}")
        if 'category' in row:
            print(f"  Category: {row['category']}")

if __name__ == "__main__":
    main()