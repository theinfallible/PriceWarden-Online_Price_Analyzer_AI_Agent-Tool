#!/usr/bin/env python3
"""
Quick Start Script for AI Price Comparison Agent
This script helps you set up and run the entire project step by step
"""

import os
import sys
import subprocess
from pathlib import Path
import shutil
import platform

class QuickStart:
    def __init__(self):
        self.root_dir = Path.cwd()
        self.venv_dir = self.root_dir / "venv"
        self.is_windows = platform.system() == "Windows"
        
    def print_header(self, text):
        """Print formatted header"""
        print("\n" + "=" * 60)
        print(f"  {text}")
        print("=" * 60)
    
    def print_step(self, step_num, text):
        """Print formatted step"""
        print(f"\n{'→' * 3} Step {step_num}: {text}")
    
    def run_command(self, command, shell=True):
        """Run shell command"""
        try:
            result = subprocess.run(command, shell=shell, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Error: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"❌ Error running command: {e}")
            return False
    
    def create_project_structure(self):
        """Create all necessary directories"""
        self.print_step(1, "Creating project structure")
        
        directories = [
            "agent",
            "models",
            "tools",
            "data/dataset/raw",
            "data/dataset/processed",
            "data/checkpoints",
            "config"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_path}")
        
        # Create __init__.py files
        init_files = ["agent", "models", "tools"]
        for module in init_files:
            init_file = Path(module) / "__init__.py"
            init_file.touch()
            print(f"  ✓ Created {init_file}")
    
    def setup_virtual_environment(self):
        """Set up Python virtual environment"""
        self.print_step(2, "Setting up virtual environment")
        
        if self.venv_dir.exists():
            print("  ℹ️  Virtual environment already exists")
            return True
        
        print("  Creating virtual environment...")
        if self.run_command(f"{sys.executable} -m venv venv"):
            print("  ✓ Virtual environment created")
            return True
        return False
    
    def install_dependencies(self):
        """Install required packages"""
        self.print_step(3, "Installing dependencies")
        
        # Activate virtual environment and install packages
        if self.is_windows:
            pip_path = self.venv_dir / "Scripts" / "pip"
        else:
            pip_path = self.venv_dir / "bin" / "pip"
        
        # Create requirements.txt if it doesn't exist
        if not Path("requirements.txt").exists():
            self.create_requirements_file()
        
        print("  Installing packages (this may take a few minutes)...")
        if self.run_command(f"{pip_path} install -r requirements.txt"):
            print("  ✓ Dependencies installed successfully")
            return True
        return False
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements = """# Core Dependencies
streamlit==1.28.0
pillow==10.0.0
opencv-python==4.8.1.78

# AI/ML Dependencies
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
peft>=0.5.0
datasets>=2.14.0
accelerate>=0.20.0

# Web Scraping
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.0
webdriver-manager==4.0.1

# Utilities
pyyaml==6.0.1
pandas==2.1.3
numpy>=1.24.0
tqdm==4.66.1

# Evaluation Metrics
rouge-score==0.1.2
nltk==3.8.1
scikit-learn>=1.3.0
"""
        
        with open(".venv/requirements.txt", "w") as f:
            f.write(requirements)
        print("  ✓ Created requirements.txt")
    
    def create_config_file(self):
        """Create configuration file"""
        self.print_step(4, "Creating configuration file")
        
        config_content = """# Model Configuration
model:
  base_model: "Salesforce/blip-image-captioning-base"
  checkpoint_dir: "data/checkpoints"
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.1
    target_modules: ["q_proj", "v_proj"]

# Training Configuration
training:
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 3
  gradient_accumulation_steps: 4
  warmup_steps: 100
  max_length: 128

# Web Scraping Configuration
scraping:
  target_sites:
    - name: "amazon"
      url: "https://www.amazon.in"
      search_endpoint: "/s?k="
    - name: "flipkart"  
      url: "https://www.flipkart.com"
      search_endpoint: "/search?q="
    - name: "myntra"
      url: "https://www.myntra.com"
      search_endpoint: "/search?q="
  max_results_per_site: 5
  timeout: 10
  use_selenium: false

# Agent Configuration  
agent:
  max_retries: 3
  confidence_threshold: 0.7
"""
        
        config_path = Path("config/config.yaml")
        with open(config_path, "w") as f:
            f.write(config_content)
        print(f"  ✓ Created {config_path}")
    
    def download_sample_data(self):
        """Prepare sample data for testing"""
        self.print_step(5, "Preparing sample data")
        
        # Check if data preparation script exists
        if Path("prepare_dataset.py").exists():
            if self.is_windows:
                python_path = self.venv_dir / "Scripts" / "python"
            else:
                python_path = self.venv_dir / "bin" / "python"
            
            print("  Running data preparation script...")
            if self.run_command(f"{python_path} prepare_dataset.py"):
                print("  ✓ Sample data prepared")
                return True
        else:
            print("  ⚠️  Data preparation script not found")
            print("  ℹ️  Please download the Fashion Dataset from Kaggle:")
            print("     https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
        return False
    
    def test_installation(self):
        """Test if all modules can be imported"""
        self.print_step(6, "Testing installation")
        
        if self.is_windows:
            python_path = self.venv_dir / "Scripts" / "python"
        else:
            python_path = self.venv_dir / "bin" / "python"
        
        test_script = """
import sys
try:
    import streamlit
    import torch
    import transformers
    import peft
    import requests
    import bs4
    print("✓ All core modules imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)
"""
        
        # Write temporary test script
        with open("test_imports.py", "w") as f:
            f.write(test_script)
        
        success = self.run_command(f"{python_path} test_imports.py")
        
        # Clean up
        Path("test_imports.py").unlink()
        
        return success
    
    def create_launch_scripts(self):
        """Create convenient launch scripts"""
        self.print_step(7, "Creating launch scripts")
        
        # Create run.sh for Unix-like systems
        run_sh = """#!/bin/bash
echo "🚀 Starting AI Price Comparison Agent..."
source venv/bin/activate
streamlit run app.py
"""
        
        # Create run.bat for Windows
        run_bat = """@echo off
echo 🚀 Starting AI Price Comparison Agent...
call venv\\Scripts\\activate
streamlit run app.py
"""
        
        # Create train.sh for Unix-like systems
        train_sh = """#!/bin/bash
echo "🎯 Starting model fine-tuning..."
source venv/bin/activate
python models/fine_tune.py
"""
        
        # Create train.bat for Windows
        train_bat = """@echo off
echo 🎯 Starting model fine-tuning...
call venv\\Scripts\\activate
python models\\fine_tune.py
"""
        
        # Write scripts
        with open(".venv/run.sh", "w") as f:
            f.write(run_sh)
        with open(".venv/run.bat", "w") as f:
            f.write(run_bat)
        with open(".venv/train.sh", "w") as f:
            f.write(train_sh)
        with open(".venv/train.bat", "w") as f:
            f.write(train_bat)
        
        # Make shell scripts executable on Unix-like systems
        if not self.is_windows:
            os.chmod("run.sh", 0o755)
            os.chmod("train.sh", 0o755)
        
        print("  ✓ Created launch scripts")
    
    def display_next_steps(self):
        """Display instructions for next steps"""
        self.print_header("🎉 Setup Complete!")
        
        print("\n📋 Quick Start Guide:")
        print("-" * 40)
        
        print("\n1️⃣  To run the application:")
        if self.is_windows:
            print("   > .\\run.bat")
        else:
            print("   $ ./run.sh")
        
        print("\n2️⃣  To prepare your dataset:")
        if self.is_windows:
            print("   > venv\\Scripts\\activate")
            print("   > python prepare_dataset.py")
        else:
            print("   $ source venv/bin/activate")
            print("   $ python prepare_dataset.py")
        
        print("\n3️⃣  To fine-tune the model:")
        if self.is_windows:
            print("   > .\\train.bat")
        else:
            print("   $ ./train.sh")
        
        print("\n4️⃣  To evaluate performance:")
        if self.is_windows:
            print("   > venv\\Scripts\\activate")
            print("   > python evaluate.py")
        else:
            print("   $ source venv/bin/activate")
            print("   $ python evaluate.py")
        
        print("\n📚 Documentation:")
        print("-" * 40)
        print("• Configuration: config/config.yaml")
        print("• Web scraper: tools/web_scraper.py")
        print("• Agent logic: agent/core.py")
        print("• Model: models/inference.py")
        print("• UI: app.py")
        
        print("\n⚠️  Important Notes:")
        print("-" * 40)
        print("• Update web scraper selectors for actual websites")
        print("• Download Kaggle dataset for better model performance")
        print("• Respect websites' robots.txt and rate limits")
        print("• Consider using Selenium for JavaScript-heavy sites")
        
        print("\n🔗 Resources:")
        print("-" * 40)
        print("• Kaggle Dataset: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset")
        print("• Streamlit Docs: https://docs.streamlit.io")
        print("• Hugging Face: https://huggingface.co/docs")
        
    def run(self):
        """Run the complete setup process"""
        self.print_header("🚀 AI Price Comparison Agent - Quick Setup")
        
        print("\nThis script will set up your development environment.")
        print("Press Ctrl+C at any time to cancel.\n")
        
        try:
            # Step 1: Create project structure
            self.create_project_structure()
            
            # Step 2: Set up virtual environment
            if not self.setup_virtual_environment():
                print("\n❌ Failed to create virtual environment")
                return False
            
            # Step 3: Install dependencies
            if not self.install_dependencies():
                print("\n❌ Failed to install dependencies")
                print("ℹ️  Try installing packages manually:")
                print("   pip install -r requirements.txt")
                return False
            
            # Step 4: Create config file
            self.create_config_file()
            
            # Step 5: Download sample data
            self.download_sample_data()
            
            # Step 6: Test installation
            if not self.test_installation():
                print("\n⚠️  Some modules could not be imported")
                print("ℹ️  You may need to install them manually")
            
            # Step 7: Create launch scripts
            self.create_launch_scripts()
            
            # Display next steps
            self.display_next_steps()
            
            return True
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            return False

def main():
    """Main entry point"""
    quickstart = QuickStart()
    success = quickstart.run()
    
    if success:
        print("\n✨ Setup completed successfully!")
        print("🎯 Run './run.sh' (or '.\\run.bat' on Windows) to start the application")
    else:
        print("\n⚠️  Setup completed with some issues")
        print("📋 Please check the error messages above and fix any issues")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())