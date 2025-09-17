import subprocess
import sys

def check_versions():
    import transformers
    import peft
    import torch

    print("Current versions:")
    print(f"  transformers: {transformers.__version__}")
    print(f"  peft: {peft.__version__}")
    print(f"  torch: {torch.__version__}")

    peft_version = tuple(map(int, peft.__version__.split('.')[:2]))
    transformers_version = tuple(map(int, transformers.__version__.split('.')[:2]))

    if peft_version >= (0, 6) and transformers_version < (4, 35):
        print("\nPotential compatibility issue detected!")
        print("PEFT 0.6+ works best with transformers 4.35+")
        return False

    return True

def fix_compatibility():
    print("\nInstalling compatible versions...")
    commands = [
        "pip install peft==0.5.0",
    ]

    for cmd in commands:
        print(f"Running: {cmd}")
        subprocess.run(cmd.split(), check=True)

    print("\nCompatibility fix applied!")
    print("Please restart your Python kernel/interpreter for changes to take effect.")

if __name__ == "__main__":
    print("Checking library compatibility...")
    print("="*50)

    compatible = check_versions()

    if not compatible:
        response = input("\nWould you like to fix the compatibility issue? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            fix_compatibility()
        else:
            print("\nManual fix options:")
            print("1. Downgrade PEFT: pip install peft==0.5.0")
            print("2. OR upgrade transformers: pip install transformers>=4.35.0")
    else:
        print("\nLibraries appear to be compatible!")
