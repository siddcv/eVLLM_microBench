#!/usr/bin/env python3
"""
Quick start script for BiomedCLIP LoRA fine-tuning.

This script provides a simple way to get started with LoRA fine-tuning
without needing to understand all the details.
"""

import argparse
import sys
from pathlib import Path

def print_banner():
    """Print a welcome banner."""
    print("=" * 60)
    print("🚀 BiomedCLIP LoRA Fine-tuning Quick Start")
    print("=" * 60)
    print()

def check_installation():
    """Check if the LoRA setup is properly installed."""
    print("Checking installation...")
    
    try:
        # Test imports
        import torch
        from peft import LoraConfig
        from src.evlm.models.openCLIP_models.biomedclip_lora import BioMedCLIPLoRA
        
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nTo install dependencies, run:")
        print("pip install -r requirements_lora.txt")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n📚 Usage Examples:")
    print()
    
    print("1. Test your installation:")
    print("   python test_lora_installation.py")
    print()
    
    print("2. Prepare training data from CSV:")
    print("   python src/evlm/training/prepare_training_data.py \\")
    print("       --input_file your_data.csv \\")
    print("       --output_path ./prepared_data \\")
    print("       --image_dir /path/to/images \\")
    print("       --text_column description \\")
    print("       --image_column image_path")
    print()
    
    print("3. Train the model:")
    print("   python src/evlm/training/train_biomedclip_lora.py \\")
    print("       --train_data ./prepared_data/train.jsonl \\")
    print("       --val_data ./prepared_data/val.jsonl \\")
    print("       --output_dir ./checkpoints \\")
    print("       --batch_size 16 \\")
    print("       --num_epochs 10")
    print()
    
    print("4. Run inference:")
    print("   python src/evlm/inference/biomedclip_lora_inference.py \\")
    print("       --lora_path ./checkpoints/lora_adapters \\")
    print("       --images image1.jpg image2.jpg \\")
    print("       --texts 'Description 1' 'Description 2'")
    print()

def show_data_format():
    """Show the expected data format."""
    print("\n📋 Expected Data Format:")
    print()
    print("Your training data should be in JSONL format:")
    print()
    print('{"image_path": "/path/to/image1.jpg", "text": "Description of image 1"}')
    print('{"image_path": "/path/to/image2.jpg", "text": "Description of image 2"}')
    print()
    print("Or CSV format with columns:")
    print("- image_path: Path to the image file")
    print("- text: Text description of the image")
    print()

def show_parameters():
    """Show key parameters for LoRA fine-tuning."""
    print("\n⚙️  Key LoRA Parameters:")
    print()
    print("--lora_r: LoRA rank (default: 16)")
    print("  - Higher values = more parameters = better performance")
    print("  - Recommended: 16-64 for most tasks")
    print()
    print("--lora_alpha: LoRA alpha parameter (default: 32)")
    print("  - Usually set to 2 * lora_r")
    print()
    print("--learning_rate: Learning rate (default: 1e-4)")
    print("  - Start with 1e-4, adjust based on convergence")
    print()
    print("--batch_size: Training batch size (default: 32)")
    print("  - Reduce if you run out of memory")
    print()

def show_troubleshooting():
    """Show common troubleshooting tips."""
    print("\n🔧 Troubleshooting:")
    print()
    print("Out of Memory:")
    print("- Reduce batch_size (try 8 or 16)")
    print("- Reduce lora_r (try 8 or 16)")
    print("- Use gradient accumulation")
    print()
    print("Poor Convergence:")
    print("- Check learning rate (try 5e-5 or 1e-3)")
    print("- Verify data quality and format")
    print("- Increase num_epochs")
    print()
    print("Import Errors:")
    print("- Run: pip install -r requirements_lora.txt")
    print("- Check Python path includes src/ directory")
    print()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Quick start guide for BiomedCLIP LoRA fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quick_start_lora.py --test
  python quick_start_lora.py --examples
  python quick_start_lora.py --format
  python quick_start_lora.py --params
  python quick_start_lora.py --troubleshoot
        """
    )
    
    parser.add_argument('--test', action='store_true',
                       help='Test the installation')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')
    parser.add_argument('--format', action='store_true',
                       help='Show expected data format')
    parser.add_argument('--params', action='store_true',
                       help='Show key parameters')
    parser.add_argument('--troubleshoot', action='store_true',
                       help='Show troubleshooting tips')
    parser.add_argument('--all', action='store_true',
                       help='Show all information')
    
    args = parser.parse_args()
    
    print_banner()
    
    # If no specific option is selected, show all
    if not any([args.test, args.examples, args.format, args.params, args.troubleshoot, args.all]):
        args.all = True
    
    if args.test or args.all:
        if check_installation():
            print("🎉 Installation looks good!")
        else:
            print("❌ Installation needs attention.")
        print()
    
    if args.examples or args.all:
        show_usage_examples()
    
    if args.format or args.all:
        show_data_format()
    
    if args.params or args.all:
        show_parameters()
    
    if args.troubleshoot or args.all:
        show_troubleshooting()
    
    if args.all:
        print("📖 For detailed documentation, see: docs/BIOMEDCLIP_LORA_FINETUNING.md")
        print()
        print("🚀 Ready to start fine-tuning!")
        print("Run 'python test_lora_installation.py' to verify everything is working.")

if __name__ == "__main__":
    main() 