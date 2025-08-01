#!/usr/bin/env python3
"""
Example usage of the BiomedClip LoRA fine-tuning pipeline.
"""

import sys
from pathlib import Path
import subprocess

def main():
    """Demonstrate the fine-tuning pipeline usage."""
    
    print("🚀 BiomedClip LoRA Fine-tuning Pipeline Example")
    print("=" * 60)
    
    # Check if the input data exists
    input_data = "organData/fineTuningData/cardiovascular/fineTuningData.jsonl"
    
    if not Path(input_data).exists():
        print(f"❌ Input data not found: {input_data}")
        print("Please ensure the fine-tuning data file exists.")
        return
    
    print(f"✅ Found input data: {input_data}")
    
    # Example 1: Prepare percentage-based datasets
    print("\n📊 Example 1: Prepare percentage-based datasets")
    print("Command:")
    print(f"python finetuning/scripts/prepare_data.py \\")
    print(f"    --input_file {input_data} \\")
    print(f"    --output_dir finetuning/data \\")
    print(f"    --percentages 25 50 75")
    
    # Example 2: Train a single model
    print("\n🤖 Example 2: Train a single model")
    print("Command:")
    print(f"python finetuning/scripts/train_lora.py \\")
    print(f"    --data_file finetuning/data/cardiovascular_50%.jsonl \\")
    print(f"    --output_dir finetuning/checkpoints/cardiovascular_50% \\")
    print(f"    --lora_r 8 \\")
    print(f"    --lora_alpha 16 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --max_epochs 10 \\")
    print(f"    --learning_rate 1e-4 \\")
    print(f"    --temperature 0.5")
    
    # Example 3: Run complete pipeline
    print("\n🎯 Example 3: Run complete pipeline")
    print("Command:")
    print(f"python finetuning/run_finetuning.py \\")
    print(f"    --input_data {input_data} \\")
    print(f"    --percentages 25 50 75 \\")
    print(f"    --lora_r 8 \\")
    print(f"    --lora_alpha 16 \\")
    print(f"    --batch_size 8 \\")
    print(f"    --max_epochs 10")
    
    # Example 4: Evaluate fine-tuned model
    print("\n📈 Example 4: Evaluate fine-tuned model")
    print("Command:")
    print(f"python src/evlm/inference/model_inference_wrapper.py \\")
    print(f"    --dataset_name cardiovascular \\")
    print(f"    --model FineTunedBioMedCLIP \\")
    print(f"    --output_dir output_results")
    
    print("\n" + "=" * 60)
    print("📝 Next Steps:")
    print("1. Install dependencies: pip install -r finetuning/requirements.txt")
    print("2. Run the complete pipeline: python finetuning/run_finetuning.py --input_data <your_data_file>")
    print("3. Monitor training with TensorBoard: tensorboard --logdir finetuning/logs")
    print("4. Evaluate models using the existing inference pipeline")
    
    print("\n📁 Expected Output Structure:")
    print("finetuning/")
    print("├── data/")
    print("│   ├── cardiovascular_25%.jsonl")
    print("│   ├── cardiovascular_50%.jsonl")
    print("│   └── cardiovascular_75%.jsonl")
    print("├── checkpoints/")
    print("│   ├── cardiovascular_25%/")
    print("│   ├── cardiovascular_50%/")
    print("│   └── cardiovascular_75%/")
    print("└── logs/")
    print("    └── biomedclip_lora/")
    
    print("\n🎉 Ready to start fine-tuning!")

if __name__ == "__main__":
    main() 