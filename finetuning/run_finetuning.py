#!/usr/bin/env python3
"""
Script to run the complete fine-tuning pipeline for BiomedClip with LoRA.
"""

import subprocess
import sys
from pathlib import Path
import argparse
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("stdout:", e.stdout)
        if e.stderr:
            print("stderr:", e.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete fine-tuning pipeline')
    parser.add_argument('--input_data', type=str, required=True,
                       help='Path to the input fine-tuning data JSONL file')
    parser.add_argument('--percentages', type=int, nargs='+', default=[25, 50, 75],
                       help='Percentages to create (default: 25 50 75)')
    parser.add_argument('--lora_r', type=int, default=8,
                       help='LoRA rank (default: 8)')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha (default: 16)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--max_epochs', type=int, default=10,
                       help='Maximum epochs (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for contrastive loss (default: 0.5)')
    parser.add_argument('--skip_data_prep', action='store_true',
                       help='Skip data preparation step')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_data).exists():
        print(f"❌ Error: Input file not found: {args.input_data}")
        return
    
    # Step 1: Prepare percentage-based datasets
    if not args.skip_data_prep:
        print("\n📊 Step 1: Preparing percentage-based datasets...")
        
        data_prep_cmd = [
            sys.executable, "finetuning/scripts/prepare_data.py",
            "--input_file", args.input_data,
            "--output_dir", "finetuning/data",
            "--percentages"
        ] + [str(p) for p in args.percentages]
        
        if not run_command(" ".join(data_prep_cmd), "Data preparation"):
            print("❌ Data preparation failed. Exiting.")
            return
    
    # Step 2: Train models for each percentage
    print("\n🤖 Step 2: Training models for each percentage...")
    
    for percentage in args.percentages:
        data_file = f"finetuning/data/cardiovascular_{percentage}%.jsonl"
        
        if not Path(data_file).exists():
            print(f"❌ Data file not found: {data_file}")
            continue
        
        print(f"\n🎯 Training model for {percentage}% of data...")
        
        # Create output directory for this percentage
        output_dir = f"finetuning/checkpoints/cardiovascular_{percentage}%"
        os.makedirs(output_dir, exist_ok=True)
        
        # Training command
        train_cmd = [
            sys.executable, "finetuning/scripts/train_lora.py",
            "--data_file", data_file,
            "--output_dir", output_dir,
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--batch_size", str(args.batch_size),
            "--max_epochs", str(args.max_epochs),
            "--learning_rate", str(args.learning_rate),
            "--temperature", str(args.temperature)
        ]
        
        if not run_command(" ".join(train_cmd), f"Training for {percentage}%"):
            print(f"❌ Training failed for {percentage}%. Continuing with next percentage...")
            continue
        
        print(f"✅ Training completed for {percentage}%")
    
    print("\n🎉 Fine-tuning pipeline completed!")
    print("\n📁 Generated files:")
    print("  - Percentage datasets: finetuning/data/")
    print("  - Model checkpoints: finetuning/checkpoints/")
    print("  - LoRA weights: finetuning/checkpoints/*/final_lora_weights.pt")
    print("  - Training logs: finetuning/logs/")

if __name__ == "__main__":
    main() 