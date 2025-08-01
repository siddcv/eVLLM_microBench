import json
import random
from pathlib import Path
import argparse
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path where to save the JSONL file
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_percentage_dataset(input_file: str, output_dir: str, percentages: List[int], seed: int = 42) -> None:
    """
    Create percentage-based datasets from the input file.
    
    Args:
        input_file: Path to the input JSONL file
        output_dir: Directory to save the percentage datasets
        percentages: List of percentages to create (e.g., [25, 50, 75])
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the original data
    print(f"Loading data from: {input_file}")
    data = load_jsonl(input_file)
    total_samples = len(data)
    print(f"Total samples: {total_samples}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Group data by image_id to maintain image-text pairs
    image_groups = {}
    for item in data:
        image_id = item['image_id']
        if image_id not in image_groups:
            image_groups[image_id] = []
        image_groups[image_id].append(item)
    
    unique_images = list(image_groups.keys())
    print(f"Unique images: {len(unique_images)}")
    
    # Create datasets for each percentage
    for percentage in percentages:
        # Calculate number of unique images to include
        num_images = int(len(unique_images) * percentage / 100)
        
        # Randomly sample images
        selected_images = random.sample(unique_images, num_images)
        
        # Create dataset with all text entries for selected images
        percentage_data = []
        for image_id in selected_images:
            percentage_data.extend(image_groups[image_id])
        
        # Save the percentage dataset
        output_file = output_path / f"cardiovascular_{percentage}%.jsonl"
        save_jsonl(percentage_data, str(output_file))
        
        print(f"Created {percentage}% dataset: {len(percentage_data)} samples from {num_images} images")
        print(f"Saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Create percentage-based datasets for fine-tuning')
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to the input JSONL file')
    parser.add_argument('--output_dir', type=str, default='finetuning/data',
                       help='Directory to save the percentage datasets')
    parser.add_argument('--percentages', type=int, nargs='+', default=[25, 50, 75],
                       help='Percentages to create (default: 25 50 75)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return
    
    # Create percentage datasets
    create_percentage_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        percentages=args.percentages,
        seed=args.seed
    )
    
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main() 