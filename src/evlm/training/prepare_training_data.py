import json
import pandas as pd
from pathlib import Path
import argparse
from typing import List, Dict, Any
import random

def prepare_training_data(
    input_data: List[Dict[str, Any]], 
    output_path: str,
    image_dir: str = None,
    text_column: str = 'text',
    image_column: str = 'image_path',
    split_ratio: float = 0.8
) -> None:
    """
    Prepare training data for BiomedCLIP fine-tuning.
    
    Args:
        input_data: List of dictionaries containing image and text data.
        output_path: Path to save the prepared data.
        image_dir: Directory containing images (if image paths are relative).
        text_column: Name of the column containing text data.
        image_column: Name of the column containing image paths.
        split_ratio: Ratio for train/validation split.
    """
    
    # Prepare data
    prepared_data = []
    
    for item in input_data:
        # Extract text and image path
        text = item.get(text_column, '')
        image_path = item.get(image_column, '')
        
        # Skip if missing required data
        if not text or not image_path:
            continue
        
        # Make image path absolute if image_dir is provided
        if image_dir and not Path(image_path).is_absolute():
            image_path = str(Path(image_dir) / image_path)
        
        # Create training example
        training_example = {
            'image_path': image_path,
            'text': text
        }
        
        prepared_data.append(training_example)
    
    # Shuffle data
    random.shuffle(prepared_data)
    
    # Split into train and validation
    split_idx = int(len(prepared_data) * split_ratio)
    train_data = prepared_data[:split_idx]
    val_data = prepared_data[split_idx:]
    
    # Save train data
    train_path = Path(output_path) / 'train.jsonl'
    train_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation data
    val_path = Path(output_path) / 'val.jsonl'
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Prepared {len(train_data)} training examples and {len(val_data)} validation examples")
    print(f"Train data saved to: {train_path}")
    print(f"Validation data saved to: {val_path}")

def prepare_from_csv(
    csv_path: str,
    output_path: str,
    image_dir: str = None,
    text_column: str = 'text',
    image_column: str = 'image_path',
    split_ratio: float = 0.8
) -> None:
    """
    Prepare training data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        output_path: Path to save the prepared data.
        image_dir: Directory containing images.
        text_column: Name of the column containing text data.
        image_column: Name of the column containing image paths.
        split_ratio: Ratio for train/validation split.
    """
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    # Convert to list of dictionaries
    data = df.to_dict('records')
    
    # Prepare data
    prepare_training_data(
        data, 
        output_path, 
        image_dir, 
        text_column, 
        image_column, 
        split_ratio
    )

def prepare_from_jsonl(
    jsonl_path: str,
    output_path: str,
    image_dir: str = None,
    text_column: str = 'text',
    image_column: str = 'image_path',
    split_ratio: float = 0.8
) -> None:
    """
    Prepare training data from a JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file.
        output_path: Path to save the prepared data.
        image_dir: Directory containing images.
        text_column: Name of the column containing text data.
        image_column: Name of the column containing image paths.
        split_ratio: Ratio for train/validation split.
    """
    
    # Read JSONL file
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Prepare data
    prepare_training_data(
        data, 
        output_path, 
        image_dir, 
        text_column, 
        image_column, 
        split_ratio
    )

def create_sample_data(output_path: str, num_samples: int = 100) -> None:
    """
    Create sample training data for testing.
    
    Args:
        output_path: Path to save the sample data.
        num_samples: Number of sample data points to create.
    """
    
    # Sample data
    sample_data = []
    
    for i in range(num_samples):
        # Create sample image path (you'll need to replace with actual images)
        image_path = f"sample_images/sample_{i:03d}.jpg"
        
        # Create sample text
        texts = [
            "A medical image showing normal tissue structure.",
            "Microscopic view of cellular components.",
            "Pathological sample with abnormal findings.",
            "Histological section of tissue sample.",
            "Medical imaging showing diagnostic features."
        ]
        
        text = random.choice(texts)
        
        sample_data.append({
            'image_path': image_path,
            'text': text
        })
    
    # Save sample data
    sample_path = Path(output_path) / 'sample_data.jsonl'
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(sample_path, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sample data saved to: {sample_path}")
    print("Note: You'll need to replace the image paths with actual image files.")

def main():
    parser = argparse.ArgumentParser(description='Prepare training data for BiomedCLIP fine-tuning')
    
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input file (CSV or JSONL)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save prepared data')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Directory containing images (if paths are relative)')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the column containing text data')
    parser.add_argument('--image_column', type=str, default='image_path',
                       help='Name of the column containing image paths')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Ratio for train/validation split')
    parser.add_argument('--create_sample', action='store_true',
                       help='Create sample data instead of processing input file')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of sample data points to create')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_data(args.output_path, args.num_samples)
    else:
        # Determine file type and process accordingly
        input_path = Path(args.input_file)
        
        if input_path.suffix.lower() == '.csv':
            prepare_from_csv(
                args.input_file,
                args.output_path,
                args.image_dir,
                args.text_column,
                args.image_column,
                args.split_ratio
            )
        elif input_path.suffix.lower() == '.jsonl':
            prepare_from_jsonl(
                args.input_file,
                args.output_path,
                args.image_dir,
                args.text_column,
                args.image_column,
                args.split_ratio
            )
        else:
            print(f"Unsupported file format: {input_path.suffix}")
            print("Supported formats: .csv, .jsonl")

if __name__ == "__main__":
    main() 