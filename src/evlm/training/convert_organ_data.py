#!/usr/bin/env python3
"""
Convert organ data format to BiomedCLIP LoRA training format.

This script converts the complex organ data JSONL format to the simple
image-text pairs format required for BiomedCLIP LoRA training.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

def extract_text_from_questions(questions: Dict[str, Any]) -> List[str]:
    """
    Extract meaningful text descriptions from the questions field.
    
    Args:
        questions: Dictionary containing question-answer pairs.
    
    Returns:
        List of text descriptions extracted from questions.
    """
    texts = []
    
    for question_name, question_data in questions.items():
        if isinstance(question_data, dict) and 'question' in question_data:
            # Extract the question text
            question_text = question_data['question']
            
            # Extract the answer
            answer = question_data.get('answer', '')
            
            # Create a meaningful description
            if answer and answer != 'none of the above':
                # Create a description based on the question type
                if 'classification' in question_name:
                    texts.append(f"Medical image showing {answer}")
                elif 'domain' in question_name:
                    texts.append(f"Image from {answer} field")
                elif 'modality' in question_name:
                    texts.append(f"Image acquired using {answer}")
                elif 'stain' in question_name:
                    texts.append(f"Image stained with {answer}")
                elif 'subdomain' in question_name:
                    texts.append(f"Image from {answer} subfield")
                elif 'submodality' in question_name:
                    texts.append(f"Image using {answer} technique")
    
    return texts

def extract_text_from_captions(captions: Dict[str, Any]) -> List[str]:
    """
    Extract text descriptions from the captions field.
    
    Args:
        captions: Dictionary containing caption data.
    
    Returns:
        List of text descriptions extracted from captions.
    """
    texts = []
    
    for caption_name, caption_data in captions.items():
        if isinstance(caption_data, dict) and 'question' in caption_data:
            question_text = caption_data['question']
            answer_idx = caption_data.get('answer_idx', '0')
            options = caption_data.get('options', [])
            
            try:
                answer_idx = int(answer_idx)
                if 0 <= answer_idx < len(options):
                    answer = options[answer_idx]
                    if answer != 'none of the above':
                        texts.append(answer)
            except (ValueError, IndexError):
                continue
    
    return texts

def create_training_example(data_item: Dict[str, Any], image_dir: str) -> Optional[Dict[str, str]]:
    """
    Create a training example from a data item.
    
    Args:
        data_item: Dictionary containing the data item.
        image_dir: Directory containing the images.
    
    Returns:
        Dictionary with image_path and text, or None if invalid.
    """
    # Extract image path
    image_filename = data_item.get('image', '')
    if not image_filename:
        return None
    
    # Construct full image path
    image_path = str(Path(image_dir) / image_filename)
    
    # Extract text descriptions
    texts = []
    
    # Extract from questions
    questions = data_item.get('questions', {})
    texts.extend(extract_text_from_questions(questions))
    
    # Extract from captions
    captions = data_item.get('captions', {})
    texts.extend(extract_text_from_captions(captions))
    
    # Add metadata-based descriptions
    metadata_texts = []
    
    # Add domain information
    domain = data_item.get('domain', '')
    if domain:
        metadata_texts.append(f"Medical image from {domain} domain")
    
    # Add subdomain information
    subdomain = data_item.get('subdomain', '')
    if subdomain:
        metadata_texts.append(f"Image from {subdomain}")
    
    # Add modality information
    modality = data_item.get('modality', '')
    if modality:
        metadata_texts.append(f"Image acquired using {modality}")
    
    # Add stain information
    stain = data_item.get('stain', '')
    if stain:
        metadata_texts.append(f"Image stained with {stain}")
    
    # Add label information
    label_name = data_item.get('label_name', '')
    if label_name:
        metadata_texts.append(f"Image showing {label_name}")
    
    # Combine all texts
    all_texts = texts + metadata_texts
    
    if not all_texts:
        return None
    
    # Use the first meaningful text, or combine multiple texts
    if len(all_texts) == 1:
        text = all_texts[0]
    else:
        # Combine multiple descriptions
        text = ". ".join(all_texts[:3])  # Limit to first 3 descriptions
    
    return {
        'image_path': image_path,
        'text': text
    }

def convert_organ_data(
    input_file: str,
    output_dir: str,
    image_dir: str,
    split_ratio: float = 0.8,
    max_samples: Optional[int] = None
) -> None:
    """
    Convert organ data format to BiomedCLIP training format.
    
    Args:
        input_file: Path to the input JSONL file.
        output_dir: Directory to save the converted data.
        image_dir: Directory containing the images.
        split_ratio: Ratio for train/validation split.
        max_samples: Maximum number of samples to process (for testing).
    """
    
    print(f"Converting data from: {input_file}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    
    # Read input data
    data_items = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            if max_samples and line_num >= max_samples:
                break
            
            try:
                data_item = json.loads(line.strip())
                data_items.append(data_item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num + 1}: {e}")
                continue
    
    print(f"Loaded {len(data_items)} data items")
    
    # Convert to training format
    training_examples = []
    skipped = 0
    
    for item in data_items:
        example = create_training_example(item, image_dir)
        if example:
            training_examples.append(example)
        else:
            skipped += 1
    
    print(f"Created {len(training_examples)} training examples")
    print(f"Skipped {skipped} items (no valid text/image)")
    
    if not training_examples:
        print("No valid training examples found!")
        return
    
    # Shuffle data
    random.shuffle(training_examples)
    
    # Split into train/validation
    split_idx = int(len(training_examples) * split_ratio)
    train_data = training_examples[:split_idx]
    val_data = training_examples[split_idx:]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train data
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Save validation data
    val_file = output_path / "val.jsonl"
    with open(val_file, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nConversion completed!")
    print(f"Train examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    print(f"Train file: {train_file}")
    print(f"Validation file: {val_file}")
    
    # Show some examples
    print(f"\nSample training examples:")
    for i, example in enumerate(train_data[:3]):
        print(f"  {i+1}. Image: {example['image_path']}")
        print(f"     Text: {example['text']}")
        print()

def main():
    parser = argparse.ArgumentParser(description='Convert organ data to BiomedCLIP training format')
    
    parser.add_argument('--input_file', type=str, required=True,
                       help='Path to input JSONL file (e.g., cardiovascular.jsonl)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save converted data')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing the images')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                       help='Ratio for train/validation split (default: 0.8)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Validate image directory
    if not Path(args.image_dir).exists():
        print(f"Error: Image directory {args.image_dir} does not exist")
        return
    
    # Convert data
    convert_organ_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        split_ratio=args.split_ratio,
        max_samples=args.max_samples
    )

if __name__ == "__main__":
    main() 