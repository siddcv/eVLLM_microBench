#!/usr/bin/env python3
"""
Test script to demonstrate organ data conversion and show what goes into test split.
"""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from evlm.training.convert_organ_data import create_training_example

def test_single_example():
    """Test conversion of a single example from your data."""
    
    # Your sample data (from the attached file)
    sample_data = {
        "image_id": "02bf6398-5f9a-479a-9359-8182ff96ce43",
        "image": "02bf6398-5f9a-479a-9359-8182ff96ce43.png",
        "label": 65,
        "label_name": "chronic_heart_failure",
        "dataset": "nirschl_et_al_2018",
        "domain": "pathology",
        "institution": "upenn",
        "license": "CC-BY-4.0",
        "microns_per_pixel": 2.0,
        "modality": "light microscopy",
        "ncbitaxon_id": "NCBITaxon_9606",
        "ncbitaxon_name": "Homo sapiens",
        "pmid": None,
        "split": "test",  # This indicates it's in the test split
        "stain": "H&E",
        "subdomain": "cardiovascular pathology",
        "submodality": "brightfield microscopy",
        "synthetic": False,
        "captions": {
            "modality_0": {
                "id": "f01087e7-6e77-4b77-9462-95afb53dabca",
                "name": "modality",
                "question": "A micrograph acquired through {class}.",
                "answer_idx": "2",
                "options": [
                    "A micrograph acquired through fluorescence microscopy.",
                    "A micrograph acquired through electron microscopy.",
                    "A micrograph acquired through light microscopy.",
                    "none of the above"
                ],
                "tags": None
            },
            "classification_0": {
                "id": "0d1bb749-cb9e-403a-9d80-a48cd22e68b2",
                "name": "classification",
                "question": "An H&E micrograph of {class}.",
                "answer_idx": "1",
                "options": [
                    "An H&E micrograph of not chronic heart failure.",
                    "An H&E micrograph of chronic heart failure.",
                    "none of the above"
                ],
                "tags": None
            }
        },
        "questions": {
            "classification": {
                "answer": "chronic heart failure",
                "answer_idx": 1,
                "id": "0d1bb749-cb9e-403a-9d80-a48cd22e68b2",
                "name": "classification",
                "options": ["not chronic heart failure", "chronic heart failure", "none of the above"],
                "question": "H&E stained light micrograph of human cardiac tissue. Based on the image, what is the most likely clinical chronic heart diagnosis?",
                "tags": None
            },
            "domain": {
                "answer": "pathology",
                "answer_idx": 0,
                "id": "d93d29da-3097-4b1f-8ca4-84a3551aaa25",
                "name": "domain",
                "options": ["pathology", "radiology", "biology", "cytology", "dermatology", "none of the above"],
                "question": "What is the most likely field of study this micrograph would be used for?",
                "tags": None
            },
            "modality": {
                "answer": "light microscopy",
                "answer_idx": 2,
                "id": "f01087e7-6e77-4b77-9462-95afb53dabca",
                "name": "modality",
                "options": ["fluorescence microscopy", "electron microscopy", "light microscopy", "none of the above"],
                "question": "What is the most likely microscopy modality used to acquire this image?",
                "tags": None
            },
            "stain": {
                "answer": "H&E",
                "answer_idx": 0,
                "id": "cc361a46-c0c8-4599-a194-b1998529de0e",
                "name": "stain",
                "options": ["H&E", "Ziehl-Neelsen", "Basic fuchsin", "PAS", "IHC(HDab)", "none of the above"],
                "question": "What is the most likely technique used to stain this micrograph?",
                "tags": None
            },
            "subdomain": {
                "answer": "cardiovascular pathology",
                "answer_idx": 1,
                "id": "85bf8e8a-8704-418a-b6aa-1840377360d2",
                "name": "subdomain",
                "options": ["gynecologic pathology", "cardiovascular pathology", "molecular pathology", "hematopathology", "head and neck pathology", "none of the above"],
                "question": "What is the most likely subfield of study this micrograph would be used for?",
                "tags": None
            },
            "submodality": {
                "answer": "brightfield microscopy",
                "answer_idx": 3,
                "id": "35282c0b-3d4c-4150-8e41-a77712ffae43",
                "name": "submodality",
                "options": ["mixed", "polarized light microscopy", "darkfield microscopy", "brightfield microscopy", "differential interference contrast microscopy", "none of the above"],
                "question": "What is the most likely microscopy submodality used to acquire this image?",
                "tags": None
            }
        }
    }
    
    print("=== Testing Data Conversion ===")
    print()
    
    # Show original data structure
    print("Original data structure:")
    print(f"- Image: {sample_data['image']}")
    print(f"- Split: {sample_data['split']}")  # This shows it's in test split
    print(f"- Domain: {sample_data['domain']}")
    print(f"- Subdomain: {sample_data['subdomain']}")
    print(f"- Modality: {sample_data['modality']}")
    print(f"- Stain: {sample_data['stain']}")
    print(f"- Label: {sample_data['label_name']}")
    print()
    
    # Convert to training format
    image_dir = "organData/cardiovascular/json"  # Assuming images are in json subdirectory
    training_example = create_training_example(sample_data, image_dir)
    
    if training_example:
        print("Converted training example:")
        print(f"- Image path: {training_example['image_path']}")
        print(f"- Text: {training_example['text']}")
        print()
        
        # Show what would go into different splits
        print("Split assignment:")
        print(f"- Original split: {sample_data['split']}")
        print(f"- Would go into: {'TEST' if sample_data['split'] == 'test' else 'TRAIN'}")
        print()
        
        # Show the actual text that would be used for training
        print("Text extraction breakdown:")
        
        # Extract from questions
        from evlm.training.convert_organ_data import extract_text_from_questions
        question_texts = extract_text_from_questions(sample_data['questions'])
        print("From questions:")
        for text in question_texts:
            print(f"  - {text}")
        
        # Extract from captions
        from evlm.training.convert_organ_data import extract_text_from_captions
        caption_texts = extract_text_from_captions(sample_data['captions'])
        print("From captions:")
        for text in caption_texts:
            print(f"  - {text}")
        
        print()
        print("Final combined text:")
        print(f"  '{training_example['text']}'")
        
    else:
        print("Failed to create training example!")

def show_split_logic():
    """Show how the split logic works."""
    print("\n=== Split Logic ===")
    print()
    print("Your data has a 'split' field that indicates the original split:")
    print("- 'train' → Goes into TRAIN split")
    print("- 'test' → Goes into TEST split")
    print("- 'val' → Goes into VALIDATION split")
    print()
    print("When converting for LoRA training:")
    print("1. All 'train' items → Training data")
    print("2. All 'test' items → Validation data (for monitoring)")
    print("3. All 'val' items → Validation data")
    print()
    print("You can also override this by using --split_ratio to create new splits.")

def show_usage_example():
    """Show how to use the conversion script."""
    print("\n=== Usage Example ===")
    print()
    print("To convert your cardiovascular data:")
    print()
    print("python src/evlm/training/convert_organ_data.py \\")
    print("    --input_file organData/cardiovascular/cardiovascular.jsonl \\")
    print("    --output_dir ./prepared_data \\")
    print("    --image_dir organData/cardiovascular/json \\")
    print("    --split_ratio 0.8")
    print()
    print("This will create:")
    print("- ./prepared_data/train.jsonl")
    print("- ./prepared_data/val.jsonl")
    print()
    print("Then you can train with:")
    print()
    print("python src/evlm/training/train_biomedclip_lora.py \\")
    print("    --train_data ./prepared_data/train.jsonl \\")
    print("    --val_data ./prepared_data/val.jsonl \\")
    print("    --output_dir ./checkpoints")

def main():
    """Main function."""
    test_single_example()
    show_split_logic()
    show_usage_example()

if __name__ == "__main__":
    main() 