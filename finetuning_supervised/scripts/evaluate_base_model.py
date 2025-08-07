#!/usr/bin/env python3
"""
Evaluate base BioMedCLIP model on test data - Simple version
"""

import json
import os
import sys
import torch
import pandas as pd
from pathlib import Path
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the BioMedCLIP model
from biomedclip_lora import BioMedCLIPLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_base_model(test_file: str, organ_domain: str):
    """Evaluate base BioMedCLIP model on test data."""
    
    # Load test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    # Initialize base model (without LoRA)
    model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
    model.load_base_model()  # Load base model without LoRA adapters
    
    results = []
    
    for data_point in tqdm(test_data, desc="Evaluating base model"):
        image_path = data_point.get('image_path', '')
        question = data_point.get('question', '')
        answer_options = data_point.get('answer_options', [])
        correct_answer_idx = data_point.get('correct_answer_idx', 0)
        question_class = data_point.get('question_type', '')
        
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Get model prediction
        try:
            prediction_result = model.predict_single(image_path, question, answer_options)
            
            predicted_idx = prediction_result['predicted_idx']
            confidence = prediction_result['confidence']
            scores = prediction_result['all_probs']
            
            # Create result entry
            result = {
                'question_class': question_class,
                'questions': str([f"{question} {option}" for option in answer_options]),
                'image_id': data_point.get('image_id', ''),
                'correct_answer': answer_options[correct_answer_idx],
                'correct_idx': correct_answer_idx,
                'model_answers': {
                    'pred': [predicted_idx],
                    'probs': [scores],
                    'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
                },
                'microns_per_pixel': data_point.get('microns_per_pixel', 2.0),
                'domain': data_point.get('domain', ''),
                'subdomain': data_point.get('subdomain', ''),
                'modality': data_point.get('modality', ''),
                'submodality': data_point.get('submodality', ''),
                'normal_or_abnormal': data_point.get('normal_or_abnormal', '')
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error predicting for {data_point.get('image_id', '')}: {e}")
            continue
    
    # Save results
    df = pd.DataFrame(results)
    output_dir = Path(f"output_results/combined_grained_tasks/BioMedCLIP_Base")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{organ_domain}.csv"
    df.to_csv(csv_path, index=False)
    
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct_idx'] == r['model_answers']['pred'][0])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0
    
    logger.info(f"Base model accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"Results saved to {csv_path}")
    
    # Per-question-type accuracy
    if len(results) > 0:
        question_types = df['question_class'].unique()
        for q_type in question_types:
            q_results = df[df['question_class'] == q_type]
            q_correct = sum(1 for _, row in q_results.iterrows() 
                          if row['correct_idx'] == row['model_answers']['pred'][0])
            q_total = len(q_results)
            q_accuracy = q_correct / q_total if q_total > 0 else 0.0
            logger.info(f"{q_type} accuracy: {q_accuracy:.4f} ({q_correct}/{q_total})")
    else:
        logger.warning("No results generated")

def main():
    parser = argparse.ArgumentParser(description='Evaluate base BioMedCLIP model')
    parser.add_argument('--test_file', type=str, required=True,
                       help='Path to test.json file')
    parser.add_argument('--organ_domain', type=str, required=True,
                       help='Organ domain name')
    
    args = parser.parse_args()
    evaluate_base_model(args.test_file, args.organ_domain)

if __name__ == "__main__":
    main()