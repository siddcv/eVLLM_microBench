#!/usr/bin/env python3
"""
Evaluate hybrid models (base BioMedCLIP + soup models) on test data.
Uses the same evaluation methodology as soup and baseline evaluations.
"""

import json
import os
import yaml
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from scipy import stats
from typing import Dict, List, Tuple, Any

# Import the LoRA model
from biomedclip_lora import BioMedCLIPLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_confidence_interval(correct: int, total: int, confidence: float = 0.95):
    """Calculate confidence interval for accuracy using Wilson score interval."""
    if total == 0:
        return 0.0, 0.0
    
    if correct == 0:
        # Wilson score interval for 0 successes
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = 0.0
        upper = (z**2) / (total + z**2)
        return lower, upper
    elif correct == total:
        # Wilson score interval for all successes
        z = stats.norm.ppf((1 + confidence) / 2)
        lower = total / (total + z**2)
        upper = 1.0
        return lower, upper
    else:
        # Standard Wilson score interval
        p_hat = correct / total
        z = stats.norm.ppf((1 + confidence) / 2)
        
        denominator = 1 + z**2 / total
        centre_adjustment = z * np.sqrt(p_hat * (1 - p_hat) / total + z**2 / (4 * total**2))
        centre = (p_hat + z**2 / (2 * total)) / denominator
        
        lower = (centre - centre_adjustment) / denominator
        upper = (centre + centre_adjustment) / denominator
        
        return max(0.0, lower), min(1.0, upper)

def extract_test_path(test_file: str) -> str:
    """Extract test path from test file path for output directory naming."""
    test_path = Path(test_file)
    # Extract the domain and task type from the path
    # e.g., "finetuning_supervised/data/splits/cardiovascular/fine/fixed/test.json"
    # becomes "cardiovascular_fine"
    parts = test_path.parts
    if 'splits' in parts:
        splits_idx = parts.index('splits')
        if splits_idx + 2 < len(parts):
            domain = parts[splits_idx + 1]  # cardiovascular
            task_type = parts[splits_idx + 2]  # fine
            return f"{domain}_{task_type}"
    # Fallback: use the filename without extension
    return test_path.stem

def save_results(results: List[Dict], hybrid_domain: str, task_type: str, test_file: str):
    """Save evaluation results to CSV and accuracy metrics to JSON."""
    if not results:
        logger.warning("No results to save")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Extract test path for output directory
    test_path = extract_test_path(test_file)
    
    # Save to CSV: include task_type, hybrid domain, and test path
    out_dir = Path(f"output_results/{task_type}_grained_tasks/Hybrid_Base_Soup/{hybrid_domain}/{test_path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on test path
    csv_path = out_dir / f"{test_path}_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    # Calculate and print accuracy
    correct = 0
    total = 0
    for result in results:
        if result['correct_idx'] == result['model_answers']['pred'][0]:
            correct += 1
        total += 1
    accuracy = correct / total if total > 0 else 0.0
    
    # Calculate 95% confidence interval
    ci_lower, ci_upper = calculate_confidence_interval(correct, total, confidence=0.95)
    logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    # Per-question-type accuracy (also collect to save)
    per_qtype_metrics: Dict[str, Dict[str, Any]] = {}
    question_types = df['question_class'].unique()
    for q_type in question_types:
        q_results = df[df['question_class'] == q_type]
        q_correct = sum(1 for _, row in q_results.iterrows()
                      if row['correct_idx'] == row['model_answers']['pred'][0])
        q_total = len(q_results)
        q_accuracy = q_correct / q_total if q_total > 0 else 0.0
        
        # Calculate confidence interval for this question type
        q_ci_lower, q_ci_upper = calculate_confidence_interval(q_correct, q_total, confidence=0.95)
        
        per_qtype_metrics[q_type] = {
            'accuracy': q_accuracy,
            'correct': q_correct,
            'total': q_total,
            'confidence_interval_95': {
                'lower': q_ci_lower,
                'upper': q_ci_upper
            }
        }
        logger.info(f"{q_type} accuracy: {q_accuracy:.4f} ({q_correct}/{q_total})")
        logger.info(f"{q_type} 95% CI: [{q_ci_lower:.4f}, {q_ci_upper:.4f}]")
    
    # Save accuracy metrics alongside CSV
    metrics = {
        'hybrid_domain': hybrid_domain,
        'task_type': task_type,
        'test_file': test_file,
        'test_path': test_path,
        'model_type': 'hybrid_base_soup',
        'overall': {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'confidence_interval_95': {
                'lower': ci_lower,
                'upper': ci_upper
            }
        },
        'per_question_type': per_qtype_metrics,
    }
    metrics_path = out_dir / f"{test_path}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Accuracy metrics saved to {metrics_path}")

def evaluate_hybrid_model(hybrid_adapter_path: str, test_file: str, hybrid_domain: str, task_type: str):
    """Evaluate hybrid model (base BioMedCLIP + soup) on test data."""
    
    # Load test data
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    logger.info(f"Loaded {len(test_data)} test samples from {test_file}")
    
    # Initialize LoRA model and load hybrid adapter
    model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
    
    # Convert file path to directory path for PEFT
    hybrid_adapter_dir = str(Path(hybrid_adapter_path).parent)
    logger.info(f"Loading hybrid model from {hybrid_adapter_dir}")
    
    try:
        model.load_lora_adapter(hybrid_adapter_dir)
        logger.info("Hybrid model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load hybrid model: {e}")
        raise
    
    results = []
    
    for data_point in tqdm(test_data, desc=f"Evaluating {task_type}"):
        # Extract data
        image_id = data_point.get('image_id', '')
        image_path = data_point.get('image_path', '')
        question = data_point.get('question', '')
        answer_options = data_point.get('answer_options', [])
        correct_answer_idx = data_point.get('correct_answer_idx', 0)
        question_type = data_point.get('question_type', '')
        
        if not image_path or not question or not answer_options:
            logger.warning(f"Skipping sample {image_id}: missing required fields")
            continue
        
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
                'question_type': question_type,
                'questions': str([f"{question} {option}" for option in answer_options]),
                'image_id': image_id,
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
            logger.error(f"Error predicting for {image_id}: {e}")
            continue
    
    logger.info(f"Total results generated: {len(results)}")
    
    # Save results using the enhanced save_results function
    save_results(results, hybrid_domain, task_type, test_file)

def main():
    parser = argparse.ArgumentParser(description='Evaluate hybrid models (base BioMedCLIP + soup) on test data')
    parser.add_argument('--hybrid-adapter', type=str, required=True,
                       help='Path to hybrid adapter directory')
    parser.add_argument('--hybrid-domain', type=str, required=True,
                       help='Domain name for the hybrid model (e.g., cardiovascular, neuropathology)')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test file (JSON)')
    parser.add_argument('--task-type', type=str, required=True, choices=['coarse', 'fine', 'combined'],
                       help='Task type of the test data (coarse, fine, or combined)')
    
    args = parser.parse_args()
    
    # Run evaluation
    logger.info("Evaluating hybrid model (base BioMedCLIP + soup)")
    logger.info(f"Hybrid domain: {args.hybrid_domain}, Task type: {args.task_type}")
    logger.info(f"Test file: {args.test_file}")
    
    evaluate_hybrid_model(
        hybrid_adapter_path=args.hybrid_adapter,
        test_file=args.test_file,
        hybrid_domain=args.hybrid_domain,
        task_type=args.task_type
    )
    
    logger.info("="*60)
    logger.info("Hybrid model evaluation completed!")
    logger.info("="*60)

if __name__ == "__main__":
    main()
