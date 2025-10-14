#!/usr/bin/env python3
"""
Soup model evaluation script for BioMedCLIP LoRA models.
Evaluates a soup model (merged LoRA adapter) on test data from any domain.
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
from scipy import stats  # For confidence interval calculations

# Import the LoRA model
from biomedclip_lora import BioMedCLIPLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoupEvaluator:
    """Evaluates soup models (merged LoRA adapters) on test data."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.base_dir = Path(self.config['output']['base_dir'])
        self.results_dir = self.base_dir / self.config['output']['results_dir']
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_confidence_interval(self, correct: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
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
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from test file."""
        test_path = Path(test_file)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_path}")
        
        # Determine file type and load accordingly
        if test_path.suffix == '.json':
            with open(test_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif test_path.suffix == '.jsonl':
            data = []
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {test_path.suffix}")
        
        logger.info(f"Loaded {len(data)} test samples from {test_path}")
        return data
    
    def evaluate_soup_model(self, soup_adapter_path: str, eval_data: List[Dict], task_type: str, test_domain: str) -> List[Dict]:
        """Evaluate soup model on evaluation data."""
        # Convert file path to directory path for PEFT
        soup_adapter_dir = str(Path(soup_adapter_path).parent)
        logger.info(f"Loading soup model from {soup_adapter_dir}")
        
        # Initialize LoRA model
        model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
        model.load_lora_adapter(soup_adapter_dir)
        
        results = []
        
        for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
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
        return results
    
    def save_results(self, results: List[Dict], soup_name: str, test_domain: str, task_type: str, test_file: str):
        """Save evaluation results to CSV and accuracy metrics to JSON."""
        if not results:
            logger.warning("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create output directory structure
        out_dir = Path(f"output_results/soup_evaluation/{soup_name}/{test_domain}_{task_type}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = out_dir / f"{test_domain}_{task_type}_results_{timestamp}.csv"
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
        ci_lower, ci_upper = self.calculate_confidence_interval(correct, total, confidence=0.95)
        
        logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
        logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Per-question-type accuracy
        per_qtype_metrics: Dict[str, Dict[str, Any]] = {}
        question_types = df['question_type'].unique()
        for q_type in question_types:
            q_results = df[df['question_type'] == q_type]
            q_correct = sum(1 for _, row in q_results.iterrows() 
                          if row['correct_idx'] == row['model_answers']['pred'][0])
            q_total = len(q_results)
            q_accuracy = q_correct / q_total if q_total > 0 else 0.0
            
            # Calculate confidence interval for this question type
            q_ci_lower, q_ci_upper = self.calculate_confidence_interval(q_correct, q_total, confidence=0.95)
            
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
            'soup_name': soup_name,
            'test_domain': test_domain,
            'task_type': task_type,
            'test_file': test_file,
            'timestamp': timestamp,
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
        metrics_path = out_dir / f"{test_domain}_{task_type}_metrics_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Accuracy metrics saved to {metrics_path}")
        
        return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate soup model (merged LoRA adapter) on test data')
    parser.add_argument('--soup-adapter', type=str, required=True,
                       help='Path to soup adapter model (adapter_model.safetensors)')
    parser.add_argument('--soup-name', type=str, required=True,
                       help='Name of the soup model (for output organization)')
    parser.add_argument('--test-domain', type=str, required=True,
                       help='Domain of the test data (e.g., cardiovascular, neuropathology)')
    parser.add_argument('--task-type', type=str, required=True, choices=['coarse', 'fine', 'combined'],
                       help='Task type of the test data (coarse, fine, or combined)')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test file (JSON or JSONL)')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SoupEvaluator(args.config)
    
    # Load test data
    logger.info(f"Loading test data from {args.test_file}")
    test_data = evaluator.load_test_data(args.test_file)
    
    # Run evaluation
    logger.info(f"Evaluating soup model: {args.soup_name}")
    logger.info(f"Test domain: {args.test_domain}, Task type: {args.task_type}")
    
    results = evaluator.evaluate_soup_model(
        args.soup_adapter, 
        test_data, 
        args.task_type, 
        args.test_domain
    )
    
    # Save results
    metrics = evaluator.save_results(
        results, 
        args.soup_name, 
        args.test_domain, 
        args.task_type, 
        args.test_file
    )
    
    logger.info("="*60)
    logger.info("Soup evaluation completed!")
    logger.info(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
