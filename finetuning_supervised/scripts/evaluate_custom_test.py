#!/usr/bin/env python3
"""
Custom test set evaluation script for BioMedCLIP LoRA models.
Always uses 0.70 split ratio and accepts custom test file path.
Based on the existing evaluate_lora_single.py framework.
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

class CustomTestEvaluator:
    """Evaluates LoRA models against custom test sets using all available split ratios."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.base_dir = Path(self.config['output']['base_dir'])
        self.model_dir = self.base_dir / self.config['output']['model_dir']
        self.results_dir = self.base_dir / self.config['output']['results_dir']
        
        # Use all available split ratios
        self.split_ratios = ["0.10", "0.25", "0.50", "0.70"]
        
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
    
    def find_best_model(self, organ_domain: str, task_type: str, split_ratio: str) -> str:
        """Find the best model checkpoint for a specific split ratio."""
        model_path = self.model_dir / organ_domain / task_type / split_ratio
        
        if not model_path.exists():
            logger.warning(f"Model directory not found: {model_path}")
            return None
        
        # Look for LoRA adapter directories
        adapter_dirs = [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith('lora_adapters')]
        
        if not adapter_dirs:
            logger.warning(f"No LoRA adapters found in {model_path}")
            return None
        
        # Find the one with highest accuracy (extract from directory name)
        best_adapter = None
        best_acc = 0.0
        
        for adapter_dir in adapter_dirs:
            try:
                # Extract accuracy from directory name (format: lora_adapters_epoch_X_acc_Y)
                acc_str = adapter_dir.name.split('acc_')[-1]
                acc = float(acc_str)
                
                if acc > best_acc:
                    best_acc = acc
                    best_adapter = adapter_dir
            except:
                continue
        
        return str(best_adapter) if best_adapter else None
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from custom test file."""
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
    
    def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict], task_type: str, organ_domain: str) -> List[Dict]:
        """Evaluate LoRA model on evaluation data."""
        logger.info(f"Loading LoRA model from {adapter_path}")
        
        # Initialize LoRA model
        model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
        model.load_lora_adapter(adapter_path)
        
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
    
    def extract_test_path(self, test_file: str) -> str:
        """Extract test path from test file path for output directory naming."""
        test_path = Path(test_file)
        
        # Expected path format: finetuning_supervised/data/splits/{organ_domain}/{task_type}/fixed/test.json
        # We want to extract: {organ_domain}_{task_type}
        path_parts = test_path.parts
        
        try:
            splits_index = path_parts.index('splits')
            if splits_index + 2 < len(path_parts):
                organ_domain = path_parts[splits_index + 1]
                task_type = path_parts[splits_index + 2]
                return f"{organ_domain}_{task_type}"
        except ValueError:
            pass
        
        # Fallback: try to extract from the path string
        path_str = str(test_path)
        if 'splits/' in path_str:
            parts = path_str.split('splits/')[1].split('/')
            if len(parts) >= 2:
                organ_domain = parts[0]
                task_type = parts[1]
                return f"{organ_domain}_{task_type}"
        
        # If all else fails, use a default
        return "custom_test"
    
    def save_results(self, results: List[Dict], organ_domain: str, task_type: str, split_ratio: str, test_file: str):
        """Save evaluation results to CSV and accuracy metrics to JSON."""
        if not results:
            logger.warning("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Extract test path for output directory
        test_path = self.extract_test_path(test_file)
        
        # Save to CSV: include task_type, organ, split ratio, and test path
        out_dir = Path(f"output_results/{task_type}_grained_tasks/BioMedCLIP_LoRA/{organ_domain}/{split_ratio}/{test_path}")
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
        ci_lower, ci_upper = self.calculate_confidence_interval(correct, total, confidence=0.95)
        
        logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
        logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        # Per-question-type accuracy (also collect to save)
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
            'organ_domain': organ_domain,
            'task_type': task_type,
            'split_ratio': split_ratio,
            'test_file': test_file,
            'test_path': test_path,
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
    
    def run_evaluation(self, organ_domain: str, task_type: str, test_file: str):
        """Run the complete evaluation pipeline for all split ratios."""
        logger.info(f"Starting evaluation for {organ_domain} {task_type} against all split ratios")
        
        successful_evaluations = 0
        total_evaluations = len(self.split_ratios)
        
        for split_ratio in self.split_ratios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {organ_domain} {task_type} with split ratio {split_ratio}")
            logger.info(f"{'='*60}")
            
            try:
                # Find best model for this split ratio
                logger.info(f"Finding best model for {organ_domain} {task_type} (split ratio: {split_ratio})")
                adapter_path = self.find_best_model(organ_domain, task_type, split_ratio)
                
                if not adapter_path:
                    logger.warning(f"No LoRA adapter found for {organ_domain} {task_type} {split_ratio}")
                    continue
                
                # Load test data
                logger.info(f"Loading test data from {test_file}")
                test_data = self.load_test_data(test_file)
                
                if not test_data:
                    logger.warning(f"No test data loaded for {organ_domain} {task_type} {split_ratio}")
                    continue
                
                # Run evaluation
                logger.info("Running evaluation...")
                results = self.evaluate_lora_model(adapter_path, test_data, task_type, organ_domain)
                
                if not results:
                    logger.warning(f"No evaluation results generated for {organ_domain} {task_type} {split_ratio}")
                    continue
                
                # Save results
                logger.info("Saving results...")
                self.save_results(results, organ_domain, task_type, split_ratio, test_file)
                
                successful_evaluations += 1
                logger.info(f"Successfully completed evaluation for {organ_domain} {task_type} {split_ratio}")
                
            except Exception as e:
                logger.error(f"Failed evaluation for {organ_domain} {task_type} {split_ratio}: {e}")
                logger.error("Continuing with next split ratio...")
                continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"All evaluations completed! Success: {successful_evaluations}/{total_evaluations}")
        logger.info(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA model on custom test set (runs against all available split ratios)')
    parser.add_argument('--organ-domain', type=str, required=True,
                       help='Organ domain (e.g., cardiovascular, neuropathology) - defines which model to use')
    parser.add_argument('--task-type', type=str, required=True, choices=['coarse', 'fine', 'combined'],
                       help='Task type (coarse, fine, or combined) - defines which model to use')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to custom test file (JSON or JSONL) - the data to test the model against')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = CustomTestEvaluator(args.config)
    
    # Run evaluation using the user-specified organ domain and task type
    evaluator.run_evaluation(args.organ_domain, args.task_type, args.test_file)

if __name__ == "__main__":
    main() 