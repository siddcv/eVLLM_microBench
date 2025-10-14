#!/usr/bin/env python3
"""
Custom evaluation script for LoRA-finetuned BioMedCLIP models.
Creates output CSV in the same format as the existing project.
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

class LoRASingleEvaluator:
    """Evaluator for LoRA-finetuned BioMedCLIP models with single task evaluation."""
    
    def __init__(self, config_path: str):
        """Initialize evaluator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup paths
        self.base_dir = Path(self.config['output']['base_dir'])
        self.model_dir = self.base_dir / self.config['output']['model_dir']
        self.results_dir = self.base_dir / self.config['output']['results_dir']
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _normalize_ratio(self, split_ratio: str) -> str:
        try:
            return f"{float(split_ratio):.2f}"
        except Exception:
            return split_ratio
    
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
        """Find the best model checkpoint for a given configuration."""
        ratio_dir = self._normalize_ratio(split_ratio)
        model_path = self.model_dir / organ_domain / task_type / ratio_dir
        
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
    
    def load_evaluation_data(self, organ_domain: str, task_type: str) -> List[Dict]:
        """Load evaluation data for a specific organ domain and task type."""
        # Always load from the new fixed test split
        data_file = f"finetuning_supervised/data/splits/{organ_domain}/{task_type}/fixed/test.json"
        
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return []
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} samples from {data_file}")
        return data

    def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict], task_type: str, organ_domain: str) -> List[Dict]:
        """Evaluate LoRA model on evaluation data."""
        # Load the LoRA model
        model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
        model.load_lora_adapter(adapter_path)
        
        results = []
        
        for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
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
                    'question_type': question_class,  # Changed from 'question_class' to 'question_type'
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
        
        return results
    
    def save_results(self, results: List[Dict], organ_domain: str, task_type: str, split_ratio: str):
        """Save evaluation results to CSV and accuracy metrics to JSON."""
        if not results:
            logger.warning("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Normalize ratio for pathing
        ratio_dir = self._normalize_ratio(split_ratio)
        
        # Save to CSV: include task_type, organ, and split ratio in the path
        out_dir = Path(f"output_results/{task_type}_grained_tasks/BioMedCLIP_LoRA/{organ_domain}/{ratio_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{organ_domain}.csv"
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
        question_types = df['question_type'].unique()  # Changed from 'question_class' to 'question_type'
        for q_type in question_types:
            q_results = df[df['question_type'] == q_type]  # Changed from 'question_class' to 'question_type'
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
            'split_ratio': ratio_dir,
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
        metrics_path = out_dir / 'accuracy.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Accuracy metrics saved to {metrics_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA-finetuned BioMedCLIP model')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--organ_domain', type=str, required=True,
                       help='Organ domain to evaluate')
    parser.add_argument('--task_type', type=str, choices=['coarse', 'fine', 'combined'],
                       help='Task type (coarse, fine, or combined). Not required if --run-all is used.')
    parser.add_argument('--split_ratio', type=str,
                       help='Split ratio used for training. Not required if --run-all is used.')
    parser.add_argument('--run-all', action='store_true',
                       help='Evaluate all models across all task types and split ratios')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.run_all:
        if args.task_type or args.split_ratio:
            logger.warning("--run-all specified: ignoring --task-type and --split-ratio arguments")
    else:
        if not args.task_type or not args.split_ratio:
            parser.error("Either --run-all or both --task-type and --split-ratio must be specified")
    
    # Initialize evaluator
    evaluator = LoRASingleEvaluator(args.config)
    
    if args.run_all:
        # Evaluate all models across all combinations
        task_types = ['coarse', 'fine', 'combined']
        split_ratios = ['0.10', '0.25', '0.50', '0.70']
        
        logger.info(f"Running evaluation for all combinations:")
        logger.info(f"Task types: {task_types}")
        logger.info(f"Split ratios: {split_ratios}")
        logger.info(f"Total combinations: {len(task_types) * len(split_ratios)}")
        
        successful_evaluations = 0
        total_evaluations = len(task_types) * len(split_ratios)
        
        for task_type in task_types:
            for split_ratio in split_ratios:
                logger.info(f"\n{'='*60}")
                logger.info(f"Starting evaluation for {args.organ_domain} - {task_type} - {split_ratio}")
                logger.info(f"{'='*60}")
                
                try:
                    # Find best model
                    adapter_path = evaluator.find_best_model(args.organ_domain, task_type, split_ratio)
                    
                    if not adapter_path:
                        logger.warning(f"No LoRA adapter found for {args.organ_domain} {task_type} {split_ratio}")
                        continue
                    
                    # Load evaluation data
                    eval_data = evaluator.load_evaluation_data(args.organ_domain, task_type)
                    
                    if not eval_data:
                        logger.warning(f"No evaluation data found for {args.organ_domain} {task_type}")
                        continue
                    
                    # Evaluate model
                    results = evaluator.evaluate_lora_model(adapter_path, eval_data, task_type, args.organ_domain)
                    
                    # Save results
                    evaluator.save_results(results, args.organ_domain, task_type, split_ratio)
                    
                    successful_evaluations += 1
                    logger.info(f"Successfully completed evaluation for {task_type} - {split_ratio}")
                    
                except Exception as e:
                    logger.error(f"Failed evaluation for {task_type} - {split_ratio}: {e}")
                    logger.error("Continuing with next combination...")
                    continue
                
                logger.info(f"Completed evaluation for {task_type} - {split_ratio}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"All evaluations completed! Success: {successful_evaluations}/{total_evaluations}")
        logger.info(f"{'='*60}")
    else:
        # Evaluate single model
        # Find best model
        adapter_path = evaluator.find_best_model(args.organ_domain, args.task_type, args.split_ratio)
        
        if not adapter_path:
            logger.error(f"No LoRA adapter found for {args.organ_domain} {args.task_type} {args.split_ratio}")
            return
        
        # Load evaluation data
        eval_data = evaluator.load_evaluation_data(args.organ_domain, args.task_type)
        
        if not eval_data:
            logger.error("No evaluation data found")
            return

        # Evaluate model
        results = evaluator.evaluate_lora_model(adapter_path, eval_data, args.task_type, args.organ_domain)
        
        # Save results (include task_type and split ratio)
        evaluator.save_results(results, args.organ_domain, args.task_type, args.split_ratio)
        
        logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()