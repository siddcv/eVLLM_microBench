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
        logger.info(f"Loading LoRA model from {adapter_path}")
        
        # Initialize LoRA model
        model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
        model.load_lora_adapter(adapter_path)
        
        results = []
        
        for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
            # For the new test data format, the data is already flattened
            image_id = data_point.get('image_id', '')
            image_path = data_point.get('image_path', '')  # Use the preprocessed path directly
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
    
    def save_results(self, results: List[Dict], organ_domain: str, task_type: str, split_ratio: str):
        """Save evaluation results in CSV format."""
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
        logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Per-question-type accuracy (also collect to save)
        per_qtype_metrics: Dict[str, Dict[str, float]] = {}
        question_types = df['question_class'].unique()
        for q_type in question_types:
            q_results = df[df['question_class'] == q_type]
            q_correct = sum(1 for _, row in q_results.iterrows() 
                          if row['correct_idx'] == row['model_answers']['pred'][0])
            q_total = len(q_results)
            q_accuracy = q_correct / q_total if q_total > 0 else 0.0
            per_qtype_metrics[q_type] = {
                'accuracy': q_accuracy,
                'correct': q_correct,
                'total': q_total,
            }
            logger.info(f"{q_type} accuracy: {q_accuracy:.4f} ({q_correct}/{q_total})")
        
        # Save accuracy metrics alongside CSV
        metrics = {
            'organ_domain': organ_domain,
            'task_type': task_type,
            'split_ratio': ratio_dir,
            'overall': {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
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
    parser.add_argument('--task_type', type=str, required=True, choices=['coarse', 'fine', 'combined'],
                       help='Task type (coarse, fine, or combined)')
    parser.add_argument('--split_ratio', type=str, required=True,
                       help='Split ratio used for training')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LoRASingleEvaluator(args.config)
    
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