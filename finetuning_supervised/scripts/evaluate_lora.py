#!/usr/bin/env python3
"""
Evaluation script for LoRA-finetuned BioMedCLIP models.
Integrates with existing evaluation pipeline and compares base vs fine-tuned models.
"""

import json
import os
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import the LoRA model
from biomedclip_lora import BioMedCLIPLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoRAEvaluator:
    """Evaluator for LoRA-finetuned BioMedCLIP models."""
    
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
        
        # Evaluation parameters
        self.organ_domains = self.config['data']['organ_domains']
        self.task_types = self.config['data']['task_types']
        self.splits = self.config['data']['splits']
        self.question_types = self.config['data']['question_types']
    
    def load_evaluation_data(self, organ_domain: str, task_type: str) -> Dict[str, List[Dict]]:
        """Load evaluation data for a specific organ domain and task type."""
        data_dir = self.base_dir / 'data' / 'splits' / organ_domain / task_type
        
        eval_data = {}
        
        for split_ratio in self.splits:
            split_dir = data_dir / str(split_ratio)
            val_path = split_dir / 'val.json'
            
            if val_path.exists():
                with open(val_path, 'r') as f:
                    eval_data[str(split_ratio)] = json.load(f)
                logger.info(f"Loaded {len(eval_data[str(split_ratio)])} samples for {organ_domain} {task_type} {split_ratio}")
            else:
                logger.warning(f"Validation data not found: {val_path}")
        
        return eval_data
    
    def find_best_model(self, organ_domain: str, task_type: str, split_ratio: str) -> str:
        """Find the best model checkpoint for a given configuration."""
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
    
    def evaluate_base_model(self, eval_data: List[Dict]) -> Dict:
        """Evaluate the base BioMedCLIP model (no LoRA)."""
        logger.info("Evaluating base BioMedCLIP model...")
        
        # Initialize base model
        model = BioMedCLIPLoRA()
        model.load_base_model()
        
        # Run evaluation
        results = model.evaluate_dataset(eval_data)
        
        return results
    
    def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict]) -> Dict:
        """Evaluate a LoRA-finetuned model."""
        logger.info(f"Evaluating LoRA model: {adapter_path}")
        
        # Initialize model with LoRA adapter
        model = BioMedCLIPLoRA()
        model.load_lora_adapter(adapter_path)
        
        # Run evaluation
        results = model.evaluate_dataset(eval_data)
        
        return results
    
    def evaluate_organ_domain(self, organ_domain: str):
        """Evaluate all configurations for a specific organ domain."""
        logger.info(f"Evaluating {organ_domain} domain...")
        
        results = {
            'organ_domain': organ_domain,
            'base_model': {},
            'lora_models': {}
        }
        
        for task_type in self.task_types:
            logger.info(f"Evaluating {task_type}-grained tasks...")
            
            # Load evaluation data
            eval_data = self.load_evaluation_data(organ_domain, task_type)
            
            if not eval_data:
                logger.warning(f"No evaluation data found for {organ_domain} {task_type}")
                continue
            
            # Evaluate base model on first split (for comparison)
            first_split = str(self.splits[0])
            if first_split in eval_data:
                base_results = self.evaluate_base_model(eval_data[first_split])
                results['base_model'][task_type] = base_results
                logger.info(f"Base model accuracy: {base_results['overall_accuracy']:.4f}")
            
            # Evaluate LoRA models for each split
            results['lora_models'][task_type] = {}
            
            for split_ratio in self.splits:
                split_str = str(split_ratio)
                
                if split_str not in eval_data:
                    continue
                
                # Find best LoRA model
                best_adapter = self.find_best_model(organ_domain, task_type, split_str)
                
                if best_adapter:
                    lora_results = self.evaluate_lora_model(best_adapter, eval_data[split_str])
                    results['lora_models'][task_type][split_str] = lora_results
                    logger.info(f"LoRA {split_str} accuracy: {lora_results['overall_accuracy']:.4f}")
                else:
                    logger.warning(f"No LoRA model found for {organ_domain} {task_type} {split_str}")
        
        return results
    
    def save_results(self, results: Dict, organ_domain: str):
        """Save evaluation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.results_dir / f"{organ_domain}_evaluation_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Also save as CSV for easy analysis
        self.save_results_csv(results, organ_domain, timestamp)
    
    def save_results_csv(self, results: Dict, organ_domain: str, timestamp: str):
        """Save results in CSV format for easy analysis."""
        csv_data = []
        
        # Base model results
        for task_type, base_results in results['base_model'].items():
            csv_data.append({
                'organ_domain': organ_domain,
                'task_type': task_type,
                'split_ratio': 'base',
                'model_type': 'base',
                'overall_accuracy': base_results['overall_accuracy'],
                'total_samples': base_results['total_samples'],
                'average_confidence': base_results['average_confidence'],
                'modality_accuracy': base_results['question_type_accuracies'].get('modality', 0),
                'domain_accuracy': base_results['question_type_accuracies'].get('domain', 0),
                'stain_accuracy': base_results['question_type_accuracies'].get('stain', 0),
                'classification_accuracy': base_results['question_type_accuracies'].get('classification', 0),
                'submodality_accuracy': base_results['question_type_accuracies'].get('submodality', 0)
            })
        
        # LoRA model results
        for task_type, split_results in results['lora_models'].items():
            for split_ratio, lora_results in split_results.items():
                csv_data.append({
                    'organ_domain': organ_domain,
                    'task_type': task_type,
                    'split_ratio': split_ratio,
                    'model_type': 'lora',
                    'overall_accuracy': lora_results['overall_accuracy'],
                    'total_samples': lora_results['total_samples'],
                    'average_confidence': lora_results['average_confidence'],
                    'modality_accuracy': lora_results['question_type_accuracies'].get('modality', 0),
                    'domain_accuracy': lora_results['question_type_accuracies'].get('domain', 0),
                    'stain_accuracy': lora_results['question_type_accuracies'].get('stain', 0),
                    'classification_accuracy': lora_results['question_type_accuracies'].get('classification', 0),
                    'submodality_accuracy': lora_results['question_type_accuracies'].get('submodality', 0)
                })
        
        # Save CSV
        df = pd.DataFrame(csv_data)
        csv_path = self.results_dir / f"{organ_domain}_evaluation_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV results saved to {csv_path}")
    
    def generate_summary_report(self):
        """Generate a summary report of all evaluations."""
        logger.info("Generating summary report...")
        
        summary_data = []
        
        # Collect all CSV files
        csv_files = list(self.results_dir.glob("*_evaluation_*.csv"))
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            summary_data.append(df)
        
        if summary_data:
            # Combine all results
            combined_df = pd.concat(summary_data, ignore_index=True)
            
            # Save combined results
            combined_path = self.results_dir / "combined_evaluation_results.csv"
            combined_df.to_csv(combined_path, index=False)
            
            # Generate summary statistics
            summary_stats = {
                'total_evaluations': len(combined_df),
                'organ_domains': combined_df['organ_domain'].unique().tolist(),
                'task_types': combined_df['task_type'].unique().tolist(),
                'split_ratios': combined_df['split_ratio'].unique().tolist(),
                'average_base_accuracy': combined_df[combined_df['model_type'] == 'base']['overall_accuracy'].mean(),
                'average_lora_accuracy': combined_df[combined_df['model_type'] == 'lora']['overall_accuracy'].mean(),
                'best_lora_accuracy': combined_df[combined_df['model_type'] == 'lora']['overall_accuracy'].max(),
                'accuracy_improvement': combined_df[combined_df['model_type'] == 'lora']['overall_accuracy'].mean() - 
                                     combined_df[combined_df['model_type'] == 'base']['overall_accuracy'].mean()
            }
            
            # Save summary
            summary_path = self.results_dir / "evaluation_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            
            logger.info(f"Summary report saved to {summary_path}")
            logger.info(f"Combined results saved to {combined_path}")
            
            # Print summary
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            print(f"Total evaluations: {summary_stats['total_evaluations']}")
            print(f"Organ domains: {', '.join(summary_stats['organ_domains'])}")
            print(f"Average base model accuracy: {summary_stats['average_base_accuracy']:.4f}")
            print(f"Average LoRA model accuracy: {summary_stats['average_lora_accuracy']:.4f}")
            print(f"Best LoRA accuracy: {summary_stats['best_lora_accuracy']:.4f}")
            print(f"Average improvement: {summary_stats['accuracy_improvement']:.4f}")
            print("="*60)
    
    def evaluate_all_domains(self):
        """Evaluate all organ domains."""
        logger.info("Starting evaluation of all organ domains...")
        
        for organ_domain in self.organ_domains:
            try:
                results = self.evaluate_organ_domain(organ_domain)
                self.save_results(results, organ_domain)
            except Exception as e:
                logger.error(f"Error evaluating {organ_domain}: {e}")
                continue
        
        # Generate summary report
        self.generate_summary_report()
        
        logger.info("Evaluation completed!")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA-finetuned BioMedCLIP models')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--organ-domain', type=str, default=None,
                       help='Evaluate specific organ domain (optional)')
    parser.add_argument('--summary-only', action='store_true',
                       help='Generate summary report only')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LoRAEvaluator(args.config)
    
    if args.summary_only:
        # Generate summary only
        evaluator.generate_summary_report()
    elif args.organ_domain:
        # Evaluate specific organ domain
        if args.organ_domain in evaluator.organ_domains:
            results = evaluator.evaluate_organ_domain(args.organ_domain)
            evaluator.save_results(results, args.organ_domain)
        else:
            logger.error(f"Invalid organ domain: {args.organ_domain}")
    else:
        # Evaluate all domains
        evaluator.evaluate_all_domains()

if __name__ == "__main__":
    main() 