#!/usr/bin/env python3
"""
Evaluate LoRA models across all model types (alignment, text_encoder, vision_encoder)
for a given organ domain and custom test file.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import yaml
from tqdm import tqdm
import numpy as np
from scipy import stats  # For confidence interval calculations

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# from evlm.models.openCLIP_models.biomedclip import BioMedCLIP
from biomedclip_lora import BioMedCLIPLoRA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AllModelTypesEvaluator:
    """Evaluator that runs inference against all model types for a given organ domain."""
    
    def __init__(self, config_path: str):
        """Initialize the evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.base_dir = Path(self.config['output']['base_dir'])
        self.models_dir = self.base_dir / self.config['output']['model_dir']
        
        # Define model types to evaluate
        self.model_types = ['alignment', 'text_encoder', 'vision_encoder']
        
        # Define output directory structure - will be set dynamically based on task type
        self.output_base = None  # Will be set when we know the task type
        
        logger.info(f"Initialized evaluator with models directory: {self.models_dir}")
        logger.info(f"Model types to evaluate: {self.model_types}")
    
    def calculate_confidence_interval(self, correct: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
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
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def find_best_model(self, organ_domain: str, model_type: str) -> Optional[Path]:
        """Find the best model checkpoint for a given configuration."""
        model_dir = self.models_dir / organ_domain / model_type
        
        if not model_dir.exists():
            logger.warning(f"Model directory not found: {model_dir}")
            return None
        
        # Look for LoRA adapter directories
        adapter_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith('lora_adapters')]
        
        if not adapter_dirs:
            logger.warning(f"No LoRA adapter directories found in {model_dir}")
            return None
        
        # Find the one with highest accuracy
        best_adapter = None
        best_acc = -1
        
        for adapter_dir in adapter_dirs:
            try:
                # Extract accuracy from directory name (e.g., "lora_adapters_epoch_2_acc_0.9307")
                acc_part = adapter_dir.name.split('_acc_')[-1]
                acc = float(acc_part)
                
                if acc > best_acc:
                    best_acc = acc
                    best_adapter = adapter_dir
                    
            except (ValueError, IndexError):
                logger.warning(f"Could not parse accuracy from directory name: {adapter_dir.name}")
                continue
        
        if best_adapter:
            logger.info(f"Found best model for {organ_domain}/{model_type}: {best_adapter.name} (acc: {best_acc:.4f})")
        
        return best_adapter
    
    def extract_task_type(self, test_file: str) -> str:
        """Extract task type from test file path."""
        test_path = Path(test_file)
        
        # Expected path format: finetuning_supervised/data/splits/{organ_domain}/{task_type}/fixed/test.json
        # We want to extract: {task_type}
        path_parts = test_path.parts
        
        try:
            splits_index = path_parts.index('splits')
            if splits_index + 2 < len(path_parts):
                task_type = path_parts[splits_index + 2]
                return task_type
        except ValueError:
            pass
        
        # Fallback: try to extract from the path string
        path_str = str(test_path)
        if 'splits/' in path_str:
            parts = path_str.split('splits/')[1].split('/')
            if len(parts) >= 2:
                task_type = parts[1]
                return task_type
        
        # If all else fails, use a default
        return "combined"
    
    def load_test_data(self, test_file: str) -> List[Dict]:
        """Load test data from JSON/JSONL file."""
        test_path = Path(test_file)
        
        if not test_path.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        if test_path.suffix == '.json':
            with open(test_path, 'r') as f:
                return json.load(f)
        elif test_path.suffix == '.jsonl':
            data = []
            with open(test_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {test_path.suffix}. Use .json or .jsonl")
    
    def evaluate_model(self, model_path: Path, test_data: List[Dict], 
                      organ_domain: str, model_type: str) -> Dict:
        """Evaluate a single model on the test data."""
        logger.info(f"Evaluating {model_type} model")
        
        try:
            # Load the LoRA model
            model = BioMedCLIPLoRA()
            model.load_lora_adapter(str(model_path))
            
            results = []
            correct = 0
            total = 0
            
            # Run inference on test data
            for item in tqdm(test_data, desc=f"Evaluating {model_type}"):
                try:
                    # Extract data
                    image_path = item.get('image_path')
                    question = item.get('question')
                    answer_options = item.get('answer_options', [])
                    correct_answer_idx = item.get('correct_answer_idx')
                    question_type = item.get('question_type', 'unknown')
                    
                    if not all([image_path, question, answer_options, correct_answer_idx is not None]):
                        logger.warning(f"Skipping item with missing data: {item}")
                        continue
                    
                    # Run prediction
                    prediction = model.predict_single(image_path, question, answer_options)
                    predicted_answer = prediction['predicted_answer']
                    confidence = prediction.get('confidence', 0.0)
                    
                    # Convert predicted answer to index for comparison
                    try:
                        predicted_idx = answer_options.index(predicted_answer)
                    except ValueError:
                        # If predicted answer not in options, mark as incorrect
                        predicted_idx = -1
                    
                    # Check if correct
                    is_correct = predicted_idx == correct_answer_idx
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    # Store result
                    result = {
                        'image_path': image_path,
                        'question': question,
                        'answer_options': answer_options,
                        'correct_answer_idx': correct_answer_idx,
                        'correct_answer': answer_options[correct_answer_idx] if correct_answer_idx < len(answer_options) else 'unknown',
                        'predicted_answer': predicted_answer,
                        'predicted_idx': predicted_idx,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'question_type': question_type,
                        'model_type': model_type
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    continue
            
            # Calculate overall accuracy
            accuracy = correct / total if total > 0 else 0.0
            
            # Calculate per-question-type accuracy
            per_qtype_metrics = {}
            for result in results:
                q_type = result['question_type']
                if q_type not in per_qtype_metrics:
                    per_qtype_metrics[q_type] = {'correct': 0, 'total': 0}
                
                per_qtype_metrics[q_type]['total'] += 1
                if result['is_correct']:
                    per_qtype_metrics[q_type]['correct'] += 1
            
            # Calculate per-question-type accuracy
            for q_type in per_qtype_metrics:
                q_total = per_qtype_metrics[q_type]['total']
                q_correct = per_qtype_metrics[q_type]['correct']
                per_qtype_metrics[q_type]['accuracy'] = q_correct / q_total if q_total > 0 else 0.0
            
            # Calculate 95% confidence interval for overall accuracy
            ci_lower, ci_upper = self.calculate_confidence_interval(correct, total, confidence=0.95)
            
            logger.info(f"{model_type} evaluation completed: {correct}/{total} correct ({accuracy:.4f})")
            logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            # Add confidence intervals to per-question-type metrics
            for q_type in per_qtype_metrics:
                q_correct = per_qtype_metrics[q_type]['correct']
                q_total = per_qtype_metrics[q_type]['total']
                q_ci_lower, q_ci_upper = self.calculate_confidence_interval(q_correct, q_total, confidence=0.95)
                per_qtype_metrics[q_type]['confidence_interval_95'] = {
                    'lower': q_ci_lower,
                    'upper': q_ci_upper
                }
                logger.info(f"{q_type} accuracy: {per_qtype_metrics[q_type]['accuracy']:.4f} ({q_correct}/{q_total})")
                logger.info(f"{q_type} 95% CI: [{q_ci_lower:.4f}, {q_ci_upper:.4f}]")
            
            return {
                'results': results,
                'overall_accuracy': accuracy,
                'correct': correct,
                'total': total,
                'per_question_type': per_qtype_metrics,
                'confidence_interval_95': {
                    'lower': ci_lower,
                    'upper': ci_upper
                }
            }
            
        except Exception as e:
            logger.error(f"Error evaluating {model_type} model: {e}")
            return None
    
    def save_results(self, results: Dict, organ_domain: str, model_type: str, test_file: str):
        """Save evaluation results to CSV and metrics files."""
        if not results:
            logger.warning(f"No results to save for {model_type}")
            return
        
        # Create output directory
        test_path = Path(test_file).stem
        out_dir = self.output_base / organ_domain / model_type
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CSV results
        df = pd.DataFrame(results['results'])
        csv_path = out_dir / f"{test_path}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        # Save metrics
        metrics = {
            'organ_domain': organ_domain,
            'model_type': model_type,
            'test_file': test_file,
            'test_path': test_path,
            'overall': {
                'accuracy': results['overall_accuracy'],
                'correct': results['correct'],
                'total': results['total'],
                'confidence_interval_95': results['confidence_interval_95']
            },
            'per_question_type': results['per_question_type']
        }
        
        metrics_path = out_dir / f"{test_path}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    def run_evaluation(self, organ_domain: str, test_file: str):
        """Run evaluation against all model types."""
        # Extract task type and set output directory
        task_type = self.extract_task_type(test_file)
        self.output_base = Path(f"output_results/{task_type}_grained_tasks/BioMedCLIP_LoRA")
        
        logger.info(f"Starting evaluation for {organ_domain} {task_type} against all model types")
        logger.info(f"Output directory: {self.output_base}")
        
        # Load test data once
        test_data = self.load_test_data(test_file)
        logger.info(f"Loaded {len(test_data)} test samples")
        
        successful_evaluations = 0
        total_evaluations = len(self.model_types)
        
        for model_type in self.model_types:
            logger.info(f"\n{'='*80}")
            logger.info(f"Evaluating {model_type} model for {organ_domain}")
            logger.info(f"{'='*80}")
            
            try:
                # Find best model
                model_path = self.find_best_model(organ_domain, model_type)
                
                if not model_path:
                    logger.warning(f"No model found for {organ_domain}/{model_type}")
                    continue
                
                # Run evaluation
                results = self.evaluate_model(model_path, test_data, organ_domain, model_type)
                
                if results:
                    # Save results
                    self.save_results(results, organ_domain, model_type, test_file)
                    successful_evaluations += 1
                    logger.info(f"Successfully completed {organ_domain}/{model_type}")
                else:
                    logger.warning(f"Evaluation failed for {organ_domain}/{model_type}")
            
            except Exception as e:
                logger.error(f"Error processing {organ_domain}/{model_type}: {e}")
                continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"All evaluations completed! Success: {successful_evaluations}/{total_evaluations}")
        logger.info(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA models across all model types for a given organ domain')
    parser.add_argument('--organ-domain', type=str, required=True,
                       help='Organ domain (e.g., cardiovascular, neuropathology)')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to custom test file (JSON or JSONL)')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = AllModelTypesEvaluator(args.config)
    
    # Run evaluation
    evaluator.run_evaluation(args.organ_domain, args.test_file)

if __name__ == "__main__":
    main() 