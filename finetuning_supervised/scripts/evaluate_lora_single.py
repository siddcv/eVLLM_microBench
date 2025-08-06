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
    
    def load_evaluation_data(self, organ_domain: str, task_type: str) -> List[Dict]:
        """Load evaluation data for a specific organ domain and task type."""
        # Load from the original test data
        if task_type == 'coarse':
            data_file = f"organData/coarseGrain/{organ_domain}/test_200.jsonl"
        else:  # fine
            data_file = f"organData/fineGrain/{organ_domain}/test_200.jsonl"
        
        if not os.path.exists(data_file):
            logger.error(f"Data file not found: {data_file}")
            return []
        
        data = []
        with open(data_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} samples from {data_file}")
        return data
    
    # def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict], task_type: str, organ_domain: str) -> List[Dict]:
    #     """Evaluate LoRA model on evaluation data."""
    #     logger.info(f"Loading LoRA model from {adapter_path}")
        
    #     # Initialize LoRA model
    #     model = BioMedCLIPLoRA(adapter_path)
        
    #     results = []
        
    #     for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
    #         # Extract metadata
    #         metadata = data_point.get('metadata', {})
    #         image_id = metadata.get('image_id', '')
    #         image_path = metadata.get('image', '')
    #         workspace = '/workspace/eVLLM_Sidd/eVLLM_microBench'
    #         image_file = metadata.get('image', '')
    #         image_path = f"{workspace}/organData/{task_type}Grain/{organ_domain}/images/{image_file}"
    #         print(image_path)

    #         # Fix image path
    #         if 'organData/coarse/' in image_path:
    #             image_path = image_path.replace('organData/coarse/', 'organData/coarseGrain/')
    #             image_path = image_path.replace('/images/', '/')
    #         elif 'organData/fine/' in image_path:
    #             image_path = image_path.replace('organData/fine/', 'organData/fineGrain/')
    #             image_path = image_path.replace('/images/', '/')
            
    #         if not os.path.exists(image_path):
    #             logger.warning(f"Image not found: {image_path}")
    #             continue
            
    #         # Extract questions based on task type
    #         if task_type == 'coarse':
    #             questions = data_point.get('captions', [])
    #         else:  # fine
    #             questions = data_point.get('custom_metadata', {})
            
    #         # Process each question
    #         for question_data in questions:
    #             if task_type == 'coarse':
    #                 # Coarse-grained questions
    #                 if isinstance(question_data, dict):
    #                     question = question_data.get('question', '')
    #                     answer_options = question_data.get('answers', [])
    #                     correct_answer = question_data.get('correct_answer', '')
    #                     question_class = question_data.get('question_type', '')
                        
    #                     if not question or not answer_options or not correct_answer:
    #                         continue
                        
    #                     # Get model prediction
    #                     try:
    #                         predicted_idx, confidence, scores = model.predict(
    #                             image_path, question, answer_options
    #                         )
                            
    #                         # Create result entry
    #                         result = {
    #                             'question_class': question_class,
    #                             'questions': str([f"{question} {option}" for option in answer_options]),
    #                             'image_id': image_id,
    #                             'correct_answer': correct_answer,
    #                             'correct_idx': answer_options.index(correct_answer),
    #                             'model_answers': {
    #                                 'pred': [predicted_idx],
    #                                 'probs': [scores.tolist()],
    #                                 'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
    #                             },
    #                             'microns_per_pixel': metadata.get('microns_per_pixel', 2.0),
    #                             'domain': metadata.get('domain', ''),
    #                             'subdomain': metadata.get('subdomain', ''),
    #                             'modality': metadata.get('modality', ''),
    #                             'submodality': metadata.get('submodality', ''),
    #                             'normal_or_abnormal': metadata.get('normal_or_abnormal', '')
    #                         }
                            
    #                         results.append(result)
                            
    #                     except Exception as e:
    #                         logger.error(f"Error predicting for {image_id}: {e}")
    #                         continue
                
    #             else:
    #                 # Fine-grained questions
    #                 for question_class, q_data in question_data.items():
    #                     if isinstance(q_data, dict):
    #                         question = q_data.get('question', '')
    #                         answer_options = q_data.get('answers', [])
    #                         correct_answer = q_data.get('correct_answer', '')
                            
    #                         if not question or not answer_options or not correct_answer:
    #                             continue
                            
    #                         # Get model prediction
    #                         try:
    #                             predicted_idx, confidence, scores = model.predict(
    #                                 image_path, question, answer_options
    #                             )
                                
    #                             # Create result entry
    #                             result = {
    #                                 'question_class': question_class,
    #                                 'questions': str([f"{question} {option}" for option in answer_options]),
    #                                 'image_id': image_id,
    #                                 'correct_answer': correct_answer,
    #                                 'correct_idx': answer_options.index(correct_answer),
    #                                 'model_answers': {
    #                                     'pred': [predicted_idx],
    #                                     'probs': [scores.tolist()],
    #                                     'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
    #                                 },
    #                                 'microns_per_pixel': metadata.get('microns_per_pixel', 2.0),
    #                                 'domain': metadata.get('domain', ''),
    #                                 'subdomain': metadata.get('subdomain', ''),
    #                                 'modality': metadata.get('modality', ''),
    #                                 'submodality': metadata.get('submodality', ''),
    #                                 'normal_or_abnormal': metadata.get('normal_or_abnormal', '')
    #                             }
                                
    #                             results.append(result)
                                
    #                         except Exception as e:
    #                             logger.error(f"Error predicting for {image_id}: {e}")
    #                             continue
        
    #     return results

    def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict], task_type: str, organ_domain: str) -> List[Dict]:
        """Evaluate LoRA model on evaluation data."""
        logger.info(f"Loading LoRA model from {adapter_path}")
        
        # Initialize LoRA model
        # model = BioMedCLIPLoRA(adapter_path)
        model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
        model.load_lora_adapter(adapter_path)
        
        results = []
        
        for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
            # Extract metadata
            metadata = data_point.get('metadata', {})
            image_id = metadata.get('image_id', '')
            image_file = metadata.get('image', '')
            workspace = '/workspace/eVLLM_Sidd/eVLLM_microBench'
            image_path = f"{workspace}/organData/{task_type}Grain/{organ_domain}/images/{image_file}"
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            custom_metadata = data_point.get('custom_metadata', {})
            question_dict = custom_metadata.get('questions', {})  # This is a dict of question_name → question_dict
            # Extract questions based on task type
            if task_type == 'coarse':
                # For coarse tasks, get all questions from custom_metadata except classification
                questions = [
                    q for q in question_dict.values()
                    if isinstance(q, dict) and q.get('question_type', '') != 'classification'
                ]
                # print(questions)
                print("333333333333333333333333333333333")
            else:
                # For fine tasks, questions are in custom_metadata
                questions = [
                    q for q in question_dict.values()
                    if isinstance(q, dict) and q.get('question_type', '') == 'classification'
                ]
            
            logger.info(f"Found {len(questions)} questions for image {image_id}")
            
            # Process each question
            for i, question_data in enumerate(questions):
                logger.info(f"Processing question {i+1}/{len(questions)} for image {image_id}")
                
                if isinstance(question_data, dict):
                    question = question_data.get('question', '')
                    answer_options = question_data.get('options', [])
                    correct_answer = question_data.get('answer', '')
                    question_class = question_data.get('name', '')  # Optional
            
                    logger.info(f"  Question: {question[:50]}...")
                    logger.info(f"  Answer options: {answer_options}")
                    logger.info(f"  Correct answer: {correct_answer}")
                    logger.info(f"  Question class: {question_class}")
                    
                    # Check each field individually
                    if not question:
                        logger.warning(f"  Skipping: question is empty")
                        continue
                    if not answer_options:
                        logger.warning(f"  Skipping: answer_options is empty")
                        continue
                    if not correct_answer:
                        logger.warning(f"  Skipping: correct_answer is empty")
                        continue
                    
                    # # Get model prediction
                    # try:
                    #     logger.info(f"  Calling model.predict...")
                    #     predicted_idx, confidence, scores = model.predict_single(
                    #         image_path, question, answer_options
                    #     )
                    #     logger.info(f"  Prediction successful: idx={predicted_idx}, confidence={confidence}")
                        
                    #     # Create result entry
                    #     result = {
                    #         'question_class': question_class,
                    #         'questions': str([f"{question} {option}" for option in answer_options]),
                    #         'image_id': image_id,
                    #         'correct_answer': correct_answer,
                    #         'correct_idx': answer_options.index(correct_answer),
                    #         'model_answers': {
                    #             'pred': [predicted_idx],
                    #             'probs': [scores.tolist()],
                    #             'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
                    #         },
                    #         'microns_per_pixel': metadata.get('microns_per_pixel', 2.0),
                    #         'domain': metadata.get('domain', ''),
                    #         'subdomain': metadata.get('subdomain', ''),
                    #         'modality': metadata.get('modality', ''),
                    #         'submodality': metadata.get('submodality', ''),
                    #         'normal_or_abnormal': metadata.get('normal_or_abnormal', '')
                    #     }
                        
                    #     results.append(result)
                    #     logger.info(f"  Added result for {question_class}")
                        
                    # except Exception as e:
                    #     logger.error(f"  Error predicting for {image_id}: {e}")
                    #     continue
                    # Get model prediction using predict_single method
                    try:
                        logger.info(f"  Calling model.predict_single...")
                        prediction_result = model.predict_single(image_path, question, answer_options)
                        
                        predicted_idx = prediction_result['predicted_idx']
                        confidence = prediction_result['confidence']
                        scores = prediction_result['all_probs']
                        
                        logger.info(f"  Prediction successful: idx={predicted_idx}, confidence={confidence}")
                        
                        # Create result entry
                        result = {
                            'question_class': question_class,
                            'questions': str([f"{question} {option}" for option in answer_options]),
                            'image_id': image_id,
                            'correct_answer': correct_answer,
                            'correct_idx': answer_options.index(correct_answer),
                            'model_answers': {
                                'pred': [predicted_idx],
                                'probs': [scores],
                                'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
                            },
                            'microns_per_pixel': metadata.get('microns_per_pixel', 2.0),
                            'domain': metadata.get('domain', ''),
                            'subdomain': metadata.get('subdomain', ''),
                            'modality': metadata.get('modality', ''),
                            'submodality': metadata.get('submodality', ''),
                            'normal_or_abnormal': metadata.get('normal_or_abnormal', '')
                        }
                        
                        results.append(result)
                        logger.info(f"  Added result for {question_class}")
                        
                    except Exception as e:
                        logger.error(f"  Error predicting for {image_id}: {e}")
                        continue
                else:
                    logger.warning(f"  Question data is not a dict: {type(question_data)}")
        
        logger.info(f"Total results generated: {len(results)}")
        return results

    # def evaluate_lora_model(self, adapter_path: str, eval_data: List[Dict], task_type: str, organ_domain: str) -> List[Dict]:
    #     """Evaluate LoRA model on evaluation data."""
    #     logger.info(f"Loading LoRA model from {adapter_path}")
        
    #     # Initialize LoRA model
    #     model = BioMedCLIPLoRA(adapter_path)
        
    #     results = []
        
    #     for data_point in tqdm(eval_data, desc=f"Evaluating {task_type}"):
    #         # Extract metadata
    #         metadata = data_point.get('metadata', {})
    #         image_id = metadata.get('image_id', '')
    #         image_file = metadata.get('image', '')
    #         workspace = '/workspace/eVLLM_Sidd/eVLLM_microBench'
    #         image_path = f"{workspace}/organData/{task_type}Grain/{organ_domain}/images/{image_file}"
    #         custom_metadata = data_point.get('custom_metadata', {})
            
    #         if not os.path.exists(image_path):
    #             logger.warning(f"Image not found: {image_path}")
    #             continue
            
    #         # Extract questions from metadata
    #         questions = custom_metadata.get('questions', [])
    #         logger.info(f"Found {len(questions)} questions for image {image_id}")
            
    #         # Process each question
    #         for question_data in questions:
    #             if isinstance(question_data, dict):
    #                 question = question_data.get('question', '')
    #                 answer_options = question_data.get('answers', [])
    #                 correct_answer = question_data.get('correct_answer', '')
    #                 question_class = question_data.get('question_type', '')
                    
    #                 logger.info(f"Processing question: {question[:50]}...")
    #                 logger.info(f"Answer options: {answer_options}")
    #                 logger.info(f"Correct answer: {correct_answer}")
    #                 logger.info(f"Question class: {question_class}")
                    
    #                 if not question or not answer_options or not correct_answer:
    #                     logger.warning(f"Skipping question due to missing data")
    #                     continue
                    
    #                 # Get model prediction
    #                 try:
    #                     predicted_idx, confidence, scores = model.predict(
    #                         image_path, question, answer_options
    #                     )
                        
    #                     # Create result entry
    #                     result = {
    #                         'question_class': question_class,
    #                         'questions': str([f"{question} {option}" for option in answer_options]),
    #                         'image_id': image_id,
    #                         'correct_answer': correct_answer,
    #                         'correct_idx': answer_options.index(correct_answer),
    #                         'model_answers': {
    #                             'pred': [predicted_idx],
    #                             'probs': [scores.tolist()],
    #                             'pred_prompt': [f"{question} {answer_options[predicted_idx]}"]
    #                         },
    #                         'microns_per_pixel': metadata.get('microns_per_pixel', 2.0),
    #                         'domain': metadata.get('domain', ''),
    #                         'subdomain': metadata.get('subdomain', ''),
    #                         'modality': metadata.get('modality', ''),
    #                         'submodality': metadata.get('submodality', ''),
    #                         'normal_or_abnormal': metadata.get('normal_or_abnormal', '')
    #                     }
                        
    #                     results.append(result)
    #                     logger.info(f"Added result for {question_class}")
                        
    #                 except Exception as e:
    #                     logger.error(f"Error predicting for {image_id}: {e}")
    #                     continue
        
    #     logger.info(f"Total results generated: {len(results)}")
    #     return results
    
    def save_results(self, results: List[Dict], organ_domain: str, task_type: str, split_ratio: str):
        """Save evaluation results in CSV format."""
        if not results:
            logger.warning("No results to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f"output_results/{task_type}_grained_tasks/BioMedCLIP_LoRA")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = output_dir / f"{organ_domain}.csv"
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
        
        # Per-question-type accuracy
        question_types = df['question_class'].unique()
        for q_type in question_types:
            q_results = df[df['question_class'] == q_type]
            q_correct = sum(1 for _, row in q_results.iterrows() 
                          if row['correct_idx'] == row['model_answers']['pred'][0])
            q_total = len(q_results)
            q_accuracy = q_correct / q_total if q_total > 0 else 0.0
            logger.info(f"{q_type} accuracy: {q_accuracy:.4f} ({q_correct}/{q_total})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate LoRA-finetuned BioMedCLIP model')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--organ_domain', type=str, required=True,
                       help='Organ domain to evaluate')
    parser.add_argument('--task_type', type=str, required=True, choices=['coarse', 'fine'],
                       help='Task type (coarse or fine)')
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

    # Add this after loading eval_data
    logger.info(f"Sample data point keys: {list(eval_data[0].keys())}")
    logger.info(f"Sample metadata: {eval_data[0].get('custom_metadata', {})}")
    # if args.task_type == 'coarse':
    #     # logger.info(f"Sample captions: {eval_data[0].get('custom_metadata'.get('questions',{}), [])[:2]}")  # Show first 2
    #     logger.info(f"Sample captions: {eval_data[0].get('custom_metadata', {}).get('questions', [])[:2]}")

    # else:
    #     logger.info(f"Sample custom_metadata: {eval_data[0].get('custom_metadata', {})}")
    
    # Evaluate model
    results = evaluator.evaluate_lora_model(adapter_path, eval_data, args.task_type, args.organ_domain)
    
    # Save results
    evaluator.save_results(results, args.organ_domain, args.task_type, args.split_ratio)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()