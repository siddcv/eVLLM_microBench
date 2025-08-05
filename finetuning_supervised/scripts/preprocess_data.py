#!/usr/bin/env python3
"""
Data preprocessing script for BioMedCLIP LoRA fine-tuning.
Converts JSONL format to training format with stratified splits.
"""

import json
import os
import random
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handles data preprocessing for BioMedCLIP fine-tuning."""
    
    def __init__(self, config_path: str):
        """Initialize preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.random_seed = self.config['data']['random_seed']
        self.splits = self.config['data']['splits']
        self.organ_domains = self.config['data']['organ_domains']
        self.task_types = self.config['data']['task_types']
        self.question_types = self.config['data']['question_types']
        
        # Set random seed for reproducibility
        random.seed(self.random_seed)
        
        # Setup paths
        self.base_dir = Path(self.config['output']['base_dir'])
        self.data_dir = self.base_dir / 'data'
        self.processed_dir = self.data_dir / 'processed'
        self.splits_dir = self.data_dir / 'splits'
        
        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
    
    def load_jsonl_data(self, file_path: str) -> List[Dict]:
        """Load data from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data
    
    def flatten_data(self, data: List[Dict], organ_domain: str, task_type: str) -> List[Dict]:
        """Flatten nested JSONL data to training format."""
        flattened_data = []
        
        for item in data:
            image_id = item.get('image_id', '')
            image_path = item.get('image', '')
            
            # Handle both coarse and fine-grained formats
            if 'captions' in item:
                # Coarse-grained format
                captions = item['captions']
                for question_type in self.question_types:
                    if f"{question_type}_0" in captions:
                        q_data = captions[f"{question_type}_0"]
                        
                        sample = {
                            'image_id': image_id,
                            'image_path': image_path,
                            'question': q_data['question'],
                            'answer_options': q_data['options'],
                            'correct_answer_idx': int(q_data['answer_idx']),
                            'question_type': question_type,
                            'organ_domain': organ_domain,
                            'task_type': task_type
                        }
                        flattened_data.append(sample)
            
            elif 'custom_metadata' in item and 'questions' in item['custom_metadata']:
                # Fine-grained format
                questions = item['custom_metadata']['questions']
                for question_type, q_data in questions.items():
                    sample = {
                        'image_id': image_id,
                        'image_path': image_path,
                        'question': q_data.get('question', ''),
                        'answer_options': q_data.get('options', []),
                        'correct_answer_idx': int(q_data.get('answer_idx', 0)),
                        'question_type': question_type,
                        'organ_domain': organ_domain,
                        'task_type': task_type
                    }
                    flattened_data.append(sample)
        
        return flattened_data
    
    def create_stratified_splits(self, data: List[Dict], split_ratio: float) -> Tuple[List[Dict], List[Dict]]:
        """Create stratified train/validation split based on question type."""
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Create stratified split based on question_type
        train_data, val_data = train_test_split(
            df, 
            test_size=1-split_ratio, 
            random_state=self.random_seed,
            stratify=df['question_type']
        )
        
        return train_data.to_dict('records'), val_data.to_dict('records')
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """Save processed data to JSON file."""
        output_path = self.processed_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}")
    
    def save_split_data(self, train_data: List[Dict], val_data: List[Dict], 
                       organ_domain: str, task_type: str, split_ratio: str):
        """Save train/validation splits."""
        split_dir = self.splits_dir / organ_domain / task_type / split_ratio
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Save train data
        train_path = split_dir / 'train.json'
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        # Save validation data
        val_path = split_dir / 'val.json'
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved splits to {split_dir}")
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    def process_organ_domain(self, organ_domain: str):
        """Process data for a specific organ domain."""
        logger.info(f"Processing {organ_domain} domain...")
        
        for task_type in self.task_types:
            logger.info(f"Processing {task_type}-grained tasks...")
            
            # Define input file path
            if task_type == 'coarse':
                input_file = f"organData/coarseGrain/{organ_domain}/test_200.jsonl"
            else:
                input_file = f"organData/fineGrain/{organ_domain}/test_200.jsonl"
            
            if not os.path.exists(input_file):
                logger.warning(f"File not found: {input_file}")
                continue
            
            # Load and flatten data
            raw_data = self.load_jsonl_data(input_file)
            flattened_data = self.flatten_data(raw_data, organ_domain, task_type)
            
            logger.info(f"Flattened {len(flattened_data)} samples for {organ_domain} {task_type}")
            
            # Create splits for different ratios
            for split_ratio in self.splits:
                logger.info(f"Creating {split_ratio*100}% split...")
                
                # Create stratified split
                train_data, val_data = self.create_stratified_splits(flattened_data, split_ratio)
                
                # Save splits
                self.save_split_data(train_data, val_data, organ_domain, task_type, str(split_ratio))
                
                # Save full processed data (for reference)
                full_filename = f"{organ_domain}_{task_type}_processed.json"
                self.save_processed_data(flattened_data, full_filename)
    
    def process_all_domains(self):
        """Process all organ domains."""
        logger.info("Starting data preprocessing...")
        
        for organ_domain in self.organ_domains:
            self.process_organ_domain(organ_domain)
        
        logger.info("Data preprocessing completed!")
    
    def generate_data_summary(self):
        """Generate a summary of processed data."""
        summary = {}
        
        for organ_domain in self.organ_domains:
            summary[organ_domain] = {}
            
            for task_type in self.task_types:
                summary[organ_domain][task_type] = {}
                
                for split_ratio in self.splits:
                    split_dir = self.splits_dir / organ_domain / task_type / str(split_ratio)
                    
                    if split_dir.exists():
                        train_path = split_dir / 'train.json'
                        val_path = split_dir / 'val.json'
                        
                        if train_path.exists() and val_path.exists():
                            with open(train_path, 'r') as f:
                                train_data = json.load(f)
                            with open(val_path, 'r') as f:
                                val_data = json.load(f)
                            
                            summary[organ_domain][task_type][str(split_ratio)] = {
                                'train_samples': len(train_data),
                                'val_samples': len(val_data),
                                'total_samples': len(train_data) + len(val_data)
                            }
        
        # Save summary
        summary_path = self.data_dir / 'data_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Data summary saved to {summary_path}")
        return summary

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for BioMedCLIP fine-tuning')
    parser.add_argument('--config', type=str, default='finetuning_supervised/configs/lora_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--organ-domain', type=str, default=None,
                       help='Process specific organ domain (optional)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.config)
    
    if args.organ_domain:
        # Process specific organ domain
        if args.organ_domain in preprocessor.organ_domains:
            preprocessor.process_organ_domain(args.organ_domain)
        else:
            logger.error(f"Invalid organ domain: {args.organ_domain}")
            return
    else:
        # Process all domains
        preprocessor.process_all_domains()
    
    # Generate summary
    summary = preprocessor.generate_data_summary()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA PREPROCESSING SUMMARY")
    print("="*50)
    for organ_domain, task_data in summary.items():
        print(f"\n{organ_domain.upper()}:")
        for task_type, split_data in task_data.items():
            print(f"  {task_type}-grained:")
            for split_ratio, counts in split_data.items():
                print(f"    {split_ratio}% split: {counts['train_samples']} train, {counts['val_samples']} val")

if __name__ == "__main__":
    main() 