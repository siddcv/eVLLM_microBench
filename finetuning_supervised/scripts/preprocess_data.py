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
        
        # Define fine types explicitly; treat all others as coarse (avoids missing keys like 'subdomain')
        self.fine_types = ['classification']
        
        # Train subset sizes (fractions of TOTAL dataset)
        self.train_subsplits = [0.10, 0.25, 0.50, 0.70]
        
        # Fixed split fractions
        self.fixed_val_frac = 0.15
        self.fixed_test_frac = 0.15
        self.fixed_train_frac = 0.70
        
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
            metadata = item.get('metadata', {})
            image_id = metadata.get('image_id', '')
            ########################################################################################################################
            workspace = '/workspace/eVLLM_Sidd/eVLLM_microBench'
            image_file = metadata.get('image', '')
            image_path = f"{workspace}/organData/{task_type}Grain/{organ_domain}/images/{image_file}"
            
            # Merge questions from custom_metadata['questions'] and captions
            merged_questions: Dict[str, Dict[str, Any]] = {}
            # Prefer custom_metadata['questions'] if present
            if 'custom_metadata' in item and 'questions' in item['custom_metadata']:
                for qtype, q_data in item['custom_metadata']['questions'].items():
                    merged_questions[qtype] = {
                        'question': q_data.get('question', ''),
                        'options': q_data.get('options', []),
                        'answer_idx': int(q_data.get('answer_idx', 0)),
                    }
            # Add any remaining from captions not already present
            if 'captions' in item:
                for cap_key, q_data in item['captions'].items():
                    qtype = str(cap_key).split('_')[0]
                    if qtype not in merged_questions:
                        merged_questions[qtype] = {
                            'question': q_data.get('question', ''),
                            'options': q_data.get('options', []),
                            'answer_idx': int(q_data.get('answer_idx', 0)),
                        }
            
            # Emit one sample per merged question type
            for question_type, q_norm in merged_questions.items():
                sample = {
                    'image_id': image_id,
                    'image_path': image_path,
                    'question': q_norm.get('question', ''),
                    'answer_options': q_norm.get('options', []),
                    'correct_answer_idx': int(q_norm.get('answer_idx', 0)),
                    'question_type': question_type,
                    'organ_domain': organ_domain,
                    'task_type': task_type
                }
                flattened_data.append(sample)
        
        return flattened_data
    
    # Helper: filter flattened data by task granularity
    def _filter_by_granularity(self, data: List[Dict], granularity: str) -> List[Dict]:
        if granularity == 'fine':
            return [d for d in data if d.get('question_type') in self.fine_types]
        elif granularity == 'coarse':
            return [d for d in data if d.get('question_type') not in self.fine_types]
        else:
            return data
    
    # Helper: perform fixed 70/15/15 split (stratified by question_type)
    def _fixed_split(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        if len(data) == 0:
            return [], [], []
        df = pd.DataFrame(data)
        # First: split test (15%)
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.fixed_test_frac,
            random_state=self.random_seed,
            stratify=df['question_type'] if df['question_type'].nunique() > 1 else None
        )
        # Second: split val from train_val to be 15% of total
        val_frac_of_train_val = self.fixed_val_frac / (1.0 - self.fixed_test_frac)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_frac_of_train_val,
            random_state=self.random_seed,
            stratify=train_val_df['question_type'] if train_val_df['question_type'].nunique() > 1 else None
        )
        return train_df.to_dict('records'), val_df.to_dict('records'), test_df.to_dict('records')
    
    # Helper: sample a stratified subset of a target size from a pool
    def _sample_stratified_count(self, pool: List[Dict], target_count: int) -> List[Dict]:
        if target_count <= 0 or len(pool) == 0:
            return []
        df = pd.DataFrame(pool)
        if target_count >= len(df):
            return df.to_dict('records')
        # Use train_test_split to sample target_count rows stratified by question_type when possible
        test_size = target_count / len(df)
        _, subset = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=df['question_type'] if df['question_type'].nunique() > 1 else None
        )
        return subset.to_dict('records')
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """Save processed data to JSON file."""
        output_path = self.processed_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed data to {output_path}")
    
    def _save_fixed(self, organ_domain: str, granularity: str, val_data: List[Dict], test_data: List[Dict]):
        fixed_dir = self.splits_dir / organ_domain / granularity / 'fixed'
        fixed_dir.mkdir(parents=True, exist_ok=True)
        with open(fixed_dir / 'val.json', 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        with open(fixed_dir / 'test.json', 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved fixed splits to {fixed_dir}")
    
    def _save_train_subset(self, organ_domain: str, granularity: str, ratio: float, train_subset: List[Dict]):
        ratio_dir = self.splits_dir / organ_domain / granularity / 'trains' / f"{ratio:.2f}"
        ratio_dir.mkdir(parents=True, exist_ok=True)
        with open(ratio_dir / 'train.json', 'w', encoding='utf-8') as f:
            json.dump(train_subset, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved train subset ({ratio:.2f}) to {ratio_dir}")
    
    def _build_and_save_for_granularity(self, organ_domain: str, granularity: str, source_flattened: List[Dict]):
        """Build fixed splits and train subsets for a specific granularity."""
        # Fixed 70/15/15 for this granularity
        train_pool, val_data, test_data = self._fixed_split(source_flattened)
        self._save_fixed(organ_domain, granularity, val_data, test_data)
        
        # Train subsets: sizes are fractions of TOTAL dataset length
        total_n = len(source_flattened)
        for r in self.train_subsplits:
            target_count = int(round(r * total_n))
            subset = self._sample_stratified_count(train_pool, target_count)
            self._save_train_subset(organ_domain, granularity, r, subset)
    
    def _build_and_save_combined(self, organ_domain: str, coarse_flat: List[Dict], fine_flat: List[Dict]):
        """Build combined fixed and train subsets ensuring 70/15/15 per granularity in fixed sets."""
        # Fixed per granularity
        coarse_train, coarse_val, coarse_test = self._fixed_split(coarse_flat)
        fine_train, fine_val, fine_test = self._fixed_split(fine_flat)
        
        # Combined fixed are unions of per-granularity fixed
        combined_val = coarse_val + fine_val
        combined_test = coarse_test + fine_test
        combined_train_pool = coarse_train + fine_train
        
        # Save fixed
        self._save_fixed(organ_domain, 'combined', combined_val, combined_test)
        
        # Train subsets sampled from combined train pool, sizes are fractions of TOTAL combined size
        total_n = len(coarse_flat) + len(fine_flat)
        for r in self.train_subsplits:
            target_count = int(round(r * total_n))
            subset = self._sample_stratified_count(combined_train_pool, target_count)
            self._save_train_subset(organ_domain, 'combined', r, subset)
    
    def process_organ_domain(self, organ_domain: str):
        """Process data for a specific organ domain."""
        logger.info(f"Processing {organ_domain} domain...")
        
        # Load both coarse and fine data
        coarse_file = f"organData/coarseGrain/{organ_domain}/test_200.jsonl"
        fine_file = f"organData/fineGrain/{organ_domain}/test_200.jsonl"
        
        coarse_flat = []
        fine_flat = []
        
        # Process coarse data
        if os.path.exists(coarse_file):
            raw_coarse = self.load_jsonl_data(coarse_file)
            coarse_all = self.flatten_data(raw_coarse, organ_domain, 'coarse')
            coarse_flat = self._filter_by_granularity(coarse_all, 'coarse')
            self.save_processed_data(coarse_flat, f"{organ_domain}_coarse_processed.json")
        else:
            logger.warning(f"File not found: {coarse_file}")
        
        # Process fine data  
        if os.path.exists(fine_file):
            raw_fine = self.load_jsonl_data(fine_file)
            fine_all = self.flatten_data(raw_fine, organ_domain, 'fine')
            fine_flat = self._filter_by_granularity(fine_all, 'fine')
            self.save_processed_data(fine_flat, f"{organ_domain}_fine_processed.json")
        else:
            logger.warning(f"File not found: {fine_file}")
        
        # Combined flattened (reference only)
        combined_flat = coarse_flat + fine_flat
        self.save_processed_data(combined_flat, f"{organ_domain}_combined_processed.json")
        
        # Build per-granularity outputs
        self._build_and_save_for_granularity(organ_domain, 'coarse', coarse_flat)
        self._build_and_save_for_granularity(organ_domain, 'fine', fine_flat)
        self._build_and_save_combined(organ_domain, coarse_flat, fine_flat)
    
    def process_all_domains(self):
        """Process all organ domains."""
        logger.info("Starting data preprocessing...")
        
        for organ_domain in self.organ_domains:
            self.process_organ_domain(organ_domain)
        
        logger.info("Data preprocessing completed!")
    
    def generate_data_summary(self):
        """Generate a summary of processed data."""
        summary: Dict[str, Dict[str, Any]] = {}
        tasks = ['combined', 'coarse', 'fine']
        
        for organ_domain in self.organ_domains:
            summary[organ_domain] = {}
            for task in tasks:
                task_entry: Dict[str, Any] = {}
                fixed_dir = self.splits_dir / organ_domain / task / 'fixed'
                trains_dir = self.splits_dir / organ_domain / task / 'trains'
                
                fixed_counts = {'val': 0, 'test': 0}
                if fixed_dir.exists():
                    val_path = fixed_dir / 'val.json'
                    test_path = fixed_dir / 'test.json'
                    if val_path.exists():
                        with open(val_path, 'r') as f:
                            val_data = json.load(f)
                        fixed_counts['val'] = len(val_data)
                    if test_path.exists():
                        with open(test_path, 'r') as f:
                            test_data = json.load(f)
                        fixed_counts['test'] = len(test_data)
                
                trains_counts: Dict[str, int] = {}
                if trains_dir.exists():
                    for r in self.train_subsplits:
                        r_dir = trains_dir / f"{r:.2f}" / 'train.json'
                        if r_dir.exists():
                            with open(r_dir, 'r') as f:
                                train_data = json.load(f)
                            trains_counts[f"{r:.2f}"] = len(train_data)
                
                task_entry['fixed'] = fixed_counts
                task_entry['trains'] = trains_counts
                summary[organ_domain][task] = task_entry
        
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
        for task_type, counts in task_data.items():
            print(f"  {task_type}:")
            fixed = counts.get('fixed', {})
            trains = counts.get('trains', {})
            print(f"    Fixed → Val: {fixed.get('val', 0)}, Test: {fixed.get('test', 0)}")
            if trains:
                for r_str, n in sorted(trains.items(), key=lambda x: float(x[0])):
                    print(f"    Train {r_str}: {n}")

if __name__ == "__main__":
    main()