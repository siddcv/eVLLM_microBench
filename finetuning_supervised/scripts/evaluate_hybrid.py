#!/usr/bin/env python3
"""
Evaluate hybrid models (base BioMedCLIP + soup models) on test data.
Uses the same evaluation methodology as soup and baseline evaluations.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Any
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy import stats

# Import the LoRA model
from biomedclip_lora import BioMedCLIPLoRA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_confidence_interval(correct: int, total: int, confidence: float = 0.95):
    if total == 0:
        return 0.0, 0.0
    p = correct / total
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1.0 + (z**2) / total
    centre = p + (z**2) / (2 * total)
    margin = z * np.sqrt((p * (1 - p) / total) + (z**2) / (4 * total**2))
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)

def extract_test_path(test_file: str) -> str:
    """Extract test path from test file path for output directory naming."""
    test_path = Path(test_file)
    parts = test_path.parts
    if 'splits' in parts:
        splits_idx = parts.index('splits')
        if splits_idx + 2 < len(parts):
            domain = parts[splits_idx + 1]   # e.g., cardiovascular
            task_type = parts[splits_idx + 2]  # e.g., fine
            return f"{domain}_{task_type}"
    return test_path.stem

def _load_items(path: str) -> List[Dict]:
    """Load data from JSON or JSONL file."""
    p = Path(path)
    if p.suffix == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    else:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

def save_results(results: List[Dict], hybrid_domain: str, task_type: str, test_file: str, merge_metadata: Dict[str, Any] = None):
    """Save evaluation results to CSV and accuracy metrics to JSON."""
    if not results:
        logger.warning("No results to save")
        return

    df = pd.DataFrame(results)
    test_path = extract_test_path(test_file)

    out_dir = Path(f"output_results/{task_type}_grained_tasks/Hybrid_Base_Soup/{hybrid_domain}/{test_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{test_path}_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")

    # Accuracy
    correct = sum(1 for r in results if r['correct_idx'] == r['model_answers']['pred'][0])
    total = len(results)
    accuracy = correct / total if total > 0 else 0.0

    # 95% CI
    ci_lower, ci_upper = calculate_confidence_interval(correct, total, confidence=0.95)
    logger.info(f"Overall accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Per-question-type
    per_qtype_metrics: Dict[str, Dict[str, Any]] = {}
    for q_type, q_df in df.groupby('question_type'):
        q_correct = sum(1 for _, row in q_df.iterrows() if row['correct_idx'] == row['model_answers']['pred'][0])
        q_total = len(q_df)
        q_acc = q_correct / q_total if q_total > 0 else 0.0
        q_ci_lower, q_ci_upper = calculate_confidence_interval(q_correct, q_total, confidence=0.95)
        per_qtype_metrics[q_type] = {
            'accuracy': q_acc,
            'correct': q_correct,
            'total': q_total,
            'confidence_interval_95': {'lower': q_ci_lower, 'upper': q_ci_upper}
        }
        logger.info(f"{q_type} accuracy: {q_acc:.4f} ({q_correct}/{q_total})")
        logger.info(f"{q_type} 95% CI: [{q_ci_lower:.4f}, {q_ci_upper:.4f}]")

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
            'confidence_interval_95': {'lower': ci_lower, 'upper': ci_upper}
        },
        'per_question_type': per_qtype_metrics,
        'merge_metadata': merge_metadata
    }
    metrics_path = out_dir / f"{test_path}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Accuracy metrics saved to {metrics_path}")

def evaluate_hybrid_model(hybrid_adapter_path: str, test_file: str, hybrid_domain: str, task_type: str):
    """Evaluate hybrid model (base BioMedCLIP + soup) on test data."""
    test_data = _load_items(test_file)
    logger.info(f"Loaded {len(test_data)} test samples from {test_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BioMedCLIPLoRA(eval_mode=True, context_length=256, verbose=True)
    # Set device after initialization if the model supports it
    if hasattr(model, 'device'):
        model.device = device

    # Accept either directory or file path
    if hybrid_adapter_path.endswith('.safetensors'):
        hybrid_adapter_dir = str(Path(hybrid_adapter_path).parent)
    else:
        hybrid_adapter_dir = hybrid_adapter_path

    logger.info(f"Loading hybrid model from {hybrid_adapter_dir}")
    merge_meta = None
    meta_path = Path(hybrid_adapter_dir) / "merge_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                merge_meta = json.load(f)
        except Exception:
            logger.warning("Failed to parse merge_metadata.json; continuing.")

    try:
        model.load_lora_adapter(hybrid_adapter_dir)
        logger.info("Hybrid model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load hybrid model: {e}")
        raise

    results: List[Dict[str, Any]] = []
    skipped_missing = 0
    skipped_invalid = 0

    for data_point in tqdm(test_data, desc=f"Evaluating {task_type}"):
        image_id = data_point.get('image_id', '')
        image_path = data_point.get('image_path', '')
        question = data_point.get('question', '')
        answer_options = data_point.get('answer_options', [])
        correct_answer_idx = data_point.get('correct_answer_idx', data_point.get('correct_idx', 0))
        question_type = data_point.get('question_type', '')

        if not image_path or not question or not answer_options:
            skipped_invalid += 1
            logger.warning(f"Skipping sample {image_id}: missing required fields")
            continue
        if not os.path.exists(image_path):
            skipped_missing += 1
            logger.warning(f"Image not found: {image_path}")
            continue

        try:
            prediction_result = model.predict_single(image_path, question, answer_options)
            predicted_idx = prediction_result['predicted_idx']
            confidence = prediction_result.get('confidence', None)
            scores = prediction_result.get('all_probs', None)

            result = {
                'question_type': question_type,
                'questions': str([f"{question} {option}" for option in answer_options]),
                'image_id': image_id,
                'correct_answer': answer_options[correct_answer_idx],
                'correct_idx': correct_answer_idx,
                'model_answers': {
                    'pred': [predicted_idx],
                    'probs': [scores],      # keep nesting for backward compatibility
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

    logger.info(f"Total results generated: {len(results)} "
                f"(skipped_missing={skipped_missing}, skipped_invalid={skipped_invalid})")

    save_results(results, hybrid_domain, task_type, test_file, merge_metadata=merge_meta)

def main():
    parser = argparse.ArgumentParser(description='Evaluate hybrid models (base BioMedCLIP + soup) on test data')
    parser.add_argument('--hybrid-adapter', required=True, help='Path to hybrid adapter directory or file')
    parser.add_argument('--hybrid-domain', required=True, help='Domain name for the hybrid model (e.g., cardiovascular)')
    parser.add_argument('--test-file', required=True, help='Path to test file (JSON or JSONL)')
    parser.add_argument('--task-type', required=True, choices=['coarse', 'fine', 'combined'],
                        help='Task type of the test data')
    args = parser.parse_args()

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
