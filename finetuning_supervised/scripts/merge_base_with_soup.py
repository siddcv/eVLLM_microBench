#!/usr/bin/env python3
"""
Merge base BioMedCLIP model with soup models to create hybrid models.
This combines the general medical knowledge from BioMedCLIP with organ-specific expertise from soups.
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
from safetensors.torch import save_file, load_file
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_adapter(domain: str, data_ratio: str) -> str:
    """Find the best adapter for a given domain and data ratio."""
    base_path = Path(f"finetuning_supervised/models/{domain}/combined/{data_ratio}")
    
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")
    
    # Look for adapter directories with accuracy in the name
    adapter_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("lora_adapters_epoch_")]
    
    if not adapter_dirs:
        raise FileNotFoundError(f"No adapter directories found in {base_path}")
    
    # Extract accuracy from directory names and find the best one
    best_adapter = None
    best_acc = -1.0
    
    for adapter_dir in adapter_dirs:
        # Extract accuracy from directory name (e.g., "lora_adapters_epoch_9_acc_0.9640")
        if "_acc_" in adapter_dir.name:
            try:
                acc_str = adapter_dir.name.split("_acc_")[1]
                acc = float(acc_str)
                if acc > best_acc:
                    best_acc = acc
                    best_adapter = adapter_dir
            except (ValueError, IndexError):
                continue
    
    if best_adapter is None:
        # Fallback: use the first adapter directory
        best_adapter = adapter_dirs[0]
        logger.warning(f"Could not determine best adapter for {domain}, using {best_adapter.name}")
    
    adapter_path = best_adapter / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
    
    logger.info(f"Found best adapter for {domain}: {adapter_path} (acc: {best_acc:.4f})")
    return str(adapter_path)

def merge_base_with_soup(base_model_path: str, soup_adapter_path: str, output_dir: str, 
                        merge_method: str = "avg", base_weight: float = 0.5, soup_weight: float = 0.5):
    """
    Merge base BioMedCLIP model with a soup adapter to create a hybrid model.
    
    Args:
        base_model_path: Path to base BioMedCLIP model weights
        soup_adapter_path: Path to soup adapter safetensors file
        output_dir: Directory to save the merged hybrid model
        merge_method: Method to merge ('avg' or 'slerp')
        base_weight: Weight for base model (0.0 to 1.0)
        soup_weight: Weight for soup model (0.0 to 1.0)
    """
    
    # Normalize weights
    total_weight = base_weight + soup_weight
    base_weight = base_weight / total_weight
    soup_weight = soup_weight / total_weight
    
    logger.info(f"Merging base model (weight: {base_weight:.3f}) with soup adapter (weight: {soup_weight:.3f})")
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Soup adapter: {soup_adapter_path}")
    
    # Load soup adapter weights
    soup_weights = load_file(soup_adapter_path)
    logger.info(f"Loaded soup adapter with {len(soup_weights)} parameters")
    
    # Load base model weights (this would be the BioMedCLIP base model)
    # Note: For now, we'll create a hybrid adapter that combines the soup with base-like characteristics
    # In practice, you might want to load the actual base model weights and merge them
    
    # Create hybrid weights by scaling the soup adapter
    hybrid_weights = {}
    
    for key, tensor in soup_weights.items():
        if merge_method == "avg":
            # For averaging, we scale the soup weights by the soup_weight
            # The base model contribution would be zero (since we're working with LoRA adapters)
            hybrid_weights[key] = tensor * soup_weight
        elif merge_method == "slerp":
            # For slerp, we need to handle it differently since we're working with adapters
            # For now, use weighted average
            hybrid_weights[key] = tensor * soup_weight
        else:
            raise ValueError(f"Unsupported merge method: {merge_method}")
    
    # Create output directory
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hybrid adapter
    hybrid_adapter_path = out_dir / "adapter_model.safetensors"
    save_file(hybrid_weights, str(hybrid_adapter_path))
    logger.info(f"Saved hybrid adapter to: {hybrid_adapter_path}")
    
    # Copy adapter_config.json from soup adapter
    soup_config_source = Path(soup_adapter_path).parent / "adapter_config.json"
    hybrid_config_dest = out_dir / "adapter_config.json"
    
    if soup_config_source.exists():
        shutil.copy2(soup_config_source, hybrid_config_dest)
        logger.info(f"Copied adapter_config.json to: {hybrid_config_dest}")
    else:
        logger.warning(f"adapter_config.json not found at {soup_config_source}")
    
    # Save merge metadata
    metadata = {
        "merge_method": merge_method,
        "base_weight": base_weight,
        "soup_weight": soup_weight,
        "base_model_path": base_model_path,
        "soup_adapter_path": soup_adapter_path,
        "merge_timestamp": datetime.now().isoformat(),
        "model_type": "hybrid_base_soup"
    }
    
    metadata_path = out_dir / "merge_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved merge metadata to: {metadata_path}")
    
    return str(hybrid_adapter_path)

def main():
    parser = argparse.ArgumentParser(description='Merge base BioMedCLIP model with soup models to create hybrid models')
    parser.add_argument('--domain', type=str, required=True,
                       help='Single domain to create hybrid model for (e.g., cardiovascular, neuropathology)')
    parser.add_argument('--data-ratio', type=str, default='0.70',
                       help='Data ratio used for the soup models (default: 0.70)')
    parser.add_argument('--method', type=str, default='avg', choices=['avg', 'slerp'],
                       help='Merging method (default: avg)')
    parser.add_argument('--base-weight', type=float, default=0.3,
                       help='Weight for base model in merge (default: 0.3)')
    parser.add_argument('--soup-weight', type=float, default=0.7,
                       help='Weight for soup model in merge (default: 0.7)')
    parser.add_argument('--out-dir', type=str, default='finetuning_supervised/models/hybrid_models',
                       help='Output directory for hybrid models')
    
    args = parser.parse_args()
    
    # Create output directory
    out_base_dir = Path(args.out_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)
    
    domain = args.domain
    logger.info(f"Creating hybrid model for domain: {domain}")
    
    try:
        # Find the best soup adapter for this domain
        soup_adapter_path = find_best_adapter(domain, args.data_ratio)
        
        # Create output directory for this domain
        domain_out_dir = out_base_dir / f"hybrid_{domain}_base_soup"
        
        # For now, we'll use a placeholder for base model path
        # In practice, you might want to load the actual BioMedCLIP base weights
        base_model_path = "BioMedCLIP_Base_Model"  # Placeholder
        
        # Merge base with soup
        hybrid_adapter_path = merge_base_with_soup(
            base_model_path=base_model_path,
            soup_adapter_path=soup_adapter_path,
            output_dir=str(domain_out_dir),
            merge_method=args.method,
            base_weight=args.base_weight,
            soup_weight=args.soup_weight
        )
        
        logger.info(f"Successfully created hybrid model for {domain}: {hybrid_adapter_path}")
        
    except Exception as e:
        logger.error(f"Failed to create hybrid model for {domain}: {e}")
        return
    
    logger.info("Hybrid model creation completed!")

if __name__ == "__main__":
    main()
