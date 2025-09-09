#!/usr/bin/env python3
"""
Merge base BioMedCLIP model with soup models to create hybrid models.
This scales LoRA adapters correctly so that an adapter_strength = w
results in an effective delta of size ~ w (by applying sqrt(w) to A and B).

Notes:
- We do NOT mix full base and full fine-tuned weights here. In LoRA, the base
  model W0 stays fixed; the adapter adds ΔW = B @ A (times alpha/r). This script
  simply scales the adapter strength.
- If your adapter carries heads/bias/norms, we leave them unscaled unless you
  explicitly want to blend heads with a true base checkpoint (WiSE-FT style).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import argparse
import shutil
import torch
from safetensors.torch import save_file, load_file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_best_adapter(domain: str, data_ratio: str) -> str:
    """Find the best adapter for a given domain and data ratio."""
    # Special case for soup_all_avg
    if domain == "soup_all_avg":
        soup_path = Path("finetuning_supervised/models/soup/soup_all_avg/adapter_model.safetensors")
        if soup_path.exists():
            logger.info(f"Found soup_all_avg adapter: {soup_path}")
            return str(soup_path)
        raise FileNotFoundError(f"Soup adapter not found: {soup_path}")

    # Original logic for domain-specific adapters
    base_path = Path(f"finetuning_supervised/models/{domain}/combined/{data_ratio}")
    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    # Look for adapter directories with accuracy in the name
    adapter_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("lora_adapters_epoch_")]
    if not adapter_dirs:
        raise FileNotFoundError(f"No adapter directories found in {base_path}")

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

    if best_acc < 0:
        logger.warning(f"Accuracy was not parsed from folder name; proceeding with {best_adapter.name}")

    logger.info(f"Found best adapter for {domain}: {adapter_path} (acc: {best_acc:.4f})")
    return str(adapter_path)

def _is_lora_a(key: str) -> bool:
    """Heuristics for LoRA A keys across repos."""
    k = key.lower()
    return ("w_a" in k) or ("lora_a" in k) or ("lora_down" in k) or (k.endswith("lora_down.weight"))

def _is_lora_b(key: str) -> bool:
    """Heuristics for LoRA B keys across repos."""
    k = key.lower()
    return ("w_b" in k) or ("lora_b" in k) or ("lora_up" in k) or (k.endswith("lora_up.weight"))

def merge_base_with_soup(
    soup_adapter_path: str,
    output_dir: str,
    merge_method: str = "avg",         # kept for compatibility; SLERP is a no-op here
    adapter_strength: float = 0.7
):
    """
    Create a hybrid adapter by scaling the soup adapter with proper LoRA scaling.

    Args:
        soup_adapter_path: Path to soup adapter safetensors file
        output_dir: Directory to save the merged hybrid model
        merge_method: 'avg' or 'slerp' (SLERP is not well-defined for factorized LoRA; treated as scaling)
        adapter_strength: Strength of the adapter in [0,1]; scales ΔW magnitude

    Returns:
        str: Path to saved hybrid adapter file
    """
    if not (0.0 <= adapter_strength <= 1.0):
        raise ValueError("adapter_strength must be between 0.0 and 1.0")

    logger.info(f"Creating hybrid adapter with adapter strength: {adapter_strength:.3f}")
    logger.info(f"Soup adapter: {soup_adapter_path}")

    soup_weights = load_file(soup_adapter_path)
    logger.info(f"Loaded soup adapter with {len(soup_weights)} parameters")

    # Proper LoRA scaling: ΔW = B @ A, so scale A by sqrt(w) and B by sqrt(w) to get effective w
    w = adapter_strength
    sA = w ** 0.5
    sB = w ** 0.5
    logger.info(f"LoRA scaling: A matrices × {sA:.6f}, B matrices × {sB:.6f}")

    hybrid_weights = {}
    nA = nB = nOther = 0

    for key, tensor in soup_weights.items():
        # Safety: only scale float tensors
        if not torch.is_floating_point(tensor):
            hybrid_weights[key] = tensor
            nOther += 1
            continue

        if _is_lora_a(key):
            hybrid_weights[key] = tensor * sA
            nA += 1
        elif _is_lora_b(key):
            hybrid_weights[key] = tensor * sB
            nB += 1
        else:
            # Do NOT scale heads/norms/biases unless explicitly intended
            hybrid_weights[key] = tensor
            nOther += 1

    logger.info(f"Scaled LoRA A: {nA}, LoRA B: {nB}, untouched: {nOther}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    metadata = {
        "merge_method": merge_method,
        "adapter_strength": adapter_strength,
        "scaled_A": sA,
        "scaled_B": sB,
        "soup_adapter_path": soup_adapter_path,
        "merge_timestamp": datetime.now().isoformat(),
        "model_type": "hybrid_base_soup",
        "scaling_method": "sqrt_weighted_lora",
        "note": "Adapter tensors pre-scaled; rely on original alpha in adapter_config (unchanged)."
    }
    metadata_path = out_dir / "merge_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved merge metadata to: {metadata_path}")

    return str(hybrid_adapter_path)

def main():
    ap = argparse.ArgumentParser(description='Create hybrid adapters by scaling soup models with proper LoRA scaling')
    ap.add_argument('--domain', required=True, help='Domain (e.g., cardiovascular, neuropathology, or soup_all_avg)')
    ap.add_argument('--data-ratio', default='0.70', help='Data ratio used for soup models (default: 0.70)')
    ap.add_argument('--method', default='avg', choices=['avg', 'slerp'], help='Merging method (SLERP treated as scaling)')
    ap.add_argument('--adapter-strength', type=float, default=0.7, help='Adapter strength in [0,1] (default: 0.7)')
    ap.add_argument('--out-dir', default='finetuning_supervised/models/hybrid_models', help='Output directory base')
    args = ap.parse_args()

    out_base_dir = Path(args.out_dir)
    out_base_dir.mkdir(parents=True, exist_ok=True)

    domain = args.domain
    logger.info(f"Creating hybrid model for domain: {domain}")

    try:
        soup_adapter_path = find_best_adapter(domain, args.data_ratio)
        domain_out_dir = out_base_dir / f"hybrid_{domain}_base_soup"

        hybrid_adapter_path = merge_base_with_soup(
            soup_adapter_path=soup_adapter_path,
            output_dir=str(domain_out_dir),
            merge_method=args.method,
            adapter_strength=args.adapter_strength
        )
        logger.info(f"Successfully created hybrid model for {domain}: {hybrid_adapter_path}")
    except Exception as e:
        logger.error(f"Failed to create hybrid model for {domain}: {e}")
        return

    logger.info("Hybrid model creation completed!")

if __name__ == "__main__":
    main()
