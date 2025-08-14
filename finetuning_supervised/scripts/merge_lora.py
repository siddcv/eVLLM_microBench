#!/usr/bin/env python3
import argparse, math
from pathlib import Path
from typing import List
import torch
from safetensors.torch import load_file, save_file
import re

def is_lora_key(key: str) -> bool:
    """Check if a key looks like a LoRA parameter."""
    lora_patterns = [
        'lora_A', 'lora_B', 'lora_embedding_A', 'lora_embedding_B',
        'lora_scaling', 'lora_dropout'
    ]
    return any(pattern in key for pattern in lora_patterns)

def normalize_weights(weights: List[float]) -> List[float]:
    """Normalize weights to sum to 1.0."""
    total = sum(weights)
    if total == 0:
        raise ValueError("Weights cannot all be zero")
    return [w / total for w in weights]

def _linear_avg(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=tensors[0].dtype, device=tensors[0].device)
    out = torch.zeros_like(tensors[0])
    for ti, wi in zip(tensors, w):
        out.add_(ti * wi)
    return out

def _slerp(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    # flatten to vector for angle; then reshape back
    shape = t0.shape
    v0 = t0.view(-1)
    v1 = t1.view(-1)

    # handle near-zero norms
    n0 = torch.norm(v0)
    n1 = torch.norm(v1)
    if n0 < 1e-12 or n1 < 1e-12:
        return (1 - alpha) * t0 + alpha * t1

    v0n = v0 / n0
    v1n = v1 / n1
    dot = torch.clamp(torch.dot(v0n, v1n), -1.0, 1.0)
    theta = torch.acos(dot)

    if torch.isnan(theta) or theta < 1e-6:
        return (1 - alpha) * t0 + alpha * t1

    s = math.sin
    out = (s((1 - alpha) * theta) / s(theta)) * v0 + (s(alpha * theta) / s(theta)) * v1
    return out.view(shape)

def _pairwise_slerp(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    # reduce list by weighted SLERP (weights already normalized)
    # start from the highest-weight tensor
    idx = max(range(len(weights)), key=lambda i: weights[i])
    acc = tensors[idx]
    acc_w = weights[idx]
    for i, (ti, wi) in enumerate(zip(tensors, weights)):
        if i == idx: 
            continue
        if wi <= 0: 
            continue
        alpha = wi / (acc_w + wi)
        acc = _slerp(acc, ti, alpha)
        acc_w += wi
    return acc

def merge_adapters(adapter_paths: List[Path], out_dir: Path, method: str, weights: List[float] = None):
    assert len(adapter_paths) >= 2, "Need at least 2 adapters to create a soup"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all state dicts (safetensors)
    sd_list = [load_file(str(p)) for p in adapter_paths]
    keys = sd_list[0].keys()

    # sanity: all have same keys
    for sd in sd_list[1:]:
        if sd.keys() != keys:
            diff = (set(keys) ^ set(sd.keys()))
            raise ValueError(f"Adapter key mismatch. Diff keys: {list(diff)[:10]}")

    # Normalize weights upfront
    if weights is None:
        weights = [1.0] * len(sd_list)
    else:
        assert len(weights) == len(sd_list), f"Expected {len(sd_list)} weights, got {len(weights)}"
    
    # Normalize weights to sum to 1.0
    original_weights = weights.copy()
    weights = normalize_weights(weights)
    
    print("Weights summary:")
    for i, (original, normalized) in enumerate(zip(original_weights, weights)):
        print(f"  Adapter {i+1}: original={original:.3f}, normalized={normalized:.3f}")

    # Identify LoRA keys
    lora_keys = {k for k in keys if is_lora_key(k)}
    non_lora_keys = keys - lora_keys
    
    print(f"Found {len(lora_keys)} LoRA keys and {len(non_lora_keys)} non-LoRA keys")

    merged = {}
    merged_count = 0
    passed_through_count = 0
    
    for k in keys:
        if k in lora_keys:
            # Merge LoRA parameters
            if not torch.is_floating_point(sd_list[0][k]):
                print(f"Warning: Non-floating point LoRA key {k}, passing through from first adapter")
                merged[k] = sd_list[0][k]
                passed_through_count += 1
                continue

            tensors = [sd[k] for sd in sd_list]
            if method == "avg":
                merged[k] = _linear_avg(tensors, weights)
            elif method == "slerp":
                merged[k] = _pairwise_slerp(tensors, weights)
            else:
                raise ValueError(f"Unknown method: {method}")
            merged_count += 1
            
        else:
            # Pass through non-LoRA parameters from first adapter
            merged[k] = sd_list[0][k]
            passed_through_count += 1

    print(f"Merging complete: {merged_count} keys merged, {passed_through_count} keys passed through")

    # Save as a new adapter safetensors
    out_path = out_dir / "adapter_model.safetensors"
    save_file(merged, str(out_path))
    print(f"[ModelSoup] Saved merged adapter to: {out_path}")

def extract_domain_info(adapter_path: Path) -> str:
    """Extract domain information from adapter path"""
    path_parts = adapter_path.parts
    for part in path_parts:
        if part in ['cardiovascular', 'gastrointestinal', 'hematopathology', 'neuropathology']:
            return part
    return "unknown"

def generate_soup_name(adapter_paths: List[Path], method: str) -> str:
    """Generate a name for the soup based on domains and method"""
    domains = [extract_domain_info(path) for path in adapter_paths]
    domain_names = []
    for domain in domains:
        if domain == 'cardiovascular':
            domain_names.append('cardio')
        elif domain == 'gastrointestinal':
            domain_names.append('gastro')
        elif domain == 'hematopathology':
            domain_names.append('hemato')
        elif domain == 'neuropathology':
            domain_names.append('neuro')
        else:
            domain_names.append(domain)
    
    return f"soup_{'_'.join(domain_names)}_{method}"

def generate_results_path(soup_name: str, test_domain: str = None) -> Path:
    """Generate path for soup results"""
    base_path = Path("finetuning_supervised/models/soup")
    if test_domain:
        return base_path / soup_name / f"test_{test_domain}"
    return base_path / soup_name

def find_best_adapters(domains: List[str], data_ratio: float = 0.70):
    """Find the best performing adapters for given domains and data ratio"""
    base_path = Path("finetuning_supervised/models")
    adapter_paths = []
    
    for domain in domains:
        # Try both 0.70 and 0.7 formats
        if data_ratio == 0.70:
            domain_path = base_path / domain / "combined" / "0.70"
            alt_path = base_path / domain / "combined" / "0.7"
        elif data_ratio == 0.7:
            domain_path = base_path / domain / "combined" / "0.7"
            alt_path = base_path / domain / "combined" / "0.70"
        else:
            domain_path = base_path / domain / "combined" / str(data_ratio)
            alt_path = None
        
        if domain_path.exists():
            pass  # Use domain_path as is
        elif alt_path and alt_path.exists():
            domain_path = alt_path
        else:
            print(f"Warning: Neither {domain_path} nor {alt_path} exists")
            continue
        
        # Find the best adapter in this domain directory
        best_adapter = None
        best_acc = -1.0
        
        for item in domain_path.iterdir():
            if item.is_dir() and "lora_adapters_epoch" in item.name:
                # Extract accuracy from directory name
                acc_match = re.search(r'acc_(\d+\.\d+)', item.name)
                if acc_match:
                    acc = float(acc_match.group(1))
                    if acc > best_acc:
                        best_acc = acc
                        best_adapter = item / "adapter_model.safetensors"
        
        if best_adapter and best_adapter.exists():
            adapter_paths.append(best_adapter)
            print(f"Found best adapter for {domain}: {best_adapter} (acc: {best_acc:.4f})")
        else:
            print(f"Warning: No valid adapter found for {domain}")
    
    return adapter_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", 
                    help="Paths to adapter_model.safetensors for each organ")
    ap.add_argument("--domains", nargs="+", 
                    help="Domain names to automatically find best adapters for")
    ap.add_argument("--data-ratio", type=float, default=0.70,
                    help="Data ratio to use when finding adapters (default: 0.70)")
    ap.add_argument("--weights", nargs="*", type=float, default=None,
                    help="Optional weights per adapter (will be normalized to sum to 1.0)")
    ap.add_argument("--method", choices=["avg", "slerp"], default="avg")
    ap.add_argument("--out-dir", type=str, required=True,
                    help="Output directory for the merged adapter")
    args = ap.parse_args()

    if args.adapters:
        adapter_paths = [Path(p) for p in args.adapters]
    elif args.domains:
        adapter_paths = find_best_adapters(args.domains, args.data_ratio)
        if len(adapter_paths) < 2:
            print("Error: Need at least 2 adapters to create a soup")
            return
    else:
        print("Error: Must specify either --adapters or --domains")
        return

    out_dir = Path(args.out_dir)
    merge_adapters(adapter_paths, out_dir, args.method, args.weights)

if __name__ == "__main__":
    main()
