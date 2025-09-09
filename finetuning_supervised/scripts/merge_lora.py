#!/usr/bin/env python3
import argparse, math
import shutil
from pathlib import Path
from typing import List, Optional
import torch
from safetensors.torch import load_file, save_file
import re

def _linear_avg(tensors: List[torch.Tensor], weights: List[float]) -> torch.Tensor:
    w = torch.tensor(weights, dtype=tensors[0].dtype, device=tensors[0].device)
    w = w / w.sum()
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
    # reduce list by weighted SLERP (normalize weights to sum=1)
    w = [w / sum(weights) for w in weights]
    # start from the highest-weight tensor
    idx = max(range(len(w)), key=lambda i: w[i])
    acc = tensors[idx]
    acc_w = w[idx]
    for i, (ti, wi) in enumerate(zip(tensors, w)):
        if i == idx: 
            continue
        if wi <= 0: 
            continue
        alpha = wi / (acc_w + wi)
        acc = _slerp(acc, ti, alpha)
        acc_w += wi
    return acc

def _apply_wiseft_scaling(param_dict: dict, tau: float) -> dict:
    """
    WiSE-FT for LoRA: scale A/B so the effective delta (B@A) is multiplied by tau.
    We scale A and B by sqrt(tau). Norms/bias/heads are left untouched.
    """
    s = tau ** 0.5
    out = {}
    for k, t in param_dict.items():
        kl = k.lower()
        if ("w_a" in kl) or ("lora_a" in kl):
            out[k] = t * s
        elif ("w_b" in kl) or ("lora_b" in kl):
            out[k] = t * s
        else:
            out[k] = t
    return out




def _task_arithmetic(tensors: List[torch.Tensor], weights: List[float], beta: float = 0.5) -> torch.Tensor:
    """
    Task Arithmetic: Center deltas to emphasize complementary information, then recombine.
    
    Args:
        tensors: List of organ-specific delta tensors
        weights: Organ weights (alpha_o)
        beta: Controls how much generic skill to keep (default: 0.5)
    
    Returns:
        Merged tensor using task arithmetic
    """
    # Normalize weights
    w = torch.tensor(weights, dtype=tensors[0].dtype, device=tensors[0].device)
    w = w / w.sum()
    
    # Compute mean delta across organs (shared "generic" skill)
    mean_delta = torch.stack(tensors).mean(dim=0)
    
    # Center each organ's delta (remove common mean)
    centered_tensors = [t - mean_delta for t in tensors]
    
    # Weighted combination of centered deltas
    centered_sum = torch.zeros_like(tensors[0])
    for centered_tensor, weight in zip(centered_tensors, w):
        centered_sum.add_(centered_tensor * weight)
    
    # Add back beta * mean_delta (shared component)
    result = centered_sum + beta * mean_delta
    
    return result




def _ties_merging(
    tensors: List[torch.Tensor],
    weights: List[float],
    keep_top_p: float = 0.25,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    TIES-style merge (Trim-Intersect-Resolve) with quantile trimming,
    agreement on non-zero signs, and weighted mean over non-zeros.

    Args:
        tensors: list of tensors (same shape)
        weights: per-adapter weights (used in agreement averaging)
        keep_top_p: fraction to keep per adapter by magnitude (e.g., 0.25)
        eps: numerical stability for denominators

    Returns:
        merged tensor
    """
    # Stack as [n, ...] and flatten to [n, K] for trimming
    T = torch.stack(tensors, dim=0)
    n = T.shape[0]
    flat = T.view(n, -1)

    # TRIM: keep top-p by magnitude per adapter via quantile
    mags = flat.abs()
    kth = torch.quantile(mags, q=max(0.0, min(1.0, 1.0 - keep_top_p)), dim=1, keepdim=True)
    keep = (mags >= kth).view_as(T)
    T_trim = T * keep

    # AGREEMENT on non-zero signs
    signs = torch.sign(T_trim)  # -1, 0, +1
    nonzero = (signs != 0)
    nz_count = nonzero.sum(dim=0)
    pos_count = (signs > 0).sum(dim=0)
    neg_count = (signs < 0).sum(dim=0)
    agree_pos = (nz_count > 0) & (neg_count == 0)
    agree_neg = (nz_count > 0) & (pos_count == 0)
    agree = agree_pos | agree_neg
    disagree = (nz_count > 1) & (~agree)

    out = torch.zeros_like(tensors[0])

    # AGREEMENT: weighted mean over non-zero contributors only
    if agree.any():
        mask = agree
        vals = T_trim[:, mask]  # [n, K]
        w = torch.tensor(weights, dtype=vals.dtype, device=vals.device)
        w = w / (w.sum() + eps)
        nz = (vals != 0).to(vals.dtype)
        denom = (w.view(-1, 1) * nz).sum(dim=0).clamp_min(eps)
        num = (w.view(-1, 1) * vals).sum(dim=0)
        out[mask] = num / denom

    # DISAGREEMENT: take contributor with max magnitude among non-zeros
    if disagree.any():
        mask = disagree
        vals = T_trim[:, mask]       # [n, K]
        mags = vals.abs()
        idx = mags.argmax(dim=0)     # [K]
        gathered = vals.gather(0, idx.unsqueeze(0)).squeeze(0)
        out[mask] = gathered

    # Elsewhere no contributors -> zeros remain
    return out

def merge_adapters(
    adapter_paths: List[Path],
    out_dir: Path,
    method: str,
    weights: List[float] = None,
    beta: float = 0.5,
    trim_threshold: float = 0.75,
    ties_keep_top_p: float = 0.25,
    wiseft_tau: float = 0.7,
    wiseft_premerge: str = "avg",
):
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

    # default: equal weights
    if weights is None:
        weights = [1.0] * len(sd_list)
    else:
        assert len(weights) == len(sd_list)

    merged = {}
    for k in keys:
        # only merge LoRA params and related biases; skip non-float tensors safely
        if not torch.is_floating_point(sd_list[0][k]):
            merged[k] = sd_list[0][k]
            continue

        tensors = [sd[k] for sd in sd_list]
        
        if method in {"avg", "slerp", "wiseft"}:
            # First do the organ pre-merge (avg or slerp).
            if (method == "avg") or (method == "wiseft" and wiseft_premerge == "avg"):
                premerged = _linear_avg(tensors, weights)
            else:  # slerp or wiseft with slerp premerge
                premerged = _pairwise_slerp(tensors, weights)

            if method == "wiseft":
                # We can't scale per-key here because we need key context (A/B vs others).
                # So temporarily store premerged tensors; we'll apply WiSE-FT after we build whole dict.
                merged[k] = premerged
            else:
                merged[k] = premerged
                
        elif method == "task_arithmetic":
            # Task arithmetic: center deltas and recombine
            merged[k] = _task_arithmetic(tensors, weights, beta)
        elif method == "ties":
            # TIES with quantile trimming and weighted agreement averaging
            merged[k] = _ties_merging(tensors, weights, keep_top_p=ties_keep_top_p)

        else:
            raise ValueError(f"Unknown method: {method}")

    # If WiSE-FT, now scale A/B by √tau to get base ⊕ tau*Δ_soup
    if method == "wiseft":
        if not (0.0 <= wiseft_tau <= 1.0):
            raise ValueError("--wiseft-tau must be in [0,1]")
        merged = _apply_wiseft_scaling(merged, wiseft_tau)

    # Save as a new adapter safetensors
    out_path = out_dir / "adapter_model.safetensors"
    save_file(merged, str(out_path))
    print(f"[ModelSoup] Saved merged adapter to: {out_path}")
    
    # Copy adapter_config.json from the first adapter
    config_source = adapter_paths[0].parent / "adapter_config.json"
    config_dest = out_dir / "adapter_config.json"
    if config_source.exists():
        shutil.copy2(config_source, config_dest)
        print(f"[ModelSoup] Copied adapter_config.json to: {config_dest}")
    else:
        print(f"[ModelSoup] Warning: adapter_config.json not found at {config_source}")

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
                    help="Optional weights per adapter (same order)")
    ap.add_argument("--method", choices=["avg", "slerp", "task_arithmetic", "ties", "wiseft"], default="avg")
    ap.add_argument("--beta", type=float, default=0.5,
                    help="Beta parameter for task arithmetic (controls shared component, default: 0.5)")
    ap.add_argument("--trim-threshold", type=float, default=0.75,
                    help="[Deprecated for TIES] legacy threshold; use --ties-keep-top-p instead")
    ap.add_argument("--ties-keep-top-p", type=float, default=0.25,
                    help="Fraction to keep per adapter by magnitude in TIES (default: 0.25)")
    ap.add_argument("--wiseft-tau", type=float, default=0.7,
                    help="WiSE-FT mix with base (0..1). Scales LoRA delta by tau (via √tau on A/B).")
    ap.add_argument("--wiseft-premerge", choices=["avg", "slerp"], default="avg",
                    help="How to pre-merge the organ adapters before WiSE-FT scaling.")
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
    merge_adapters(
        adapter_paths, out_dir, args.method, args.weights, args.beta,
        args.trim_threshold, args.ties_keep_top_p,
        wiseft_tau=args.wiseft_tau, wiseft_premerge=args.wiseft_premerge,
    )

if __name__ == "__main__":
    main()
