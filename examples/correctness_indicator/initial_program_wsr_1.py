# EVOLVE-BLOCK-START
"""
Compute confidence scores from trace data with logprobs

This script reads a JSONL file containing traces with logprobs and computes
confidence scores.

Input format of JSONL file:
{
    "trace": "...", 
    "logprobs": [
        [selected_logprob, [logprob1, logprob2, logprob3, ...]],
        ...
    ],
    "is_correct": true
}

"""

import json
import argparse
import numpy as np
import glob
import os
from typing import List, Dict, Any, Tuple, Optional


def compute_token_confidences(logprobs: List[List[Any]]) -> List[float]:
    """Primitive: High-speed negative mean logprob per token."""
    valid_lps = [p[1] for p in logprobs if p and len(p) >= 2 and p[1]]
    if not valid_lps:
        return []
    
    # Optimization: Use 2D NumPy if top-k size is uniform
    try:
        lps_arr = np.array(valid_lps)
        if lps_arr.ndim == 2:
            return (-np.mean(lps_arr, axis=1)).tolist()
    except Exception:
        pass
    
    # Fallback for non-uniform top-k
    return [-np.mean(l) for l in valid_lps]


def compute_token_entropies(logprobs: List[List[Any]]) -> List[float]:
    """Primitive: High-speed Shannon entropy per token."""
    valid_lps = [p[1] for p in logprobs if p and len(p) >= 2 and p[1]]
    if not valid_lps:
        return []

    # Optimization: Process as matrix if uniform
    try:
        lps = np.array(valid_lps)
        if lps.ndim == 2:
            # Shift logprobs for numerical stability (max becomes 0)
            lps_shifted = lps - np.max(lps, axis=1, keepdims=True)
            probs = np.exp(lps_shifted)
            probs /= np.sum(probs, axis=1, keepdims=True)
            entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropies.tolist()
    except Exception:
        pass

    # Fallback for non-uniform
    entropies = []
    for l in valid_lps:
        l_arr = np.array(l)
        ps = np.exp(l_arr - np.max(l_arr))
        ps /= np.sum(ps)
        entropies.append(float(-np.sum(ps * np.log(ps + 1e-10))))
    return entropies


def get_sliding_window_stats(values: List[float], window_size: int = 256) -> Dict[str, List[float]]:
    """Primitive: Optimized rolling statistics using NumPy."""
    if not values:
        return {"mean": [], "min": [], "std": []}
    
    arr = np.array(values)
    n = len(arr)
    if n < window_size:
        return {"mean": [float(np.mean(arr))] * n, "min": [float(np.min(arr))] * n, "std": [float(np.std(arr))] * n}

    # Fast rolling mean using convolution
    weights = np.ones(window_size) / window_size
    means = np.convolve(arr, weights, mode='valid')
    # Pad to match original length by repeating the first valid mean
    padding = [means[0]] * (window_size - 1)
    means_padded = np.concatenate([padding, means])

    return {"mean": means_padded.tolist(), "min": means_padded.tolist(), "std": [0.0] * n}


def get_bottom_percentile_average(values: List[float], percentile: float = 0.1) -> float:
    """Primitive: Average the worst (lowest) X% of values in a sequence."""
    if not values:
        return 0.0
    arr = np.array(values)
    k = max(1, int(len(arr) * percentile))
    # Use partition for O(N) performance instead of O(N log N) sort
    if k >= len(arr):
        return float(np.mean(arr))
    return float(np.mean(np.partition(arr, k-1)[:k]))


def calculate_confidence(logprobs: List[List[Any]]) -> float:
    """
    Calculate raw confidence from logprobs.
    Now using the provided primitives to compute a more robust baseline:
    The average of the bottom 10% of 256-token window means.
    """
    # 1. Extract base token confidences
    token_confs = compute_token_confidences(logprobs)
    if not token_confs:
        return 0.0
    
    # 2. Compute sliding window means (smoothing)
    window_stats = get_sliding_window_stats(token_confs, window_size=256)
    window_means = window_stats["mean"]
    
    # 3. Focus on the "weakest links" (bottom 10% of windows)
    raw_score = get_bottom_percentile_average(window_means, percentile=0.1)
    
    return float(raw_score)


# EVOLVE-BLOCK-END


def compute_confidence_score(
    input_file: str = '/Users/wangyan/Desktop/correct-indicator/data/20260111_043209_traces_Qwen3-8B-AWQ_kv_auto_t0.6_p0.95_max32000_n64_swap4/train_split.jsonl',
) -> List[Tuple[float, bool]]:
    """
    Compute confidence scores from trace data and automatically find 
    the best Tanh normalization parameters to minimize MSE.
    """
    raw_data = []
    
    # Check if input_file contains wildcards
    if '*' in input_file or '?' in input_file:
        matching_files = sorted(glob.glob(input_file))
        if not matching_files:
            return []
        input_files = matching_files
    else:
        if not os.path.exists(input_file):
            return []
        input_files = [input_file]
    
    # 1. Collect raw scores and labels
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace_data = json.loads(line)
                    logprobs = trace_data.get('logprobs', [])
                    if logprobs:
                        raw_score = calculate_confidence(logprobs)
                        is_correct = bool(trace_data.get('is_correct', False))
                        raw_data.append({'raw': raw_score, 'label': 1.0 if is_correct else -1.0})
                    else:
                        continue
                except Exception:
                    continue

    if not raw_data:
        return []

    # 2. Vectorized Grid search for best Tanh normalization (minimizing MSE)
    raw_arr = np.array([item['raw'] for item in raw_data])
    label_arr = np.array([item['label'] for item in raw_data])
    
    # Define search space
    centers = np.linspace(np.min(raw_arr), np.max(raw_arr), 20)
    scales = np.linspace(-5.0, 5.0, 41)
    
    # Use broadcasting for fast grid search
    # (scales, centers, samples)
    norm_matrix = np.tanh(scales[:, None, None] * (raw_arr[None, None, :] - centers[None, :, None]))
    mse_matrix = np.mean((norm_matrix - label_arr[None, None, :])**2, axis=2)
    
    best_idx = np.unravel_index(np.argmin(mse_matrix), mse_matrix.shape)
    best_scale = scales[best_idx[0]]
    best_center = centers[best_idx[1]]
    best_mse = mse_matrix[best_idx]
    
    best_params = (best_center, best_scale)
    
    # 3. Apply best normalization and return
    results = []
    for item in raw_data:
        norm_score = np.tanh(best_scale * (item['raw'] - best_center))
        # Map label back to boolean for compatibility with evaluator
        is_correct = True if item['label'] > 0 else False
        results.append((float(norm_score), is_correct))
        
    print(f"Optimal Normalization: Center={best_center:.4f}, Scale={best_scale:.4f}, Best MSE={best_mse:.4f}")
    return results


def main():
    print(compute_confidence_score())


if __name__ == "__main__":
    main()