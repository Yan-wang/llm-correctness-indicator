# EVOLVE-BLOCK-START
"""
Compute confidence features from trace data with logprobs.

Input format:
{
    "trace": "...", 
    "logprobs": [
        [selected_logprob, [logprob1, logprob2, logprob3, ...]],
        ...
    ],
    "is_correct": true,
    "problem_id": "...",
    "unique_id": "..."
}
"""

import json
import numpy as np
import os
import glob
from typing import List, Dict, Any, Tuple, Optional


def compute_token_confidences(logprobs: List[List[Any]]) -> List[float]:
    """Primitive: High-speed negative mean logprob per token."""
    valid_lps = [p[1] for p in logprobs if p and len(p) >= 2 and p[1]]
    if not valid_lps:
        return []
    
    try:
        lps_arr = np.array(valid_lps)
        if lps_arr.ndim == 2:
            return (-np.mean(lps_arr, axis=1)).tolist()
    except Exception:
        pass
    
    return [-np.mean(l) for l in valid_lps]


def get_sliding_window_stats(values: List[float], window_size: int = 256) -> Dict[str, List[float]]:
    """Primitive: Fully vectorized rolling statistics (O(N))."""
    if not values:
        return {"mean": [], "min": [], "std": []}
    
    arr = np.array(values)
    n = len(arr)
    if n < window_size:
        return {
            "mean": [float(np.mean(arr))] * n, 
            "min": [float(np.min(arr))] * n, 
            "std": [float(np.std(arr))] * n
        }

    # Rolling Mean using convolution
    weights = np.ones(window_size) / window_size
    means = np.convolve(arr, weights, mode='valid')

    # Pad to match original length
    padding_len = window_size - 1
    means_padded = np.concatenate([[means[0]] * padding_len, means])

    return {
        "mean": means_padded.tolist()
    }


def get_bottom_percentile_average(values: List[float], percentile: float = 0.1) -> float:
    """Primitive: Average the worst (lowest) X% of values in a sequence."""
    if not values:
        return 0.0
    arr = np.array(values)
    k = max(1, int(len(arr) * percentile))
    if k >= len(arr):
        return float(np.mean(arr))
    return float(np.mean(np.partition(arr, k-1)[:k]))


def calculate_features(logprobs: List[List[Any]]) -> Dict[str, float]:
    """
    Evolve this function: Propose a set of features (symbols).
    The system will automatically find the optimal weights to maximize voting probability.
    """
    token_confs = compute_token_confidences(logprobs)
    if not token_confs:
        return {"const": 1.0}

    window_stats = get_sliding_window_stats(token_confs, window_size=256)
    
    features = {
        "bottom_10_window_mean": get_bottom_percentile_average(window_stats["mean"], 0.1),
        "avg_token_conf": float(np.mean(token_confs)),
        "min_token_conf": float(np.min(token_confs)),
        "token_conf_std": float(np.std(token_confs))
    }
    
    return features


# EVOLVE-BLOCK-END
