# EVOLVE-BLOCK-START
"""
Compute confidence features from trace data with logprobs.

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
    
    try:
        lps_arr = np.array(valid_lps)
        if lps_arr.ndim == 2:
            return (-np.mean(lps_arr, axis=1)).tolist()
    except Exception:
        pass
    
    return [-np.mean(l) for l in valid_lps]


def compute_token_entropies(logprobs: List[List[Any]]) -> List[float]:
    """Primitive: High-speed Shannon entropy per token."""
    valid_lps = [p[1] for p in logprobs if p and len(p) >= 2 and p[1]]
    if not valid_lps:
        return []

    try:
        lps = np.array(valid_lps)
        if lps.ndim == 2:
            lps_shifted = lps - np.max(lps, axis=1, keepdims=True)
            probs = np.exp(lps_shifted)
            probs /= np.sum(probs, axis=1, keepdims=True)
            entropies = -np.sum(probs * np.log(probs + 1e-10), axis=1)
            return entropies.tolist()
    except Exception:
        pass

    entropies = []
    for l in valid_lps:
        l_arr = np.array(l)
        ps = np.exp(l_arr - np.max(l_arr))
        ps /= np.sum(ps)
        entropies.append(float(-np.sum(ps * np.log(ps + 1e-10))))
    return entropies


def get_sliding_window_stats(values: List[float], window_size: int = 256) -> Dict[str, List[float]]:
    """Primitive: Fully vectorized rolling statistics (O(N))."""
    if not values:
        return {"mean": [], "min": [], "std": []}
    
    arr = np.array(values)
    n = len(arr)
    if n < window_size:
        return {"mean": [float(np.mean(arr))] * n, "min": [float(np.min(arr))] * n, "std": [float(np.std(arr))] * n}

    # 1. Rolling Mean using convolution
    weights = np.ones(window_size) / window_size
    means = np.convolve(arr, weights, mode='valid')
    
    # 2. Rolling Std using Var = E[X^2] - (E[X])^2
    arr_sq = arr ** 2
    means_sq = np.convolve(arr_sq, weights, mode='valid')
    variance = means_sq - (means ** 2)
    stds = np.sqrt(np.maximum(variance, 0)) # Handle precision errors
    
    # 3. Fast Rolling Min (using a sliding window view optimization)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, window_shape=window_size)
    mins = np.min(windows, axis=1)

    # Pad to match original length
    padding_len = window_size - 1
    means_padded = np.concatenate([[means[0]] * padding_len, means])
    stds_padded = np.concatenate([[stds[0]] * padding_len, stds])
    mins_padded = np.concatenate([[mins[0]] * padding_len, mins])

    return {
        "mean": means_padded.tolist(), 
        "min": mins_padded.tolist(), 
        "std": stds_padded.tolist()
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


def calculate_confidence(logprobs: List[List[Any]]) -> Dict[str, float]:
    """
    Evolve this function: Propose a set of features (symbols) that might be useful
    for predicting correctness. Return them as a dictionary.
    
    Example:
    return {
        "weakest_link": ...,
        "avg_entropy": ...,
        "confidence_stability": ...
    }
    """
    token_confs = compute_token_confidences(logprobs)
    token_entropies = compute_token_entropies(logprobs)
    
    if not token_confs:
        return {"const": 0.0}

    # Example symbols
    window_stats = get_sliding_window_stats(token_confs, window_size=256)
    
    features = {
        "bottom_10_conf": get_bottom_percentile_average(window_stats["mean"], 0.1),
        "avg_conf": float(np.mean(token_confs)),
        "max_entropy": float(np.max(token_entropies)) if token_entropies else 0.0,
        "std_conf": float(np.std(token_confs))
    }
    
    return features


# EVOLVE-BLOCK-END


def lasso_regression(X: np.ndarray, y: np.ndarray, alpha: float = 0.05, iterations: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Simple Lasso coordinate descent implementation for feature selection and weighting."""
    n_samples, n_features = X.shape
    coeffs = np.zeros(n_features)
    
    # Standardize X and center y
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    for _ in range(iterations):
        for j in range(n_features):
            # Compute partial residual
            y_pred = X_scaled @ coeffs
            res = y_centered - y_pred + coeffs[j] * X_scaled[:, j]
            
            # Soft thresholding
            rho = np.dot(X_scaled[:, j], res)
            # Normalizing by n_samples to keep alpha scale consistent
            if rho < -alpha * n_samples:
                coeffs[j] = (rho + alpha * n_samples) / np.sum(X_scaled[:, j]**2)
            elif rho > alpha * n_samples:
                coeffs[j] = (rho - alpha * n_samples) / np.sum(X_scaled[:, j]**2)
            else:
                coeffs[j] = 0.0
                
    return coeffs, X_mean, X_std, y_mean


def compute_confidence_score(
    input_file: str = '/Users/wangyan/Desktop/correct-indicator/data/20260111_043209_traces_Qwen3-8B-AWQ_kv_auto_t0.6_p0.95_max32000_n64_swap4/train_split.jsonl',
) -> List[Tuple[float, bool]]:
    """
    Compute confidence scores by discovering features and running a deterministic Lasso regression.
    """
    all_features = []
    labels = []
    
    if '*' in input_file or '?' in input_file:
        input_files = sorted(glob.glob(input_file))
    else:
        input_files = [input_file] if os.path.exists(input_file) else []
    
    if not input_files:
        return []

    # 1. Extract features for all traces
    for file_path in input_files:
        with open(file_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    data = json.loads(line)
                    lps = data.get('logprobs', [])
                    if not lps: continue
                    
                    feats = calculate_confidence(lps)
                    all_features.append(feats)
                    labels.append(1.0 if data.get('is_correct') else -1.0)
                except:
                    continue

    if not all_features:
        return []

    # 2. Convert to matrix
    feature_names = sorted(list(all_features[0].keys()))
    X = np.array([[f.get(name, 0.0) for name in feature_names] for f in all_features])
    y = np.array(labels)

    # 3. Math Safety: Clean features before Lasso
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # 4. Run Lasso regression on full training split
    try:
        # alpha=0.05 forces sparsity
        coeffs, X_mean, X_std, y_mean = lasso_regression(X, y, alpha=0.05)
    except Exception:
        coeffs = np.zeros(X.shape[1])
        X_mean = np.mean(X, axis=0)
        X_std = np.ones(X.shape[1])
        y_mean = np.mean(y)

    # 5. Predict scores for all traces
    X_scaled = (X - X_mean) / X_std
    all_scores = (X_scaled @ coeffs + y_mean).tolist()

    # 6. Return rich results for all samples for artifact generation
    results = {
        "samples": [],
        "weights": {name: float(coeffs[j]) for j, name in enumerate(feature_names)}
    }
    for i in range(len(X)):
        results["samples"].append({
            "score": float(all_scores[i]),
            "is_correct": bool(y[i] > 0),
            "features": {name: float(X[i, j]) for j, name in enumerate(feature_names)}
        })
    return results


def main():
    results = compute_confidence_score()
    if results:
        print(f"Computed {len(results)} scores. First 5: {results[:5]}")


if __name__ == "__main__":
    main()
