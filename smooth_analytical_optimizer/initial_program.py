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
        return {"mean": [float(np.mean(arr))] * n, "min": [float(np.min(arr))] * n, "std": [float(np.std(arr))] * n}

    # Rolling Mean using convolution
    weights = np.ones(window_size) / window_size
    means = np.convolve(arr, weights, mode='valid')
    
    # Pad to match original length
    padding_len = window_size - 1
    means_padded = np.concatenate([[means[0]] * padding_len, means])

    return {
        "mean": means_padded.tolist(),
        "min": [], # Placeholder for consistency
        "std": []  # Placeholder for consistency
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


def calculate_confidence(logprobs: List[List[Any]]) -> float:
    """
    Evolve this function: Construct a final confidence score directly.
    Higher score means more confident the answer is correct.
    """
    token_confs = compute_token_confidences(logprobs)
    if not token_confs:
        return 0.0

    # Example: Combination of bottom-10 confidence and average confidence
    window_stats = get_sliding_window_stats(token_confs, window_size=256)
    bottom_10 = get_bottom_percentile_average(window_stats["mean"], 0.1)
    avg_conf = float(np.mean(token_confs))
    
    # Direct combination
    score = bottom_10 * 0.7 + avg_conf * 0.3
    
    return score


# EVOLVE-BLOCK-END


def _get_trace_files(input_file: str) -> List[str]:
    """Helper to resolve manifest or direct data files."""
    if not os.path.exists(input_file):
        if '*' in input_file or '?' in input_file:
            return sorted(glob.glob(input_file))
        return []

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            # If it's valid JSON, it's a data file
            json.loads(content)
            return [input_file]
        except json.JSONDecodeError:
            # Check if it's JSONL or a manifest
            lines = content.splitlines()
            if not lines: return []
            try:
                # If first line is JSON, it's JSONL
                json.loads(lines[0])
                return [input_file]
            except json.JSONDecodeError:
                # Otherwise it's a manifest
                base_dir = os.path.dirname(input_file)
                trace_files = []
                for line in lines:
                    rel_path = line.strip()
                    if rel_path:
                        abs_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(abs_path):
                            trace_files.append(abs_path)
                return trace_files


def compute_confidence_score(
    input_file: str = '/Users/wangyan/Desktop/correct-indicator/data/merged_split/train_split.jsonl',
    train_file: str = '/Users/wangyan/Desktop/correct-indicator/data/merged_split/train_split.jsonl'
) -> Dict[str, Any]:
    """
    Compute confidence scores directly without regression.
    """
    trace_files = _get_trace_files(input_file)
    if not trace_files:
        return {"samples": []}

    results = {"samples": []}
    for file_path in trace_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                try:
                    # Single JSON problem object
                    prob_data = json.loads(content)
                    traces = prob_data.get("traces", [prob_data]) if isinstance(prob_data, dict) else []
                except json.JSONDecodeError:
                    # JSONL
                    traces = []
                    for line in content.splitlines():
                        if line.strip():
                            try: traces.append(json.loads(line))
                            except: continue
                
                for data in traces:
                    lps = data.get('logprobs', [])
                    if not lps: continue
                    
                    # Direct score calculation
                    score = calculate_confidence(lps)
                    
                    results["samples"].append({
                        "score": float(score),
                        "is_correct": bool(data.get('is_correct')),
                        "problem_id": data.get("problem_id", "unknown"),
                        "unique_id": data.get("unique_id", "unknown"),
                        "extracted_answer": data.get("extracted_answer", "unknown")
                    })
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            continue

    return results


def main():
    results = compute_confidence_score()
    if results and results["samples"]:
        print(f"Computed {len(results['samples'])} scores. First 5 samples:")
        for s in results["samples"][:5]:
            print(f"  ID: {s['unique_id']}, Score: {s['score']:.4f}, Correct: {s['is_correct']}")


if __name__ == "__main__":
    main()
