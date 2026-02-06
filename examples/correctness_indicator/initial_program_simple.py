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
from collections import Counter
import random


def compute_confidence_from_jsonl_logprobs(logprobs: List[List[Any]]) -> List[float]:
    """
    Compute confidence score from logprobs in JSONL format
    
    The logprobs format:
    [[selected_logprob, [logprob1, logprob2, ...]], ...]
    
    - For each position, compute mean of all logprobs in the list
    - Confidence is negative of mean logprob
    - Skip positions with no logprobs (matching deepconf_offline.py behavior)
    
    Args:
        logprobs: List of [selected_logprob, [logprob1, logprob2, ...]]
                  where logprobs_list contains ALL top-k logprobs from vLLM output
    
    Returns:
        List of confidence scores (one per token position with valid logprobs)
    """
    confs = []
    for position_data in logprobs:
        if position_data and len(position_data) >= 2:
            # position_data is [selected_logprob, [logprob1, logprob2, ...]]
            logprobs_list = position_data[1]
            if logprobs_list:
                # Compute mean of all logprobs in the list
                # This matches: np.mean([lp.logprob for lp in token_logprobs.values()])
                # Both extract ALL logprobs from the same vLLM token_logprobs dict
                mean_logprob = np.mean(logprobs_list)
                # Confidence is negative of mean logprob (exactly as in deepconf_offline.py)
                confs.append(round(-mean_logprob, 3))
            # Skip positions with empty logprobs_list (matching deepconf_offline.py behavior)
    return confs


def calculate_confidence(logprobs: List[List[Any]]) -> float:
    """
    Calculate raw confidence from logprobs.
    This version computes the mean token confidence as a starting point.
    LLM can evolve this logic to use windowing, different percentiles, etc.
    """
    token_confs = compute_confidence_from_jsonl_logprobs(logprobs)
    if not token_confs:
        return 0.0
    
    # Example: Return mean token confidence as a raw score
    return float(np.mean(token_confs))


# EVOLVE-BLOCK-END


def compute_confidence_score(
    input_file: str = './data/20260111_043209_traces_Qwen3-8B-AWQ_kv_auto_t0.6_p0.95_max32000_n64_swap4/train_split.jsonl',
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

    # 2. Grid search for best Tanh normalization (minimizing MSE)
    raw_arr = np.array([item['raw'] for item in raw_data])
    label_arr = np.array([item['label'] for item in raw_data])
    
    best_mse = float('inf')
    best_params = (0.0, 1.0)
    
    # Define search space for center (based on raw score range) and scale
    centers = np.linspace(np.min(raw_arr), np.max(raw_arr), 25)
    scales = np.linspace(-5.0, 5.0, 51)
    
    for c in centers:
        for s in scales:
            normalized = np.tanh(s * (raw_arr - c))
            mse = np.mean((normalized - label_arr)**2)
            if mse < best_mse:
                best_mse = mse
                best_params = (c, s)
    
    best_center, best_scale = best_params
    
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