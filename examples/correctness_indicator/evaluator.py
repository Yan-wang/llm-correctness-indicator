#!/usr/bin/env python3
"""
Evaluate WSR (Weighted Separation Ratio) reward function

This script computes the WSR reward from confidence scores and correctness labels.
It can work with the output of compute_confidence_scores.py directly or from saved data.

WSR formula:
    WSR = (p * μ₁ - (1-p) * μ₀) / √(p * σ₁² + (1-p) * σ₀²)
    
Where:
    p = proportion of correct traces
    μ₁ = mean confidence of correct traces
    μ₀ = mean confidence of incorrect traces
    σ₁² = variance of correct traces' confidence
    σ₀² = variance of incorrect traces' confidence
"""

import numpy as np
import argparse
import importlib.util
import os
from typing import List, Tuple, Optional, Dict, Any
try:
    # Preferred: return EvaluationResult for richer artifacts support
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    # Fallback: return a plain metrics dict if openevolve isn't importable in this context
    EvaluationResult = None


def generate_sample_artifacts(data: Dict[str, Any], num_samples: int = 5) -> str:
    """
    Generate a markdown string showing learned weights and problematic samples.
    """
    if not data or "samples" not in data or "weights" not in data:
        return "No diagnostic data available for artifacts."

    results = data["samples"]
    weights = data["weights"]

    lines = ["## Lasso Regression Diagnostics\n"]
    
    # 1. Show Weights
    lines.append("### Learned Feature Weights (Standardized)")
    lines.append("Features with 0.0000 were rejected. Focus on creating features that the Lasso model can use.\n")
    sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_weights:
        status = "✅" if abs(val) > 1e-5 else "❌"
        lines.append(f"- {status} **{name}**: `{val:.6f}`")
    lines.append("")

    # 2. Show Problematic Samples (Misclassifications)
    lines.append("## Error Analysis: Most Misclassified Samples\n")
    lines.append("Reviewing traces from the full training split where the current features are misleading the model.\n")
    
    # Filter and sort
    correct = sorted([r for r in results if r["is_correct"]], key=lambda x: x["score"]) # Ascending (low score = bad)
    incorrect = sorted([r for r in results if not r["is_correct"]], key=lambda x: x["score"], reverse=True) # Descending (high score = bad)
    
    lines.append("### False Negatives (Correct traces with the LOWEST scores)")
    if not correct:
        lines.append("No correct traces found.")
    else:
        for i, s in enumerate(correct[:num_samples]):
            feat_str = ", ".join([f"{k}: {v:.4f}" for k, v in s["features"].items()])
            lines.append(f"{i+1}. **Score: {s['score']:.4f}** (Should be high) | Features: `{feat_str}`")
    lines.append("")

    lines.append("### False Positives (Incorrect traces with the HIGHEST scores)")
    if not incorrect:
        lines.append("No incorrect traces found.")
    else:
        for i, s in enumerate(incorrect[:num_samples]):
            feat_str = ", ".join([f"{k}: {v:.4f}" for k, v in s["features"].items()])
            lines.append(f"{i+1}. **Score: {s['score']:.4f}** (Should be low) | Features: `{feat_str}`")
    lines.append("")

    # 3. Show Well-Classified for reference
    lines.append("## Reference: Best Classified Samples\n")
    correct_good = sorted([r for r in results if r["is_correct"]], key=lambda x: x["score"], reverse=True)
    incorrect_good = sorted([r for r in results if not r["is_correct"]], key=lambda x: x["score"])

    lines.append("### True Positives (Correct traces with highest scores)")
    for i, s in enumerate(correct_good[:num_samples]):
        feat_str = ", ".join([f"{k}: {v:.4f}" for k, v in s["features"].items()])
        lines.append(f"{i+1}. **Score: {s['score']:.4f}** | Features: `{feat_str}`")
    lines.append("")

    lines.append("### True Negatives (Incorrect traces with lowest scores)")
    for i, s in enumerate(incorrect_good[:num_samples]):
        feat_str = ", ".join([f"{k}: {v:.4f}" for k, v in s["features"].items()])
        lines.append(f"{i+1}. **Score: {s['score']:.4f}** | Features: `{feat_str}`")
    lines.append("")
        
    return "\n".join(lines)


import re

def evaluate(program_path: str = 'compute_confidence_scores.py'):
    """
    Evaluate the program by running compute_confidence_score() and calculating metrics
    on the full training split.
    """
    try:
        # Load config to determine reward metric
        reward_metric = "wsr"
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                content = f.read()
                # Simple regex to extract reward_metric from yaml
                match = re.search(r'reward_metric:\s*["\']?(\w+)["\']?', content)
                if match:
                    reward_metric = match.group(1)

        # Get absolute path if relative
        if not os.path.isabs(program_path):
            program_path = os.path.abspath(program_path)
        
        # Load the program module
        spec = importlib.util.spec_from_file_location("confidence_score_program", program_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load program from {program_path}")
            return 0.0
            
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # Check if the required function exists
        if not hasattr(program, "compute_confidence_score"):
            print(f"Error: program does not have 'compute_confidence_score' function")
            return 0.0
        
        # Call compute_confidence_score() to get validation scores and labels
        data = program.compute_confidence_score()
        
        if not data:
            print("Warning: No confidence scores returned from program")
            return 0.0

        # Handle backward compatibility and new format
        if isinstance(data, dict) and "samples" in data:
            results = data["samples"]
        else:
            # Old format or just samples
            results = data
            data = {"samples": results, "weights": {}}
            
        if isinstance(results[0], dict):
            confidence_scores = [(r["score"], r["is_correct"]) for r in results]
        else:
            confidence_scores = results
        
        # Compute metrics on validation split
        mse = float(compute_mse_from_scores(confidence_scores))
        wsr = float(compute_wsr_from_scores(confidence_scores))
        
        # Primary fitness signal: -mse or wsr
        if reward_metric == "mse":
            combined_score = -mse
        else:
            combined_score = wsr

        metrics = {
            "combined_score": combined_score,
            "wsr": wsr,
            "mse": mse,
            "error": "No error"
        }

        # Generate sample artifacts for LLM reasoning
        artifacts = {
            "num_validation_samples": len(confidence_scores),
        }
        if isinstance(data, dict):
            artifacts["classification_analysis"] = generate_sample_artifacts(data)

        if EvaluationResult is None:
            return {"metrics": metrics, "artifacts": artifacts}

        return EvaluationResult(
            metrics=metrics,
            artifacts=artifacts,
        )
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
        }
        if EvaluationResult is None:
            return {"combined_score": -1e10, "error": str(e)}

        return EvaluationResult(
            metrics={"combined_score": -1e10, "error": str(e)},
            artifacts=error_artifacts,
        )


def compute_mse_from_scores(confidence_scores: List[Tuple[float, bool]]) -> float:
    """Compute Mean Squared Error against {-1, 1} labels."""
    scores = np.array([s for s, _ in confidence_scores])
    labels = np.array([1.0 if is_correct else -1.0 for _, is_correct in confidence_scores])
    if len(scores) == 0:
        return 2.0  # Max possible MSE for {-1, 1} range
    return np.mean((scores - labels)**2)


def compute_wsr_from_scores(confidence_scores: List[Tuple[float, bool]]) -> float:
    """
    Compute WSR directly from confidence scores and correctness labels
    
    This is a convenience function that works with the output of compute_confidence_score()
    which returns List[Tuple[float, bool]] where the tuple is (confidence_score, is_correct).
    
    Args:
        confidence_scores: List of (confidence_score, is_correct) tuples
    
    Returns:
        WSR reward value (float)
    """
    # Extract weights and labels from the confidence scores
    weights = np.array([score for score, _ in confidence_scores])
    labels = np.array([1 if is_correct else 0 for _, is_correct in confidence_scores])
    
    # 类别统计
    p = np.mean(labels)  # 正确轨迹比例
    w1 = weights[labels == 1]  # 正确轨迹的权重
    w0 = weights[labels == 0]  # 错误轨迹的权重
    
    if len(w1) == 0 or len(w0) == 0:
        if len(w1) == 0:
            print(f"Warning: No correct traces found ({len(w0)} incorrect traces). WSR requires both correct and incorrect traces.")
        else:
            print(f"Warning: No incorrect traces found ({len(w1)} correct traces). WSR requires both correct and incorrect traces.")
        return 0.0  # 缺少某一类
    
    mu1 = np.mean(w1)
    mu0 = np.mean(w0)
    sigma1_sq = np.var(w1, ddof=1)  # 样本方差
    sigma0_sq = np.var(w0, ddof=1)
    
    numerator = p * mu1 - (1-p) * mu0
    denominator = np.sqrt(p * sigma1_sq + (1-p) * sigma0_sq)
    
    if denominator < 1e-10:
        return 0.0
    
    wsr = numerator / denominator
    return wsr


def load_scores_from_file(input_file: str) -> List[Tuple[float, bool]]:
    """
    Load confidence scores from a file (CSV format: confidence,is_correct)
    
    Args:
        input_file: Path to input file with confidence scores and correctness
    
    Returns:
        List of (confidence_score, is_correct) tuples
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(',')
                if len(parts) >= 2:
                    confidence = float(parts[0])
                    is_correct = parts[1].lower() in ('true', '1', 'yes')
                    results.append((confidence, is_correct))
            except ValueError:
                continue
    return results


def evaluate_wsr(
    input_file: Optional[str] = None,
    confidence_scores: Optional[List[Tuple[float, bool]]] = None,
    window_size: int = 2048,
    bottom_percent: float = 0.1
) -> float:
    """
    Evaluate WSR from input file or pre-computed confidence scores
    
    Args:
        input_file: Path to JSONL file with traces (optional, if confidence_scores not provided)
        confidence_scores: Pre-computed confidence scores (optional, if input_file not provided)
        window_size: Window size for confidence calculation (only used if input_file is provided)
        bottom_percent: Bottom percentile for confidence calculation (only used if input_file is provided)
    
    Returns:
        WSR reward value (float)
    """
    if confidence_scores is not None:
        # Use pre-computed scores
        scores = confidence_scores
    elif input_file is not None:
        # Compute scores from input file
        scores = compute_confidence_score(
            input_file=input_file,
            window_size=window_size,
            bottom_percent=bottom_percent
        )
    else:
        raise ValueError("Either input_file or confidence_scores must be provided")
    
    if not scores:
        return 0.0
    
    return compute_wsr_from_scores(scores)


# For standalone testing
if __name__ == "__main__":
    result = evaluate()
    print(result)
