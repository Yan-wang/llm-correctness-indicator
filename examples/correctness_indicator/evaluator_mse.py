#!/usr/bin/env python3
"""
Evaluate correctness indicator using Mean Squared Error (MSE)
"""

import numpy as np
import argparse
import importlib.util
import os
from typing import List, Tuple, Optional
try:
    # Preferred: return EvaluationResult for richer artifacts support
    from openevolve.evaluation_result import EvaluationResult
except Exception:  # pragma: no cover
    # Fallback: return a plain metrics dict if openevolve isn't importable in this context
    EvaluationResult = None


def evaluate(program_path: str = 'compute_confidence_scores.py', input_file: Optional[str] = None):
    """
    Evaluate MSE by running the program at the given path
    
    This function loads the program module and calls compute_confidence_score()
    to get confidence scores and labels, then computes and returns the negative MSE score.
    
    Args:
        program_path: Path to the Python program file (default: 'compute_confidence_scores.py')
                     The program should have a compute_confidence_score() function
        input_file: Optional path to a specific data file to evaluate on
    
    Returns:
        Negative MSE value (float) or EvaluationResult
    """
    try:
        # Get absolute path if relative
        if not os.path.isabs(program_path):
            # Assume relative to current working directory
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
        
        # Call compute_confidence_score() to get scores and labels
        # Pass input_file if provided and accepted by the program
        try:
            if input_file:
                confidence_scores = program.compute_confidence_score(input_file=input_file)
            else:
                confidence_scores = program.compute_confidence_score()
        except Exception as e:
            print(f"Program execution failed: {str(e)}")
            return EvaluationResult(
                metrics={"combined_score": -100.0, "error": 1.0},
                artifacts={"error": str(e)}
            )
        
        if not confidence_scores:
            print("Warning: No confidence scores returned from program")
            return EvaluationResult(
                metrics={"combined_score": -100.0, "error": 1.0},
                artifacts={"error": "No scores returned"}
            )
        
        # Compute and return MSE score
        mse_score = float(compute_mse_from_scores(confidence_scores))

        if EvaluationResult is None:
            return {"combined_score": mse_score, "error": 0.0}

        return EvaluationResult(
            metrics={"combined_score": mse_score, "error": 0.0},
            artifacts={},
        )
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        error_artifacts = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "full_traceback": traceback.format_exc(),
            "suggestion": "Check for syntax errors or missing imports in the generated code"
        }
        if EvaluationResult is None:
            return {"combined_score": -100.0, "error": 1.0}

        return EvaluationResult(
            metrics={"combined_score": -100.0, "error": 1.0},
            artifacts=error_artifacts,
        )


def compute_mse_from_scores(confidence_scores: List[Tuple[float, bool]]) -> float:
    """
    Compute negative MSE between confidence scores and correctness labels
    
    Labels: 1.0 for correct, -1.0 for incorrect.
    Higher score is better (maximizing negative MSE minimizes MSE).
    
    Args:
        confidence_scores: List of (confidence_score, is_correct) tuples
    
    Returns:
        Negative MSE value (float)
    """
    # Extract scores and labels
    scores = np.array([score for score, _ in confidence_scores])
    labels = np.array([1.0 if is_correct else -1.0 for _, is_correct in confidence_scores])
    
    if len(scores) == 0:
        return -1000.0
        
    # Compute Mean Squared Error
    mse = np.mean((scores - labels)**2)
    
    # Return negative MSE so that maximization minimizes error
    return -float(mse)


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


def evaluate_score(
    input_file: Optional[str] = None,
    confidence_scores: Optional[List[Tuple[float, bool]]] = None,
    window_size: int = 2048,
    bottom_percent: float = 0.1
) -> float:
    """
    Evaluate negative MSE from input file or pre-computed confidence scores
    
    Args:
        input_file: Path to JSONL file with traces (optional, if confidence_scores not provided)
        confidence_scores: Pre-computed confidence scores (optional, if input_file not provided)
        window_size: Window size for confidence calculation (only used if input_file is provided)
        bottom_percent: Bottom percentile for confidence calculation (only used if input_file is provided)
    
    Returns:
        Negative MSE value (float)
    """
    if confidence_scores is not None:
        # Use pre-computed scores
        scores = confidence_scores
    elif input_file is not None:
        # We need to import compute_confidence_score here or from program
        # For simplicity in this context, we assume confidence_scores are provided
        # or the program is loaded via evaluate()
        print("Warning: input_file evaluation in evaluate_score requires compute_confidence_score which is defined in the program being evaluated.")
        return 0.0
    else:
        raise ValueError("Either input_file or confidence_scores must be provided")
    
    if not scores:
        return -1000.0
    
    return compute_mse_from_scores(scores)


# For standalone testing
if __name__ == "__main__":
    result = evaluate()
    print(result)
