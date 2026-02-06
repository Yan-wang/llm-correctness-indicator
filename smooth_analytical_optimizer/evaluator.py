#!/usr/bin/env python3
"""
Evaluator for Correctness Indicator task.
Optimized for Voting Accuracy on Train and Validation sets.
"""

import numpy as np
import importlib.util
import os
import json
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    EvaluationResult = None

DATA_DIR = '/Users/wangyan/Desktop/correct-indicator/data/merged_split'

def compute_voting_metrics(samples: List[Dict[str, Any]], fixed_indices_path: str) -> Dict[str, float]:
    """
    Groups traces by problem_id and uses test cases to calculate voting accuracy.
    """
    # 1. Group by unique_id
    trace_map = {s.get("unique_id"): s for s in samples}
    
    # 2. Identify ground truth for each problem from the samples
    problem_correct_answers = {}
    for s in samples:
        pid = s.get("problem_id")
        if pid not in problem_correct_answers and s.get("is_correct"):
            problem_correct_answers[pid] = str(s.get("extracted_answer"))
    
    # 3. Load cases
    if not os.path.exists(fixed_indices_path):
        return {"voting_accuracy": 0.0, "smooth_reward": 0.0, "margin": 0.0}

    with open(fixed_indices_path, 'r') as f:
        cases = json.load(f)

    correct_votes = 0
    processed_cases = 0
    smooth_rewards = []
    margins = []
    
    # Track details for artifacts
    case_details = []
    
    for case in cases:
        pid = case["problem_id"]
        uids = case["trace_ids"]
        correct_ans = problem_correct_answers.get(pid)
        
        sampled = [trace_map[uid] for uid in uids if uid in trace_map]
        if not sampled: continue
        
        answers = [str(t.get("extracted_answer", "")) for t in sampled]
        weights = [float(t.get("score", 0.0)) for t in sampled]
        
        # Answer sums
        answer_sums = defaultdict(float)
        for ans, w in zip(answers, weights):
            answer_sums[ans] += w
            
        if not answer_sums: continue
        
        # 1. Accuracy
        voted_answer = max(answer_sums.keys(), key=lambda x: answer_sums[x])
        is_correct = (voted_answer == correct_ans)
        if is_correct:
            correct_votes += 1
        
        # 2. Smooth Reward (Softmax probability of correct answer)
        all_sums = np.array(list(answer_sums.values()))
        max_sum = np.max(all_sums)
        exp_sums = np.exp(all_sums - max_sum)
        
        correct_weight_sum = answer_sums.get(correct_ans, 0.0)
        prob_correct = np.exp(correct_weight_sum - max_sum) / np.sum(exp_sums)
        smooth_rewards.append(prob_correct)

        # 3. Margin
        other_weights = [v for ans, v in answer_sums.items() if ans != correct_ans]
        max_wrong = max(other_weights) if other_weights else 0.0
        margin = correct_weight_sum - max_wrong
        margins.append(margin)

        case_details.append({
            "problem_id": pid,
            "is_correct": is_correct,
            "prob_correct": float(prob_correct),
            "margin": float(margin),
            "voted": voted_answer,
            "target": correct_ans,
            "correct_weight": float(correct_weight_sum),
            "max_wrong_weight": float(max_wrong)
        })

        processed_cases += 1
            
    # Create artifacts: Top 5 failures by smooth reward
    failures = [c for c in case_details if not c["is_correct"]]
    failures.sort(key=lambda x: x["prob_correct"])
    
    artifacts = {
        "wrong_cases": failures[:5]
    }
            
    return {
        "voting_accuracy": float(correct_votes) / processed_cases if processed_cases > 0 else 0.0,
        "smooth_reward": float(np.mean(smooth_rewards)) if smooth_rewards else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "_case_artifacts": artifacts
    }

def _run_eval_on_file(program, input_file: str, indices_file: str, train_file: str) -> Dict[str, float]:
    """Helper to run evaluation on a specific split."""
    if not os.path.exists(input_file):
        return {}
        
    data = program.compute_confidence_score(input_file=input_file, train_file=train_file)
    if not data or "samples" not in data:
        return {}
        
    return compute_voting_metrics(data["samples"], indices_file)

def evaluate(program_path: str = 'compute_confidence_scores.py', input_file: Optional[str] = None):
    """Main entry point for OpenEvolve."""
    try:
        if not os.path.isabs(program_path):
            program_path = os.path.abspath(program_path)
        spec = importlib.util.spec_from_file_location("confidence_score_program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        train_file = os.path.join(DATA_DIR, 'train_split.jsonl')
        train_indices = os.path.join(DATA_DIR, 'train_test_cases.json')
        val_file = os.path.join(DATA_DIR, 'val_split.jsonl')
        val_indices = os.path.join(DATA_DIR, 'val_test_cases.json')

        with ThreadPoolExecutor(max_workers=2) as executor:
            train_future = executor.submit(_run_eval_on_file, program, train_file, train_indices, train_file)
            val_future = executor.submit(_run_eval_on_file, program, val_file, val_indices, train_file)
            metrics_raw = train_future.result()
            val_metrics_raw = val_future.result()
        
        # Extract artifacts
        train_artifacts = metrics_raw.pop("_case_artifacts", {})
        val_artifacts = val_metrics_raw.pop("_case_artifacts", {})
        
        metrics = {
            "voting_accuracy": metrics_raw.get("voting_accuracy", 0.0),
            "smooth_reward": metrics_raw.get("smooth_reward", 0.0),
            "margin": metrics_raw.get("margin", 0.0),
            "val_voting_accuracy": val_metrics_raw.get("voting_accuracy", 0.0),
            "val_smooth_reward": val_metrics_raw.get("smooth_reward", 0.0),
            "combined_score": metrics_raw.get("smooth_reward", 0.0)
        }

        # Combine artifacts for the LLM
        all_artifacts = {
            "train_failures": train_artifacts.get("wrong_cases", []),
            "val_failures": val_artifacts.get("wrong_cases", [])
        }

        if EvaluationResult is None:
            return metrics

        return EvaluationResult(metrics=metrics, artifacts=all_artifacts)
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        if EvaluationResult is None:
            return {"combined_score": -1e10, "error": str(e)}
        return EvaluationResult(metrics={"combined_score": -1e10, "error": str(e)}, artifacts={})

if __name__ == "__main__":
    import sys
    res = evaluate(sys.argv[1] if len(sys.argv) > 1 else 'examples/correctness_indicator/initial_program.py')
    print(json.dumps(res.metrics if hasattr(res, 'metrics') else res, indent=2))
