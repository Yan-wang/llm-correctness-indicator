#!/usr/bin/env python3
"""
Evaluator for Correctness Indicator task.
Optimized for Feature Evolution with Numerical Weight Optimization.
Maximizes collective Softmax Probability of correct answers.
"""

import numpy as np
import importlib.util
import os
import json
import re
import glob
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

try:
    from openevolve.evaluation_result import EvaluationResult
except Exception:
    EvaluationResult = None

DATA_DIR = '/Users/wangyan/Desktop/correct-indicator/data/merged_split'

def softmax_loss(weights, feature_matrix, group_info, l1_penalty=0.01):
    """
    Custom objective: -sum(log(prob_correct)).
    Optimizes the collective probability of the correct answer winning.
    """
    scores = feature_matrix @ weights
    
    total_neg_log_prob = 0
    for problem_id, info in group_info.items():
        correct_answer = info["correct_answer"]
        # Maps answer string to list of indices in feature_matrix
        answer_to_indices = info["answer_to_indices"]
        
        # Calculate sum of weights for each unique answer
        answer_sums = []
        correct_sum = 0
        
        for ans, indices in answer_to_indices.items():
            ans_sum = np.sum(scores[indices])
            answer_sums.append(ans_sum)
            if ans == correct_answer:
                correct_sum = ans_sum
        
        # Log-Sum-Exp for numerical stability
        answer_sums = np.array(answer_sums)
        max_val = np.max(answer_sums)
        log_prob_correct = (correct_sum - max_val) - np.log(np.sum(np.exp(answer_sums - max_val)))
        total_neg_log_prob -= log_prob_correct
        
    # Add L1 penalty for sparsity
    return total_neg_log_prob / len(group_info) + l1_penalty * np.sum(np.abs(weights))

def compute_voting_metrics_with_weights(samples, indices_path, weights, feature_names, scaler=None):
    """
    Applies weights to features and calculates voting metrics.
    """
    if not os.path.exists(indices_path):
        return {"voting_accuracy": 0.0, "smooth_reward": 0.0, "margin": 0.0}

    with open(indices_path, 'r') as f:
        cases = json.load(f)

    # 1. Map trace unique_id to its feature vector
    trace_map = {s["unique_id"]: s for s in samples}
    
    # 2. Pre-calculate ground truth
    problem_correct_answers = {}
    for s in samples:
        pid = s["problem_id"]
        if pid not in problem_correct_answers and s["is_correct"]:
            problem_correct_answers[pid] = str(s["extracted_answer"])

    correct_votes = 0
    smooth_rewards = []
    margins = []
    case_details = []
    
    for case in cases:
        pid = case["problem_id"]
        correct_ans = problem_correct_answers.get(pid)
        
        # Extract features for all traces in this case
        case_traces = [trace_map[uid] for uid in case["trace_ids"] if uid in trace_map]
        if not case_traces: continue
        
        # Calculate scores: x @ weights
        answers = [str(t["extracted_answer"]) for t in case_traces]
        feature_vals = np.array([[t["features"].get(name, 0.0) for name in feature_names] for t in case_traces])
        
        # Apply scaling if provided
        if scaler:
            feature_vals = scaler.transform(feature_vals)
            
        trace_scores = feature_vals @ weights
        
        # Answer sums
        answer_sums = defaultdict(float)
        for ans, score in zip(answers, trace_scores):
            answer_sums[ans] += score
            
        if not answer_sums: continue
        
        # Metrics
        voted_answer = max(answer_sums.keys(), key=lambda x: answer_sums[x])
        is_correct = (voted_answer == correct_ans)
        if is_correct: correct_votes += 1
        
        all_sums = np.array(list(answer_sums.values()))
        max_sum = np.max(all_sums)
        prob_correct = np.exp(answer_sums.get(correct_ans, 0.0) - max_sum) / np.sum(np.exp(all_sums - max_sum))
        smooth_rewards.append(prob_correct)

        other_weights = [v for ans, v in answer_sums.items() if ans != correct_ans]
        margin = answer_sums.get(correct_ans, 0.0) - (max(other_weights) if other_weights else 0.0)
        margins.append(margin)

        case_details.append({
            "problem_id": pid,
            "is_correct": is_correct,
            "prob_correct": float(prob_correct),
            "margin": float(margin),
            "voted": voted_answer,
            "target": correct_ans
        })

    return {
        "voting_accuracy": float(correct_votes) / len(cases) if cases else 0.0,
        "smooth_reward": float(np.mean(smooth_rewards)) if smooth_rewards else 0.0,
        "margin": float(np.mean(margins)) if margins else 0.0,
        "case_details": case_details
    }

def _get_trace_files(input_file: str) -> List[str]:
    """Helper to resolve manifest or direct data files."""
    if not os.path.exists(input_file):
        if '*' in input_file or '?' in input_file:
            return sorted(glob.glob(input_file))
        return []

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            json.loads(content)
            return [input_file]
        except json.JSONDecodeError:
            lines = content.splitlines()
            if not lines: return []
            try:
                json.loads(lines[0])
                return [input_file]
            except json.JSONDecodeError:
                base_dir = os.path.dirname(input_file)
                trace_files = []
                for line in lines:
                    rel_path = line.strip()
                    if rel_path:
                        abs_path = os.path.join(base_dir, rel_path)
                        if os.path.exists(abs_path):
                            trace_files.append(abs_path)
                return trace_files

def extract_features(program, input_file: str) -> Dict[str, Any]:
    """
    Extracts raw features for all traces using the program's calculate_features function.
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
                    prob_data = json.loads(content)
                    traces = prob_data.get("traces", [prob_data]) if isinstance(prob_data, dict) else []
                except json.JSONDecodeError:
                    traces = []
                    for line in content.splitlines():
                        if line.strip():
                            try: traces.append(json.loads(line))
                            except: continue
                
                for data in traces:
                    lps = data.get('logprobs', [])
                    if not lps: continue
                    
                    # Call the evolved feature function directly
                    features = program.calculate_features(lps)
                    
                    results["samples"].append({
                        "features": features,
                        "is_correct": bool(data.get('is_correct')),
                        "problem_id": data.get("problem_id", "unknown"),
                        "unique_id": data.get("unique_id", "unknown"),
                        "extracted_answer": data.get("extracted_answer", "unknown")
                    })
        except Exception:
            continue

    return results

def evaluate(program_path: str = 'compute_confidence_scores.py', input_file: Optional[str] = None):
    try:
        # 1. Load evolved program
        if not os.path.isabs(program_path):
            program_path = os.path.abspath(program_path)
        spec = importlib.util.spec_from_file_location("feature_program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)
        
        # 2. Extract raw features for both splits
        train_file = os.path.join(DATA_DIR, 'train_split.jsonl')
        val_file = os.path.join(DATA_DIR, 'val_split.jsonl')
        
        train_data = extract_features(program, train_file)
        val_data = extract_features(program, val_file)
        
        if not train_data["samples"]:
            return {"combined_score": -1e10, "error": "No training samples extracted"}

        # 3. Prepare Feature Matrix and Grouping Info for Optimizer
        feature_names = sorted(list(train_data["samples"][0]["features"].keys()))
        X_train_raw = np.array([[s["features"].get(name, 0.0) for name in feature_names] for s in train_data["samples"]])
        
        # FEATURE SCALING: Essential for L1 Regularization to be effective
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        
        # Group samples by answer choice within each problem
        group_info = {}
        for idx, s in enumerate(train_data["samples"]):
            pid = s["problem_id"]
            ans = str(s["extracted_answer"])
            if pid not in group_info:
                # Find the correct answer for this problem
                correct_ans = next((str(tmp["extracted_answer"]) for tmp in train_data["samples"] if tmp["problem_id"] == pid and tmp["is_correct"]), None)
                group_info[pid] = {"correct_answer": correct_ans, "answer_to_indices": defaultdict(list)}
            group_info[pid]["answer_to_indices"][ans].append(idx)

        # 4. Numerical Optimization (maximize softmax probability)
        # We optimize the scaled features
        initial_weights = np.zeros(len(feature_names))
        res = minimize(
            softmax_loss, 
            initial_weights, 
            args=(X_train, group_info),
            method='L-BFGS-B',
            options={'maxiter': 100}
        )
        optimized_weights = res.x

        # 5. Evaluate final metrics with optimized weights and the SAME scaler
        train_indices = os.path.join(DATA_DIR, 'train_test_cases.json')
        val_indices = os.path.join(DATA_DIR, 'val_test_cases.json')
        
        train_metrics = compute_voting_metrics_with_weights(train_data["samples"], train_indices, optimized_weights, feature_names, scaler=scaler)
        val_metrics = compute_voting_metrics_with_weights(val_data["samples"], val_indices, optimized_weights, feature_names, scaler=scaler)
        
        # 6. Build Result Object
        metrics = {
            "voting_accuracy": train_metrics["voting_accuracy"],
            "smooth_reward": train_metrics["smooth_reward"],
            "margin": train_metrics["margin"],
            "val_voting_accuracy": val_metrics["voting_accuracy"],
            "val_smooth_reward": val_metrics["smooth_reward"],
            "combined_score": train_metrics["smooth_reward"]
        }

        weight_report = {name: float(w) for name, w in zip(feature_names, optimized_weights)}
        
        failures = [c for c in train_metrics["case_details"] if not c["is_correct"]]
        failures.sort(key=lambda x: x["prob_correct"])
        
        artifacts = {
            "optimized_weights": weight_report,
            "train_failures": failures[:10],  # Increased to 10 for better training signal
        }

        if EvaluationResult is None:
            return metrics

        return EvaluationResult(metrics=metrics, artifacts=artifacts)
        
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
    if hasattr(res, 'metrics'):
        print(json.dumps(res.metrics, indent=2))
        print("\nOptimized Weights:")
        print(json.dumps(res.artifacts["optimized_weights"], indent=2))
    else:
        print(json.dumps(res, indent=2))
