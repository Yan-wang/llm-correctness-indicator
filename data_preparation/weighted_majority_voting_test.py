import json
import os
import numpy as np
import importlib.util
import sys
import time
import re
import concurrent.futures
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# =============================================================================
# IMPORT BEST PROGRAM
# =============================================================================

best_program_path = "/Users/wangyan/Desktop/correct-indicator/examples/correctness_indicator/openevolve_output_alpha_0_wsr/checkpoints/checkpoint_35/best_program_resolved.py"
spec = importlib.util.spec_from_file_location("best_program", best_program_path)
best_program = importlib.util.module_from_spec(spec)
sys.modules["best_program"] = best_program
spec.loader.exec_module(best_program)

calculate_confidence_best = best_program.calculate_confidence

# =============================================================================
# VOTING AND EVALUATION
# =============================================================================

def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    if not answers: return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights: return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def process_single_test_case(tc_id, lines):
    """Worker function to process a single test case."""
    traces = [json.loads(line) for line in lines]
    
    # Determine ground truth
    correct_answer = None
    for t in traces:
        if t.get('is_correct'):
            correct_answer = str(t.get('extracted_answer'))
            break
    
    if correct_answer is None:
        return tc_id, None
        
    prob_name = traces[0].get('problem_name', 'unknown')
    answers = [str(t.get('extracted_answer', '')) for t in traces]
    
    # 1. Simple Majority
    vote_counts = Counter(answers)
    simple_voted_answer = vote_counts.most_common(1)[0][0]
    is_simple_correct = (simple_voted_answer == correct_answer)
    
    # 2. Best Program Weighted
    calc_start = time.time()
    best_raw_scores = []
    for t in traces:
        try:
            score = calculate_confidence_best(t.get('logprobs', []))
            best_raw_scores.append(score)
        except:
            best_raw_scores.append(-100.0)
    calc_time = time.time() - calc_start
    
    min_score = min(best_raw_scores)
    best_weights = [s - min_score + 1e-6 for s in best_raw_scores]
    best_voted_answer = weighted_majority_vote(answers, best_weights)
    is_best_correct = (best_voted_answer == correct_answer)
    
    return tc_id, {
        "prob_name": prob_name,
        "is_simple_correct": is_simple_correct,
        "is_best_correct": is_best_correct,
        "calc_time": calc_time
    }

def evaluate_on_dataset(file_path: str):
    print(f"\nEvaluating Best Program on: {os.path.basename(file_path)}")
    print("-" * 100)
    
    # Fast grouping using regex
    print("Grouping lines by test case (regex)...")
    t_group_start = time.time()
    tc_lines = defaultdict(list)
    tc_id_pattern = re.compile(r'"test_case_id":\s*(\d+)')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = tc_id_pattern.search(line)
            if match:
                tc_id = int(match.group(1))
                tc_lines[tc_id].append(line)
    print(f"Grouping took: {time.time() - t_group_start:.2f}s")
    
    results = {}
    print(f"{'TC':<3} | {'Problem':<25} | {'Simple':<6} | {'BestProg':<8} | {'Time(s)':<8}")
    print("-" * 100)
    
    total_calc_time = 0
    start_eval = time.time()
    
    # Sequential processing for better stability and lower memory overhead
    for tid in sorted(tc_lines.keys()):
        tc_start = time.time()
        _, res = process_single_test_case(tid, tc_lines[tid])
        if res:
            results[tid] = res
            print(f"{tid:<3} | {res['prob_name'][:25]:<25} | {str(res['is_simple_correct']):<6} | {str(res['is_best_correct']):<8} | {time.time() - tc_start:.2f}")
            total_calc_time += res['calc_time']
                
    end_eval = time.time()
    print("-" * 100)
    
    total_test_cases = len(results)
    if total_test_cases > 0:
        simple_majority_correct = sum(1 for r in results.values() if r['is_simple_correct'])
        best_weighted_correct = sum(1 for r in results.values() if r['is_best_correct'])
        
        print(f"Total Evaluation Time: {end_eval - start_eval:.2f}s")
        print(f"Total Confidence Calculation Time: {total_calc_time:.2f}s")
        print(f"Simple Majority Accuracy:   {simple_majority_correct / total_test_cases:.4f} ({simple_majority_correct}/{total_test_cases})")
        print(f"Best Program Weighted Acc:  {best_weighted_correct / total_test_cases:.4f} ({best_weighted_correct}/{total_test_cases})")
    else:
        print("No test cases found or processed.")

if __name__ == "__main__":
    data_dir = '/Users/wangyan/Desktop/correct-indicator/data/20260111_043209_traces_Qwen3-8B-AWQ_kv_auto_t0.6_p0.95_max32000_n64_swap4'
    val_random_path = os.path.join(data_dir, 'val_random.jsonl')
    
    if os.path.exists(val_random_path):
        evaluate_on_dataset(val_random_path)
    else:
        print(f"File not found: {val_random_path}")
