import json
import os
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

def extract_token_confidences(logprobs: List[List[Any]]) -> List[float]:
    """Compute negative mean logprob per token, matching deepconf baseline."""
    confs = []
    for position_data in logprobs:
        if position_data and len(position_data) >= 2:
            logprobs_list = position_data[1]
            if logprobs_list:
                # Compute mean of all top-k logprobs at this position
                mean_logprob = np.mean(logprobs_list)
                confs.append(-mean_logprob)
    return confs

def calculate_bottom_window_confidence(confs: List[float], window_size: int = 2048, bottom_percent: float = 0.1) -> float:
    """Calculate mean confidence from sliding windows, return average of bottom percentile."""
    if not confs:
        return 0.0
    
    if len(confs) < window_size:
        return float(np.mean(confs))
    
    window_means = []
    current_sum = sum(confs[:window_size])
    window_means.append(current_sum / window_size)
    
    for i in range(1, len(confs) - window_size + 1):
        current_sum = current_sum - confs[i-1] + confs[i + window_size - 1]
        window_means.append(current_sum / window_size)
    
    if not window_means:
        return 0.0
    
    if bottom_percent == -1:  # Min window
        return float(min(window_means))
    
    num_bottom = max(1, int(len(window_means) * bottom_percent))
    if num_bottom == 1:
        return float(min(window_means))
    else:
        # Optimization: use partition instead of full sort
        bottom_means = np.partition(window_means, num_bottom-1)[:num_bottom]
        return float(np.mean(bottom_means))

def weighted_majority_vote(answers: List[str], weights: List[float]) -> Optional[str]:
    """Perform weighted majority voting."""
    if not answers:
        return None
    
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    
    if not answer_weights:
        return None
    
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def evaluate_on_dataset(file_path: str):
    print(f"\nEvaluating Baseline on: {os.path.basename(file_path)}")
    print("-" * 60)
    
    # Load traces and group by test_case_id
    test_cases = defaultdict(list)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            t = json.loads(line)
            test_cases[t['test_case_id']].append(t)
            
    total_test_cases = 0
    simple_majority_correct = 0
    weighted_majority_correct = 0
    
    for tc_id in sorted(test_cases.keys()):
        traces = test_cases[tc_id]
        prob_name = traces[0].get('problem_name', 'unknown')
        
        # Determine ground truth
        correct_answer = None
        for t in traces:
            if t.get('is_correct'):
                correct_answer = str(t.get('extracted_answer'))
                break
        
        if correct_answer is None:
            continue
            
        total_test_cases += 1
        answers = [str(t.get('extracted_answer', '')) for t in traces]
        
        # 1. Simple Majority Vote
        vote_counts = Counter(answers)
        simple_voted_answer = vote_counts.most_common(1)[0][0]
        if simple_voted_answer == correct_answer:
            simple_majority_correct += 1
            
        # 2. Weighted Majority Vote
        confidences = []
        for t in traces:
            token_confs = extract_token_confidences(t.get('logprobs', []))
            conf = calculate_bottom_window_confidence(token_confs, window_size=2048, bottom_percent=0.1)
            confidences.append(conf)
            
        weighted_voted_answer = weighted_majority_vote(answers, confidences)
        if weighted_voted_answer == correct_answer:
            weighted_majority_correct += 1
            
        print(f"TC {tc_id:<2} ({prob_name[:25]:<25}) | Simple: {simple_voted_answer == correct_answer:<5} | Weighted: {weighted_voted_answer == correct_answer}")
        
    print("-" * 60)
    print(f"Total Test Cases: {total_test_cases}")
    if total_test_cases > 0:
        print(f"Simple Majority Accuracy:   {simple_majority_correct / total_test_cases:.4f} ({simple_majority_correct}/{total_test_cases})")
        print(f"Weighted Majority Accuracy: {weighted_majority_correct / total_test_cases:.4f} ({weighted_majority_correct}/{total_test_cases})")

if __name__ == "__main__":
    data_dir = '/Users/wangyan/Desktop/correct-indicator/data/20260111_043209_traces_Qwen3-8B-AWQ_kv_auto_t0.6_p0.95_max32000_n64_swap4'
    val_random_path = os.path.join(data_dir, 'val_random.jsonl')
    
    if os.path.exists(val_random_path):
        evaluate_on_dataset(val_random_path)
    else:
        print(f"File not found: {val_random_path}")
