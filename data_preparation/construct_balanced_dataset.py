import json
import os
import random
from collections import defaultdict

AIME_DIR = "/Users/wangyan/Desktop/correct-indicator/data/aime2025"
HMMT_DIR = "/Users/wangyan/Desktop/correct-indicator/data/hmmt25_11_18"
OUTPUT_DIR = "/Users/wangyan/Desktop/correct-indicator/data/merged_split"
PROBLEMS_DIR = os.path.join(OUTPUT_DIR, "problems")

def construct_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROBLEMS_DIR, exist_ok=True)
    
    # 1. Collect all problems
    problem_pools = []
    
    # AIME labeled files
    aime_files = [f for f in os.listdir(AIME_DIR) if f.endswith('_labeled.jsonl')]
    for f in aime_files:
        path = os.path.join(AIME_DIR, f)
        problem_pools.append({"path": path, "source": "AIME-2025"})
        
    # HMMT labeled files
    hmmt_files = [f for f in os.listdir(HMMT_DIR) if f.endswith('_labeled.jsonl')]
    for f in hmmt_files:
        path = os.path.join(HMMT_DIR, f)
        problem_pools.append({"path": path, "source": "HMMT-2025"})

    selected_problems_data = []
    trace_counter = 0
    
    metadata = {}

    for prob_info in problem_pools:
        path = prob_info["path"]
        source = prob_info["source"]
        
        filename = os.path.basename(path)
        prob_id = filename.replace('_labeled.jsonl', '')
        
        correct_pool = []
        incorrect_pool = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'is_correct' not in data: continue
                    
                    trace_id = f"trace_{trace_counter:05d}"
                    trace_counter += 1
                    
                    compact_data = {
                        "unique_id": trace_id,
                        "problem_id": prob_id,
                        "source": source,
                        "is_correct": data.get("is_correct"),
                        "extracted_answer": data.get("extracted_answer"),
                        "logprobs": data.get("logprobs")
                    }
                    
                    if compact_data['is_correct']:
                        correct_pool.append(compact_data)
                    elif compact_data.get('extracted_answer'):
                        incorrect_pool.append(compact_data)
                except: continue

        # Apply Threshold: ignore if < 4 correct or < 6 incorrect
        if len(correct_pool) >= 4 and len(incorrect_pool) >= 6:
            # Sample exactly 10 traces: 4 Correct, 6 Incorrect
            sampled_correct = random.sample(correct_pool, 4)
            sampled_incorrect = random.sample(incorrect_pool, 6)
            sampled_traces = sampled_correct + sampled_incorrect
            random.shuffle(sampled_traces)
            
            # Generate 10 test cases for this problem (20 traces each, 4:6 ratio)
            c_ids = [t['unique_id'] for t in sampled_correct]
            i_ids = [t['unique_id'] for t in sampled_incorrect]
            
            problem_cases = []
            for i in range(10):
                c_samples = random.choices(c_ids, k=8)
                i_samples = random.choices(i_ids, k=12)
                combined = c_samples + i_samples
                random.shuffle(combined)
                problem_cases.append({
                    "case_id": i,
                    "trace_ids": combined
                })

            # Save to separate JSON file (one per problem)
            prob_data = {
                "problem_id": prob_id,
                "source": source,
                "traces": sampled_traces,
                "cases": problem_cases
            }
            
            prob_file_rel = f"problems/{prob_id}.json"
            prob_file_path = os.path.join(OUTPUT_DIR, prob_file_rel)
            with open(prob_file_path, 'w', encoding='utf-8') as f_prob:
                json.dump(prob_data, f_prob, indent=4, ensure_ascii=False)
            
            selected_problems_data.append({
                "problem_id": prob_id,
                "file_rel": prob_file_rel,
                "traces": sampled_traces,
                "cases": problem_cases
            })
            
            metadata[prob_id] = {
                "source": source,
                "num_correct": len(correct_pool),
                "num_incorrect": len(incorrect_pool),
                "sampled_file": prob_file_rel
            }
            print(f"Selected: {prob_id} | Saved traces and 10 cases to {prob_file_rel}")
        else:
            print(f"Skipped: {prob_id} | C: {len(correct_pool)} | I: {len(incorrect_pool)}")

    # 2. Write manifests
    TEST_PROBLEM_IDS = [
        "12_let_be_a_convex_pentagon_12",
        "22_0_from_an_unlimited_supply_of_22",
        "23_0_there_are_values_of_in_23",
        "7_0_let_be_real_numbers_such_7"
    ]

    with open(os.path.join(OUTPUT_DIR, "train_split.jsonl"), "w") as f_train, \
         open(os.path.join(OUTPUT_DIR, "val_split.jsonl"), "w") as f_val:
        for prob in selected_problems_data:
            if prob["problem_id"] in TEST_PROBLEM_IDS:
                f_val.write(prob["file_rel"] + "\n")
            else:
                f_train.write(prob["file_rel"] + "\n")

    # 3. Aggregate all test cases for global evaluation if needed
    all_train_cases = []
    all_val_cases = []
    case_counter = 0
    for prob in selected_problems_data:
        for case in prob["cases"]:
            full_case = {
                "case_id": case_counter,
                "problem_id": prob["problem_id"],
                "trace_ids": case["trace_ids"]
            }
            if prob["problem_id"] in TEST_PROBLEM_IDS:
                all_val_cases.append(full_case)
            else:
                all_train_cases.append(full_case)
            case_counter += 1

    with open(os.path.join(OUTPUT_DIR, "train_test_cases.json"), "w") as f:
        json.dump(all_train_cases, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "val_test_cases.json"), "w") as f:
        json.dump(all_val_cases, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, "test_cases.json"), "w") as f:
        json.dump(all_train_cases, f, indent=4)

    # 4. Aggregated traces for reference
    with open(os.path.join(OUTPUT_DIR, "all_traces.jsonl"), "w") as f_all:
        for prob in selected_problems_data:
            for t in prob["traces"]:
                f_all.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nDone! Selected {len(selected_problems_data)} problems.")
    print(f"Total cases: {case_counter} (Train: {len(all_train_cases)}, Val: {len(all_val_cases)})")
    print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    construct_dataset()
