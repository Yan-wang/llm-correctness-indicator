#!/usr/bin/env python3
"""
Label traces with correct/incorrect based on ground truth answers using extract_answer from deepconf/utils.py

1. Reads correct answers from aime_full.jsonl (line number = problem index)
2. Extracts answers using extract_answer function from deepconf.utils
3. Compares with ground truth to categorize as correct/incorrect
4. Filters traces with no answer into unfinished/ directory
"""

import json
import os
import argparse
import glob
import re
from typing import Optional, List, Dict, Any


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text - same as deepconf/utils.py"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    
    return None


def load_correct_answers(answer_file: str) -> Dict[int, str]:
    """Load correct answers from JSONL file"""
    correct_answers = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    data = json.loads(line)
                    problem_idx = data.get('idx', line_idx)
                    answer = str(data.get('answer', '')).strip()
                    correct_answers[problem_idx] = answer
                except json.JSONDecodeError:
                    continue
    return correct_answers

def categorize_trace(extracted_answer: Optional[str], ground_truth: str) -> Dict[str, Any]:
    """Categorize trace based on extracted answer and ground truth"""
    if extracted_answer is None:
        return {
            "final_answer": None,
            "category": "unfinished",
            "reason": "No answer extracted (no boxed answer found)"
        }
    
    # Normalize answers for comparison (strip whitespace, case-insensitive)
    extracted_normalized = str(extracted_answer).strip()
    ground_truth_normalized = str(ground_truth).strip()
    
    if extracted_normalized == ground_truth_normalized:
        return {
            "final_answer": extracted_normalized,
            "category": "correct",
            "reason": "Extracted answer matches ground truth"
        }
    else:
        return {
            "final_answer": extracted_normalized,
            "category": "incorrect",
            "reason": f"Extracted answer '{extracted_normalized}' does not match ground truth '{ground_truth_normalized}'"
        }

def process_trace(trace: Dict[str, Any], ground_truth: str) -> Dict[str, Any]:
    """Process a single trace: extract answer and categorize"""
    trace_text = trace.get('trace', '')
    extracted = extract_answer(trace_text)
    result = categorize_trace(extracted, ground_truth)
    
    trace['extracted_answer'] = result['final_answer']
    trace['category'] = result['category']
    trace['reason'] = result['reason']
    trace['is_correct'] = (result['category'] == 'correct')
    
    return trace

def process_trace_file(
    input_file: str,
    correct_answers: Dict[int, str],
    output_file: Optional[str] = None,
    problem_idx: Optional[int] = None
) -> dict:
    """Process a trace file using extract_answer for categorization"""
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f'{base_name}_labeled.jsonl'
    
    if problem_idx is None:
        filename = os.path.basename(input_file)
        match = re.search(r'^(\d+)_', filename)
        problem_idx = int(match.group(1)) if match else -1
    
    correct_answer = correct_answers.get(problem_idx, '')
    
    traces = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    labeled_traces = []
    unfinished_traces = []
    stats = {
        'total': len(traces),
        'labeled_correct': 0,
        'labeled_incorrect': 0,
        'labeled_unfinished': 0,
        'problem_idx': problem_idx,
        'correct_answer': correct_answer
    }
    
    print(f"Processing {len(traces)} traces from {os.path.basename(input_file)}...")
    
    for trace in traces:
        processed_trace = process_trace(trace, correct_answer)
        category = processed_trace.get('category', 'unfinished')
        
        if category == 'correct':
            stats['labeled_correct'] += 1
            labeled_traces.append(processed_trace)
        elif category == 'incorrect':
            stats['labeled_incorrect'] += 1
            labeled_traces.append(processed_trace)
        elif category == 'unfinished':
            stats['labeled_unfinished'] += 1
            unfinished_traces.append(processed_trace)
    
    # Write labeled traces (with answers)
    with open(output_file, 'w', encoding='utf-8') as f:
        for trace in labeled_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + '\n')
    
    # Write unfinished traces (no answer) to unfinished/ directory
    if unfinished_traces:
        unfinished_dir = os.path.join(os.path.dirname(input_file), 'unfinished')
        os.makedirs(unfinished_dir, exist_ok=True)
        unfinished_file = os.path.join(unfinished_dir, os.path.basename(output_file))
        with open(unfinished_file, 'w', encoding='utf-8') as f:
            for trace in unfinished_traces:
                f.write(json.dumps(trace, ensure_ascii=False) + '\n')
        print(f"  Wrote {len(unfinished_traces)} unfinished traces to {unfinished_file}")
    
    stats['output_file'] = output_file
    return stats

def process_directory(
    directory: str,
    answer_file: str,
    pattern: str = '*.jsonl',
    re_process: bool = False
):
    """Process all matching files in a directory"""
    correct_answers = load_correct_answers(answer_file)
    search_pattern = os.path.join(directory, pattern)
    files = sorted(glob.glob(search_pattern))
    
    if not re_process:
        files = [f for f in files if '_labeled' not in f]
    
    if not files:
        print(f'No files found matching pattern: {search_pattern}')
        return
    
    total_stats = {'total': 0, 'labeled_correct': 0, 'labeled_incorrect': 0, 'labeled_unfinished': 0}
    
    for input_file in files:
        stats = process_trace_file(input_file, correct_answers)
        print(f"{os.path.basename(input_file)}: Correct: {stats['labeled_correct']}, Incorrect: {stats['labeled_incorrect']}, Unfinished: {stats['labeled_unfinished']}")
        
        for key in total_stats:
            total_stats[key] += stats.get(key, 0)
    
    print('=' * 80)
    print('TOTAL SUMMARY:')
    for key, val in total_stats.items():
        print(f"  {key}: {val}")
    
    # Print stats by default
    print_labeled_statistics(directory)

def print_labeled_statistics(directory: str = 'aime_traces'):
    """Print statistics of is_correct distribution for each problem from labeled files"""
    from collections import defaultdict
    search_pattern = os.path.join(directory, '*_labeled.jsonl')
    labeled_files = sorted(glob.glob(search_pattern))
    
    if not labeled_files:
        return
    
    problem_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'files': 0})
    for file_path in labeled_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        if not (parts and parts[0].isdigit()): continue
        problem_idx = int(parts[0])
        
        c, inc = 0, 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    trace = json.loads(line)
                    if trace.get('is_correct') is True: c += 1
                    elif trace.get('is_correct') is False: inc += 1
        except: continue
        
        if c > 0 or inc > 0:
            problem_stats[problem_idx]['correct'] += c
            problem_stats[problem_idx]['incorrect'] += inc
            problem_stats[problem_idx]['files'] += 1

    sorted_problems = sorted(problem_stats.items())
    print('\n' + '=' * 100)
    print('IS_CORRECT DISTRIBUTION BY PROBLEM')
    print('=' * 100)
    header = 'Problem'.ljust(10) + 'Files'.ljust(10) + 'Total'.ljust(10) + 'Correct'.ljust(12) + 'Incorrect'.ljust(12) + '% Correct'
    print(header)
    print('-' * 100)
    
    for pid, stats in sorted_problems:
        total = stats['correct'] + stats['incorrect']
        pct = (100 * stats['correct'] / total) if total > 0 else 0.0
        print(f"{str(pid).ljust(10)}{str(stats['files']).ljust(10)}{str(total).ljust(10)}{str(stats['correct']).ljust(12)}{str(stats['incorrect']).ljust(12)}{pct:.1f}")

def main():
    parser = argparse.ArgumentParser(description='Label traces with correct/incorrect using extract_answer')
    parser.add_argument('input', type=str, help='Input file or directory')
    parser.add_argument('--answers', type=str, default='aime_full.jsonl', help='Correct answers file')
    parser.add_argument('--re-process', action='store_true', help='Re-process labeled files')
    parser.add_argument('--stats-only', action='store_true', help='Print stats only and exit')
    args = parser.parse_args()
    
    if args.stats_only:
        print_labeled_statistics(args.input if os.path.isdir(args.input) else os.path.dirname(args.input) if args.input else 'aime_traces')
        return
    
    if os.path.isfile(args.input):
        correct_answers = load_correct_answers(args.answers)
        stats = process_trace_file(args.input, correct_answers)
        print(f"Result: {stats}")
        # Print stats for the directory containing the file
        print_labeled_statistics(os.path.dirname(args.input))
    else:
        process_directory(args.input, args.answers, re_process=args.re_process)

if __name__ == "__main__":
    main()
