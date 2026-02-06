#!/usr/bin/env python3
"""
Prepare trace data with logprobs/logits for AlphaConf evolution

This script runs vLLM on a dataset (e.g., GSM8K or AIME) and saves trace data
with top-k tokens' logprobs/logits and correctness labels in JSONL format.

Output format (compact):
{
    "trace": "...", 
    "logprobs": [
        [selected_logprob, [logprob1, logprob2, logprob3, ...]],
        ...
    ],
    "is_correct": true
}

Each position is a list: [selected_logprob, logprobs_list]
- selected_logprob: float (the logprob of the selected/generated token, 5 significant digits)
- logprobs_list: List of all top-k logprobs (sorted by logprob, highest first, 5 significant digits)
- All floats formatted to 5 significant digits

Copyright (c) Meta Platforms, Inc. and affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import argparse
import os
import sys
import re
import hashlib
import time
import random
import copy
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dynasor.core.evaluator import math_equal


# ============= UTILITY FUNCTIONS =============

def quick_parse(text: str) -> str:
    """Parse LaTeX text content"""
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text


def equal_func(answer: str, ground_truth: str) -> bool:
    """Check if answer equals ground truth"""
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)


def extract_answer(text: str) -> Optional[str]:
    """Extract boxed answer from text"""
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


def prepare_prompt(question: str, tokenizer, model_type: str = "deepseek") -> str:
    """Prepare prompt for a single question"""
    if model_type == "deepseek":
        messages = [
            {"role": "system", "content": "You are a math problem solver. You are given a math problem and you need to solve it. Note you should provide the final answer in the format of a separate line of \"Final Answer: \\boxed{...}\" \n"},
            {"role": "user", "content": question}
        ]
    else:
        messages = [
            {"role": "user", "content": question}
        ]
    
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return full_prompt


def generate_question_short_name(question: str, index: int) -> str:
    """Generate a short name for a question based on its content and index"""
    # Remove LaTeX and special characters, take first few words
    clean_question = re.sub(r'\$.*?\$', '', question)  # Remove LaTeX math
    clean_question = re.sub(r'[^\w\s]', '', clean_question)  # Remove special chars
    words = clean_question.split()[:5]  # Take first 5 words
    short_name = '_'.join(words).lower()[:50]  # Limit length
    
    # Add index for uniqueness
    short_name = f"{short_name}_{index}"
    
    # Remove any remaining invalid filename characters
    short_name = re.sub(r'[^\w\-_]', '_', short_name)
    
    return short_name


def format_significant_digits(value: float, sig_digits: int = 5) -> float:
    """Format float to specified significant digits"""
    if value == 0.0:
        return 0.0
    # Use g format to get significant digits, then convert back to float
    formatted = f"{value:.{sig_digits}g}"
    return float(formatted)


def extract_topk_tokens_from_vllm_output(
    logprobs: Optional[List[Optional[Dict]]],
    token_ids: Optional[List[int]] = None,
    check_logits: bool = True
) -> List[List[Any]]:
    """Extract top-k tokens' logprobs from vLLM output (compact format)
    
    vLLM returns logprobs as a list where each element is a dict:
    {token_id: LogprobInfo(logprob=float, ...), ...}
    
    When logprobs=N is set, vLLM returns top-N tokens' logprobs per position.
    We extract ALL top-k tokens' logprobs and the selected token's logprob.
    
    Args:
        logprobs: List of dicts, one per token position
        token_ids: List of selected token IDs (one per position) - used to find selected logprob
        check_logits: Not used (kept for compatibility)
    
    Returns:
        List of lists, one per token position. Each list contains:
        [selected_logprob, [logprob1, logprob2, logprob3, ...]]
        - selected_logprob: float (the logprob of the selected token)
        - logprobs_list: List of all top-k logprobs (sorted by logprob, highest first)
        - All floats formatted to 5 significant digits
    """
    if not logprobs:
        return []
    
    position_data = []
    for pos_idx, token_logprobs in enumerate(logprobs):
        if token_logprobs:
            # Get selected token ID for this position
            selected_token_id = None
            selected_logprob = 0.0
            if token_ids and pos_idx < len(token_ids):
                selected_token_id = int(token_ids[pos_idx])
            
            logprobs_list = []
            for token_id, logprob_obj in token_logprobs.items():
                token_id_int = int(token_id)
                
                # Extract logprob - verify we're getting the correct attribute
                logprob_value = 0.0
                if hasattr(logprob_obj, 'logprob'):
                    # Direct attribute access
                    logprob_val = getattr(logprob_obj, 'logprob')
                    if isinstance(logprob_val, (int, float)):
                        logprob_value = float(logprob_val)
                elif isinstance(logprob_obj, (int, float)):
                    # Sometimes logprob_obj itself is the value
                    logprob_value = float(logprob_obj)
                else:
                    # Fallback: check if it's a dict or has other structure
                    if isinstance(logprob_obj, dict):
                        if 'logprob' in logprob_obj:
                            logprob_value = float(logprob_obj['logprob'])
                
                # Format to 5 significant digits
                logprob_value = format_significant_digits(logprob_value, 5)
                
                # Store selected token's logprob
                if selected_token_id is not None and token_id_int == selected_token_id:
                    selected_logprob = logprob_value
                
                logprobs_list.append(logprob_value)
            
            # Sort by logprob (highest first)
            logprobs_list.sort(reverse=True)
            # Compact format: [selected_logprob, [logprob1, logprob2, ...]]
            position_data.append([selected_logprob, logprobs_list])
        else:
            # No logprobs for this position
            selected_logprob = 0.0
            if token_ids and pos_idx < len(token_ids):
                # Try to find selected token's logprob even if not in top-k
                # But if no logprobs, we can't get it, so use 0.0
                pass
            position_data.append([selected_logprob, []])
    
    return position_data


# ============= MAIN PROCESSING =============

def process_dataset(
    dataset_path: str,
    output_dir: str,
    model: str,
    model_type: str = "deepseek",
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 32000,
    indices: Optional[str] = None,
    logprobs: int = 20,
    num_samples: Optional[int] = None,
    start_idx: int = 0,
    traces_per_question: int = 1024,
    batch_size: int = 64,
    metadata: Optional[Dict[str, Any]] = None,
    shuffle: bool = True,
    stop: Optional[List[str]] = None,
    **vllm_kwargs
):
    """
    Process dataset and save traces with logprobs (multiple traces per question)
    
    Args:
        dataset_path: Path to input dataset (JSONL format with 'question' and 'answer' fields)
        output_dir: Directory to save output JSONL files (one per question)
        model: Model name or path
        model_type: Model type ("deepseek", "gpt", etc.)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
        logprobs: Number of logprobs to return
        num_samples: Number of samples to process (None = all)
        start_idx: Starting index in dataset
        traces_per_question: Number of traces to generate per question
        batch_size: Number of prompts per request (for batching large trace counts)
        **vllm_kwargs: Additional vLLM arguments
    """
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    # Use internal 'idx' if available (e.g., from prepare_hmmt.py), 
                    # fallback to 0-indexed line number
                    idx = item.get('idx', i)
                    dataset.append((idx, item))
                except json.JSONDecodeError:
                    continue
    
    # Filter by indices if specified
    if indices and indices.strip():
        try:
            target_indices = [int(idx.strip()) for idx in indices.split(',') if idx.strip()]
            dataset = [item for item in dataset if item[0] in target_indices]
            print(f"Filtered dataset to {len(dataset)} items based on indices: {target_indices}")
        except ValueError as e:
            print(f"Error parsing indices '{indices}': {e}. Ignoring filter.")
    
    total_items = len(dataset)
    if num_samples is None:
        num_samples = max(0, total_items - start_idx)
    else:
        num_samples = max(0, min(num_samples, total_items - start_idx))
    
    end_idx = start_idx + num_samples
    selected_items = dataset[start_idx:end_idx]
    num_selected = len(selected_items)
    print(f"Processing {num_selected} items (indices {start_idx} to {start_idx + num_selected - 1} in dataset list)...")
    
    # Initialize model
    print(f"Initializing vLLM with model {model}...")
    default_kwargs = {
        "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        "enable_prefix_caching": True,
        "trust_remote_code": True,
    }
    default_kwargs.update(vllm_kwargs)
    llm = LLM(model=model, **default_kwargs)
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save metadata
    if metadata:
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        print(f"Metadata saved to {metadata_path}")
    
    # Prepare all individual trace requests
    print("\nPreparing all trace requests...")
    all_trace_requests = []
    base_seed = int(time.time_ns())
    
    # Track metadata for each question
    question_info = {}
    
    for i, (original_idx, item) in enumerate(selected_items):
        question = item.get('question', '')
        ground_truth = str(item.get('answer', '')).strip()
        
        if not question:
            continue
        
        question_short_name = generate_question_short_name(question, original_idx)
        prompt = prepare_prompt(question, tokenizer, model_type)
        
        question_info[original_idx] = {
            'ground_truth': ground_truth,
            'short_name': question_short_name,
            'output_file': os.path.join(output_dir, f"{original_idx}_{question_short_name}.jsonl")
        }
        
        # Clear existing output files if they exist to avoid mixing with old runs
        target_file = question_info[original_idx]['output_file']
        if os.path.exists(target_file):
            os.remove(target_file)

        for trace_idx in range(traces_per_question):
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=logprobs,
                seed=base_seed + trace_idx + original_idx * 1000000,
                stop=stop,
            )
            all_trace_requests.append({
                'original_idx': original_idx,
                'prompt': prompt,
                'sampling_params': sampling_params,
            })

    # Shuffle all individual requests together for maximum GPU throughput
    if shuffle:
        print(f"Shuffling {len(all_trace_requests)} total trace requests...")
        random.shuffle(all_trace_requests)
    else:
        print(f"Executing {len(all_trace_requests)} requests in sequential order...")
    
    # Create batches of size batch_size
    batches = []
    for i in range(0, len(all_trace_requests), batch_size):
        batches.append(all_trace_requests[i:i + batch_size])
    
    # Statistics tracking
    question_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'no_answer': 0, 'total': 0})
    total_stats = {'correct': 0, 'incorrect': 0, 'no_answer': 0, 'total': 0}
    
    print(f"\nStarting generation of {len(batches)} global batches (mixed questions)...")
    
    for batch_idx, batch_reqs in enumerate(batches):
        prompts = [r['prompt'] for r in batch_reqs]
        params = [r['sampling_params'] for r in batch_reqs]
        
        # Run generation for this global batch
        start_time = time.time()
        outputs = llm.generate(prompts, params)
        end_time = time.time()
        
        total_tokens = 0
        
        # Group results by question to batch writes
        results_by_question = defaultdict(list)
        batch_stats = {'correct': 0, 'incorrect': 0, 'no_answer': 0, 'total': 0}
        
        for i, (req, output) in enumerate(zip(batch_reqs, outputs)):
            original_idx = req['original_idx']
            ground_truth = question_info[original_idx]['ground_truth']
            
            # Quicker per-request progress within the batch
            if (i + 1) % 10 == 0 or (i + 1) == len(batch_reqs):
                print(f"  [Batch {batch_idx+1}] Processing result {i+1}/{len(batch_reqs)}...", end='\r')

            for single_output in output.outputs:
                trace = single_output.text
                total_tokens += len(single_output.token_ids)
                topk_tokens = extract_topk_tokens_from_vllm_output(
                    single_output.logprobs,
                    single_output.token_ids,
                    check_logits=True
                )
                
                extracted_answer = extract_answer(trace)
                is_correct = False
                
                if extracted_answer is None:
                    question_stats[original_idx]['no_answer'] += 1
                    total_stats['no_answer'] += 1
                    batch_stats['no_answer'] += 1
                else:
                    if ground_truth:
                        try:
                            is_correct = equal_func(extracted_answer, ground_truth)
                        except Exception:
                            is_correct = False
                    
                    if is_correct:
                        question_stats[original_idx]['correct'] += 1
                        total_stats['correct'] += 1
                        batch_stats['correct'] += 1
                    else:
                        question_stats[original_idx]['incorrect'] += 1
                        total_stats['incorrect'] += 1
                        batch_stats['incorrect'] += 1
                
                question_stats[original_idx]['total'] += 1
                total_stats['total'] += 1
                batch_stats['total'] += 1
                
                trace_data = {
                    "trace": trace,
                    "logprobs": topk_tokens,
                    "is_correct": is_correct,
                }
                results_by_question[original_idx].append(trace_data)
        
        print() # Clear the in-place processing progress line
        
        # Append results to respective question files
        for q_idx, traces in results_by_question.items():
            output_file = question_info[q_idx]['output_file']
            with open(output_file, 'a', encoding='utf-8') as f:
                for t in traces:
                    f.write(json.dumps(t, ensure_ascii=False) + '\n')
        
        # Print statistics
        duration = end_time - start_time
        tps = total_tokens / duration if duration > 0 else 0
        print(f"\nProgress: {batch_idx + 1}/{len(batches)} batches finished | Global Batch Size: {len(batch_reqs)}")
        print(f"Current Batch Tokens: {total_tokens} ({tps:.1f} tok/s)")
        
        # Batch Ratios
        b_total = batch_stats['total']
        if b_total > 0:
            c_perc = (batch_stats['correct'] / b_total * 100)
            i_perc = (batch_stats['incorrect'] / b_total * 100)
            n_perc = (batch_stats['no_answer'] / b_total * 100)
            print(f"Batch Performance: Correct: {batch_stats['correct']} ({c_perc:.1f}%) | "
                  f"Incorrect: {batch_stats['incorrect']} ({i_perc:.1f}%) | "
                  f"No Ans: {batch_stats['no_answer']} ({n_perc:.1f}%)")
        
        # Periodic summary table (every batch for frequent progress)
        if (batch_idx + 1) % 1 == 0 or (batch_idx + 1) == len(batches):
            print(f"\n{'Q_Idx':<7} | {'Correct':<8} | {'Incorrect':<10} | {'No Ans':<7} | {'Valid':<6} | {'Ratio':<7} | {'Progress':<8}")
            print("-" * 75)
            # Only show questions that have at least one trace generated
            for q_idx in sorted(question_stats.keys()):
                s = question_stats[q_idx]
                if s['total'] > 0:
                    valid_total = s['correct'] + s['incorrect']
                    # Ratio is Correct / Total (including No Ans)
                    ratio = (s['correct'] / s['total'])
                    # Progress is Total generated / Total requested per question
                    progress = (s['total'] / traces_per_question * 100)
                    print(f"{q_idx:<7} | {s['correct']:<8} | {s['incorrect']:<10} | {s['no_answer']:<7} | {valid_total:<6} | {ratio:<7.3f} | {progress:>6.1f}%")
            print("-" * 75)
            total_valid = total_stats['correct'] + total_stats['incorrect']
            total_ratio = (total_stats['correct'] / total_stats['total']) if total_stats['total'] > 0 else 0
            total_progress = (total_stats['total'] / (len(selected_items) * traces_per_question) * 100) if selected_items else 0
            print(f"{'TOTAL':<7} | {total_stats['correct']:<8} | {total_stats['incorrect']:<10} | {total_stats['no_answer']:<7} | {total_valid:<6} | {total_ratio:<7.3f} | {total_progress:>6.1f}%")
            
            # Progress bar-style line
            progress_pct = (batch_idx + 1) / len(batches) * 100
            bar_len = 20
            filled_len = int(bar_len * (batch_idx + 1) // len(batches))
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            print(f"Overall Progress: [{bar}] {progress_pct:.1f}% ({batch_idx + 1}/{len(batches)} batches)\n")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Total questions processed: {num_selected}")
    print(f"  Total traces generated: {total_stats['total']}")
    print(f"  Total correct traces: {total_stats['correct']} ({(total_stats['correct']/total_stats['total']*100) if total_stats['total'] > 0 else 0:.1f}%)")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


# ============= MAIN =============

def main():
    parser = argparse.ArgumentParser(
        description='Prepare trace data with logprobs for AlphaConf evolution',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('dataset', type=str, help='Input dataset path (JSONL format)')
    parser.add_argument('--model', type=str, default='stelterlab/DeepSeek-R1-0528-Qwen3-8B-AWQ',
                        help='Model name or path')
    parser.add_argument('--model-type', type=str, default='deepseek',
                        choices=['deepseek', 'gpt', 'qwen'],
                        help='Model type for prompt formatting')
    parser.add_argument('--temperature', type=float, default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                        help='Nucleus sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=32000,
                        help='Maximum tokens to generate')
    parser.add_argument('--logprobs', type=int, default=20,
                        help='Number of top-k logprobs to return per token (default: 20, can be up to vocab size)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of samples to process (None = all)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in dataset')
    parser.add_argument('--traces-per-question', type=int, default=1024,
                        help='Number of traces to generate per question')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Number of prompts per request (for batching large trace counts)')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                        help='GPU memory utilization (default: 0.9 for larger models)')
    parser.add_argument('--max-model-len', type=int, default=32768,
                        help='Maximum model length (total sequence: prompt + generation). '
                             'NOTE: This must be >= prompt_size + max_tokens, otherwise generation '
                             'will be truncated. For max_tokens=32000, set this to at least 33000-35000.')
    parser.add_argument('--quantization', type=str, default='bitsandbytes',
                        help='Quantization method (e.g., bitsandbytes)')
    parser.add_argument('--load-format', type=str, default='bitsandbytes',
                        help='Model loading format (e.g., bitsandbytes)')
    parser.add_argument('--kv-cache-dtype', type=str, default='',
                        choices=['', 'auto', 'fp8'],
                        help='Data type for kv cache storage. If "fp8", uses fp8_e5m2 on NVIDIA GPUs.')
    parser.add_argument('--indices', type=str, default='',
                        help='Comma-separated list of problem indices to process')
    parser.add_argument('--no-shuffle', action='store_true',
                        help='Do not shuffle tasks across questions; process sequentially')
    parser.add_argument('--enable-chunked-prefill', action='store_true', 
                    help='Enable chunked prefill to speed up long context generation')
    parser.add_argument('--max-num-batched-tokens', type=int, default=None,
                    help='Maximum number of batched tokens per iteration. If None, vLLM will set a safe default.')
    parser.add_argument('--enforce-eager', action='store_true', 
                    help='Disable CUDA graphs and use eager mode to save time/memory')
    parser.add_argument('--swap-space', type=int, default=4,
                    help='CPU swap space size (GB) to use when GPU KV cache is full')
    parser.add_argument('--max-num-seqs', type=int, default=None,
                    help='Maximum number of sequences per iteration')
    parser.add_argument('--stop', type=str, default=None,
                    help='Stop sequences, comma separated')
    args = parser.parse_args()
    
    stop_sequences = []
    if args.stop:
        stop_sequences = [s.strip() for s in args.stop.split(',')]
    
    vllm_kwargs = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
        'kv_cache_dtype': args.kv_cache_dtype,
#         'quantization': args.quantization,
 #        'load_format': args.load_format,
 'enable_chunked_prefill': args.enable_chunked_prefill,
 'enforce_eager': args.enforce_eager,
 'swap_space': args.swap_space,
    }
    if args.max_num_batched_tokens:
        vllm_kwargs['max_num_batched_tokens'] = args.max_num_batched_tokens
    
    if args.max_num_seqs:
        vllm_kwargs['max_num_seqs'] = args.max_num_seqs
    
    # Construct unique output directory name based on params
    model_name = args.model.split('/')[-1]
    kv_cache = args.kv_cache_dtype if args.kv_cache_dtype else "default"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    dir_parts = [
        timestamp,
        f"traces",
        model_name,
        f"kv_{kv_cache}",
        f"t{args.temperature}",
        f"p{args.top_p}",
        f"max{args.max_tokens}",
        f"n{args.traces_per_question}",
        f"swap{args.swap_space}"
    ]
    if args.num_samples:
        dir_parts.append(f"s{args.num_samples}")
    if args.no_shuffle:
        dir_parts.append("noshuf")
    
    output_dir = "_".join(dir_parts)
    
    # Capture metadata
    metadata = vars(args)
    metadata['vllm_kwargs'] = vllm_kwargs
    metadata['timestamp'] = timestamp
    metadata['command'] = " ".join(sys.argv)

    process_dataset(
        dataset_path=args.dataset,
        output_dir=output_dir,
        model=args.model,
        model_type=args.model_type,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        indices=args.indices,
        logprobs=args.logprobs,
        num_samples=args.num_samples,
        start_idx=args.start_idx,
        traces_per_question=args.traces_per_question,
        batch_size=args.batch_size,
        metadata=metadata,
        shuffle=not args.no_shuffle,
        stop=stop_sequences if stop_sequences else None,
        **vllm_kwargs
    )


if __name__ == "__main__":
    main()

