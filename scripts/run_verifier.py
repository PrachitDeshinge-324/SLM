"""
Pipeline stage 3: ask the model to judge each stored solution as Correct or Incorrect.

Reads  results/<dataset>/verifier_dataset/<model>/verifier_<run name>.jsonl
Writes results/<dataset>/verifier_output/<model>/verifier_output_<run name>.jsonl
(the input record plus the verifier's raw response and parsed verdict). Interrupted runs resume.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import torch

from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.prompts import build_prompt
from src.experiment_d.run_files import verifier_dataset_path, verifier_output_path
from src.experiment_d.sampling import count_generated_tokens
from src.experiment_d.verifier import get_verifier_messages, parse_verdict


def parse_args():
    parser = argparse.ArgumentParser(description="Judge stored solutions with the model itself (Experiment D, stage 3)")
    parser.add_argument("--results_dir", "--directory", "--output_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, required=True, choices=PRECISIONS)
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. gsm8k, cqa)")
    parser.add_argument("--n_samples", type=str, required=True, help="Samples per question of the generator run (8, or n8)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of traces for testing")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    return parser.parse_args()


def already_judged(output_file) -> set:
    """(question, trace_index) keys present in a possibly partial output file."""
    keys = set()
    if os.path.exists(output_file):
        print(f"Found existing output file: {output_file}. Scanning for already evaluated traces...")
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        rec = json.loads(line_str)
                        q = rec.get("question")
                        t_idx = rec.get("trace_index")
                        if q is not None and t_idx is not None:
                            keys.add((q, t_idx))
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning reading output file: {e}")
    return keys


def judge_batch(model, tokenizer, traces: list, max_new_tokens: int, repetition_penalty: float):
    """Greedy verdicts for a batch of traces; fills the verifier fields of each trace in place."""
    prompts = [
        build_prompt(tokenizer, get_verifier_messages(trace['question'], trace['student_solution']))
        for trace in traces
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Extract generated tokens (ignoring the prompt)
    generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
    responses = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    for idx, trace in enumerate(traces):
        valid_len = count_generated_tokens(generated_tokens[idx], tokenizer)
        trace['verifier_raw_response'] = responses[idx]
        trace['verifier_final_conclusion'] = parse_verdict(responses[idx])
        # Datasets built before the field was renamed store the trace's truncation as 'cutoff'.
        if 'cutoff' in trace:
            trace['trace_cutoff'] = bool(trace.pop('cutoff'))
        trace['verifier_length'] = valid_len
        trace['verifier_cutoff'] = valid_len >= max_new_tokens


def print_summary(output_file):
    """Confusion matrix over every record of the output file (resumed and new)."""
    with open(output_file, 'r') as f:
        results = [json.loads(line) for line in f]

    tp = fp = tn = fn = 0
    for item in results:
        verdict = item.get('verifier_final_conclusion')
        is_actually_correct = item['is_correct']
        if verdict is not None:
            if verdict and is_actually_correct:
                tp += 1
            elif verdict and not is_actually_correct:
                fp += 1
            elif not verdict and not is_actually_correct:
                tn += 1
            elif not verdict and is_actually_correct:
                fn += 1

    print("\n" + "="*40)
    print("=== VERIFIER BENCHMARK RESULTS ===")
    print("="*40)

    parsed = tp + fp + tn + fn
    print(f"Final-conclusion parse coverage: {parsed}/{len(results)} ({(parsed / len(results) * 100) if results else 0:.2f}%)")
    if parsed > 0:
        accuracy = (tp + tn) / parsed
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr  # Recall is mathematically identical to TPR
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall (TPR): {recall*100:.2f}%")
        print(f"F1 Score: {f1*100:.2f}%")
        print(f"False Positive Rate (FPR): {fpr*100:.2f}%")

        print("\n--- 2x2 Confusion Matrix ---")
        print(f"                 | Actual Correct | Actual Incorrect |")
        print(f"-----------------|----------------|------------------|")
        print(f" Model Correct   | TP: {tp:<10} | FP: {fp:<14} |")
        print(f" Model Incorrect | FN: {fn:<10} | TN: {tn:<14} |")
        print(f"------------------------------------------------------")
    else:
        print("ORM metrics: Could not parse final conclusions.")


def main():
    args = parse_args()

    input_file = verifier_dataset_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    if not input_file.exists():
        print(f"Error: verifier dataset not found: {input_file}")
        sys.exit(1)
    output_file = verifier_output_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    os.makedirs(output_file.parent, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    # Essential for batched autoregressive generation
    tokenizer.padding_side = "left"

    print(f"Loading input file: {input_file}")
    with open(input_file, 'r') as f:
        traces = [json.loads(line) for line in f]
    if args.limit is not None:
        traces = traces[:args.limit]

    # --- Robust Key-Based Resume Functionality ---
    judged = already_judged(output_file)
    if judged:
        initial_count = len(traces)
        traces = [trace for trace in traces if (trace.get("question"), trace.get("trace_index")) not in judged]
        print(f"Resuming: {initial_count - len(traces)} traces already evaluated. Remaining to evaluate: {len(traces)}")
        if len(traces) == 0:
            print(f"All {initial_count} traces have already been evaluated. Exiting.")
            return
    else:
        # If starting completely fresh, clear the output file
        open(output_file, 'w').close()

    print(f"Evaluating {len(traces)} traces in batches of {args.batch_size}...")

    current_batch_size = args.batch_size
    pbar = tqdm(total=len(traces))
    i = 0
    while i < len(traces):
        batch = traces[i : i + current_batch_size]
        try:
            judge_batch(model, tokenizer, batch, args.max_new_tokens, args.repetition_penalty)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise RuntimeError(f"CUDA OOM at batch_size=1 at index {i}; stopping.") from e
            current_batch_size = max(1, current_batch_size // 2)
            print(f"\n⚠️  OOM at index {i}. Reducing batch_size to {current_batch_size} and retrying...")
            continue

        # Incremental save (Append only the new batch)
        with open(output_file, 'a') as f:
            for trace in batch:
                f.write(json.dumps(trace) + '\n')

        i += len(batch)
        pbar.update(len(batch))
    pbar.close()

    print_summary(output_file)


if __name__ == "__main__":
    main()
