"""
Pipeline stage 1: sample N answers per question from one model at one precision.

Every raw response is stored next to its extracted answer, so runs can be re-scored later.
Output: results/<dataset>/generator/<model>/<model>_<precision>_<dataset>_n<N>.jsonl
(first line is the run configuration, then one record per question). Interrupted runs resume.
"""
import argparse
import datetime
import hashlib
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
from transformers import set_seed as hf_set_seed

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.benchmarks import load_benchmark
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer, model_device
from src.experiment_d.prompts import build_prompt, get_messages
from src.experiment_d.run_files import generator_run_path
from src.experiment_d.sampling import sample_responses
from src.experiment_d.self_consistency import summarize_votes

# Settings that MUST match for a resumed run to be valid. Operational settings like batch_size
# are hardware-dependent and allowed to change.
SCIENTIFIC_SETTINGS = [
    "model", "precision", "dataset", "n_samples",
    "temperature", "top_k", "top_p", "seed",
    "repetition_penalty", "dataset_fingerprint", "max_new_tokens", "prompt_sha256",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sample N answers per question (Experiment D, stage 1)")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="16bit")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name (e.g. gsm8k, cqa, math500 or any huggingface dataset); comma-separated for several, 'both' for gsm8k+cqa")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size for generation (lower = less VRAM)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (e.g. 1.1 for Qwen to stop loops)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions per dataset (0 for all)")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens (default 2048, adjust if needed)")

    args = parser.parse_args()
    if args.n_samples < 1 or args.batch_size < 1:
        parser.error("--n_samples and --batch_size must both be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("--temperature must be positive, --top_p in (0, 1], and --top_k non-negative")
    return args


def question_seed(base_seed: int, qid: str) -> int:
    """Derive a deterministic per-question seed so resumed runs
    produce the same samples for any given question regardless of
    how many questions were processed before it."""
    h = hashlib.md5(qid.encode()).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**31)


def seed_everything(seed: int):
    hf_set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_fingerprint(questions: list) -> str:
    """Hash of the exact questions and answers a run covers."""
    return hashlib.sha256(json.dumps(
        [(item['qid'], item['question'], item['ground_truth']) for item in questions],
        ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def read_existing_run(output_file):
    """(run_config header or None, set of question ids already written) for a possibly partial run file."""
    processed_ids = set()
    existing_config = None
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if record.get("type") == "run_config":
                        existing_config = record
                        continue
                    if "qid" in record:
                        processed_ids.add(record['qid'])
                except Exception:
                    pass
    return existing_config, processed_ids


def build_run_config(args, dataset_name: str, questions: list, model, tokenizer) -> dict:
    """The header line of a run file: everything needed to audit that two runs are comparable."""
    probe_choices = [{"label": "A", "text": "x"}] if dataset_name.lower() == "cqa" else None
    return {
        "type": "run_config", "model": args.model, "precision": args.precision,
        "dataset": dataset_name, "n_samples": args.n_samples,
        "batch_size": args.batch_size, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,

        "total_questions": len(questions),
        "dataset_fingerprint": dataset_fingerprint(questions),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "prompt_sha256": hashlib.sha256(build_prompt(
            tokenizer, get_messages(dataset_name, "PROBE QUESTION", probe_choices)
        ).encode("utf-8")).hexdigest(),
        "model_dtype": str(getattr(model, "dtype", None)),
        "device": str(model_device(model)),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
    }


def check_resume_is_valid(output_file, existing_config, processed_ids, run_config, batch_size):
    """Refuse to append to a run file whose scientific settings differ from this invocation."""
    if processed_ids and existing_config is None:
        raise RuntimeError(f"Cannot safely resume {output_file}: existing records have no run_config header. Move the file or start a fresh output directory.")
    if existing_config is not None:
        mismatches = [key for key in SCIENTIFIC_SETTINGS
                      if key in run_config and key in existing_config
                      and existing_config.get(key) != run_config.get(key)]
        if mismatches:
            raise RuntimeError(f"Cannot resume {output_file}: run settings differ for {', '.join(mismatches)}. Use a fresh output directory.")
        if existing_config.get("batch_size") != batch_size:
            print(f"Notice: Resuming with adjusted batch_size ({existing_config.get('batch_size')} -> {batch_size}).")


def sample_with_oom_backoff(model, tokenizer, prompts, args, dataset_name, first_qid):
    """
    Samples all prompts of one chunk; on CUDA out-of-memory halves the mini-batch and retries.
    Returns (grouped_responses, tokens_per_sec, latency, grouped_lengths, batch size that worked).
    """
    current_batch_size = args.batch_size
    while True:
        try:
            grouped_responses, tps, latency, grouped_lengths = sample_responses(
                model, tokenizer, prompts,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                batch_size=current_batch_size,
                repetition_penalty=args.repetition_penalty,
                stop_on_boxed=dataset_name.lower() == "math500",
            )
            return grouped_responses, tps, latency, grouped_lengths, current_batch_size
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise  # Re-raise non-OOM errors
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise RuntimeError(f"CUDA OOM at batch_size=1 for {first_qid}; stopping so the run cannot appear complete with missing questions.") from e
            current_batch_size = max(1, current_batch_size // 2)
            print(f"\n⚠️  OOM on chunk starting at {first_qid}. "
                  f"Reducing batch_size to {current_batch_size} and retrying...")


def build_question_record(item, dataset_name, responses, gen_lengths, extractor, max_new_tokens, performance):
    """One output line: the question, its vote summary and every raw sample."""
    extracted = [extractor(resp) for resp in responses]
    cutoffs = [l >= max_new_tokens for l in gen_lengths]
    # Truncated samples are partial scratchpads: they must not vote.
    metrics = summarize_votes(extracted, item['ground_truth'],
                              cutoffs=cutoffs, drop_truncated=True)
    metrics["cutoff_rate"] = sum(cutoffs) / len(cutoffs) if len(cutoffs) > 0 else 0.0

    return {
        "qid": item['qid'],
        "dataset": dataset_name,
        "question": item['question'],
        "ground_truth": item['ground_truth'],
        "metrics": metrics,
        "performance": performance,
        "raw_samples": [
            {
                "response": r,
                "extracted": e,
                "length": l,
                "cutoff": c
            }
            for r, e, l, c in zip(responses, extracted, gen_lengths, cutoffs)
        ],
    }


def run_dataset(args, dataset_name: str, questions: list, model, tokenizer):
    if args.limit > 0:
        questions = questions[:args.limit]

    output_file = str(generator_run_path(args.results_dir, args.model, args.precision, dataset_name, args.n_samples))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    existing_config, processed_ids = read_existing_run(output_file)
    run_config = build_run_config(args, dataset_name, questions, model, tokenizer)
    check_resume_is_valid(output_file, existing_config, processed_ids, run_config, args.batch_size)

    print(f"Starting {dataset_name}. Found {len(processed_ids)} already processed items.")
    extractor = get_extractor(dataset_name)

    with open(output_file, 'a') as f:
        # Only write config if file doesn't already have one
        if existing_config is None:
            f.write(json.dumps({**run_config, "timestamp": datetime.datetime.now().isoformat()}) + "\n")
            f.flush()

        # Preserve fixed chunk boundaries across resumes while filling the
        # requested sequence batch. Partially completed chunks are regenerated
        # with the same seed; only missing question records are appended.
        questions_per_chunk = max(1, args.batch_size // args.n_samples)

        for i in tqdm(range(0, len(questions), questions_per_chunk), desc=f"Processing {dataset_name}"):
            chunk = questions[i:i + questions_per_chunk]
            if all(item['qid'] in processed_ids for item in chunk):
                continue

            # Set per-chunk seed for reproducibility across resumes
            seed_everything(question_seed(args.seed, chunk[0]['qid']))

            prompts = [
                build_prompt(tokenizer, get_messages(dataset_name, item['question'], item.get('choices')))
                for item in chunk
            ]
            grouped_responses, tps, latency, grouped_lengths, used_batch_size = sample_with_oom_backoff(
                model, tokenizer, prompts, args, dataset_name, chunk[0]['qid']
            )

            for q_idx, item in enumerate(chunk):
                if item['qid'] in processed_ids:
                    continue
                performance = {
                    "tokens_per_sec": tps,
                    "latency_sec": latency / len(chunk),
                    "effective_batch_size": used_batch_size,
                }
                record = build_question_record(
                    item, dataset_name, grouped_responses[q_idx], grouped_lengths[q_idx],
                    extractor, args.max_new_tokens, performance,
                )
                f.write(json.dumps(record) + "\n")
                processed_ids.add(item['qid'])

            f.flush()  # Ensure it writes to disk immediately after each chunk


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    dataset_names = ["gsm8k", "cqa"] if args.dataset == "both" else [name.strip() for name in args.dataset.split(",")]
    datasets_to_run = [(name, load_benchmark(name)) for name in dataset_names]

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)

    for dataset_name, questions in datasets_to_run:
        run_dataset(args, dataset_name, questions, model, tokenizer)


if __name__ == "__main__":
    main()
