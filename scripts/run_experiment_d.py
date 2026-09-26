import argparse
import json
import os
import sys
from pathlib import Path
import datetime
import hashlib
from tqdm import tqdm
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

import torch
from transformers import set_seed as hf_set_seed

from src.experiment_d.data_loader import load_cqa_dataset, load_gsm8k_dataset
from src.experiment_d.prompts import get_gsm8k_messages, get_cqa_messages, build_prompt
from src.experiment_d.inference import load_model_and_tokenizer, generate_n_samples, generate_batch_prompts
from src.experiment_d.extractors import extract_gsm8k_answer, extract_cqa_answer
from src.experiment_d.metrics import compute_metrics


def _qid_seed(base_seed: int, qid: str) -> int:
    """Derive a deterministic per-question seed so resumed runs
    produce the same samples for any given question regardless of
    how many questions were processed before it."""
    h = hashlib.md5(qid.encode()).hexdigest()
    return (base_seed + int(h[:8], 16)) % (2**31)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment D (Quantization Self-Improvement)")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model ID (e.g. Qwen/Qwen3.5-0.8B)")
    parser.add_argument("--precision", type=str, choices=["16bit", "8bit", "4bit"], default="16bit")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "cqa", "both"], default="both")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of samples per prompt")
    parser.add_argument("--batch_size", type=int, default=4, help="Mini-batch size for generation (lower = less VRAM)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions per dataset (0 for all)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save JSONL logs")
    parser.add_argument("--max_new_tokens_gsm8k", type=int, default=512, help="Max new tokens for GSM8K")
    parser.add_argument("--max_new_tokens_cqa", type=int, default=384, help="Max new tokens for CQA (384 avoids mid-reasoning cutoffs)")

    args = parser.parse_args()
    if args.n_samples < 1 or args.batch_size < 1:
        parser.error("--n_samples and --batch_size must both be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1 or args.top_k < 0:
        parser.error("--temperature must be positive, --top_p in (0, 1], and --top_k non-negative")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    datasets_to_run = []
    if args.dataset in ["gsm8k", "both"]:
        datasets_to_run.append(("gsm8k", load_gsm8k_dataset()))
    if args.dataset in ["cqa", "both"]:
        datasets_to_run.append(("cqa", load_cqa_dataset()))

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)

    safe_model_name = args.model.replace("/", "_")

    for dataset_name, data in datasets_to_run:
        if args.limit > 0:
            data = data[:args.limit]

        output_file = os.path.join(args.output_dir, f"{safe_model_name}_{args.precision}_{dataset_name}_n{args.n_samples}.jsonl")
        dataset_fingerprint = hashlib.sha256(json.dumps(
            [(item['qid'], item['question'], item['ground_truth']) for item in data],
            ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")).hexdigest()

        # Check existing to resume
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

        expected_config = {
            "type": "run_config", "model": args.model, "precision": args.precision,
            "dataset": dataset_name, "n_samples": args.n_samples,
            "batch_size": args.batch_size, "temperature": args.temperature,
            "top_k": args.top_k, "top_p": args.top_p, "seed": args.seed,
            "max_new_tokens_gsm8k": args.max_new_tokens_gsm8k,
            "max_new_tokens_cqa": args.max_new_tokens_cqa,
            "total_questions": len(data),
            "dataset_fingerprint": dataset_fingerprint,
            "model_revision": getattr(model.config, "_commit_hash", None),
        }
        if processed_ids and existing_config is None:
            raise RuntimeError(f"Cannot safely resume {output_file}: existing records have no run_config header. Move the file or start a fresh output directory.")
        if existing_config is not None:
            mismatches = [key for key, value in expected_config.items()
                          if existing_config.get(key) != value]
            if mismatches:
                raise RuntimeError(f"Cannot resume {output_file}: run settings differ for {', '.join(mismatches)}. Use a fresh output directory.")

        print(f"Starting {dataset_name}. Found {len(processed_ids)} already processed items.")

        with open(output_file, 'a') as f:
            # Only write config if file doesn't already have one
            if existing_config is None:
                run_config = {
                    **expected_config,
                    "max_new_tokens_gsm8k": args.max_new_tokens_gsm8k,
                    "max_new_tokens_cqa": args.max_new_tokens_cqa,
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                f.write(json.dumps(run_config) + "\n")
                f.flush()

            # Preserve fixed chunk boundaries across resumes while filling the
            # requested sequence batch. Partially completed chunks are regenerated
            # with the same seed; only missing question records are appended.
            questions_per_batch = max(1, args.batch_size // args.n_samples)
                 
            # Process in chunks of questions
            for i in tqdm(range(0, len(data), questions_per_batch), desc=f"Processing {dataset_name}"):
                chunk = data[i:i + questions_per_batch]
                if all(item['qid'] in processed_ids for item in chunk):
                    continue
                
                # Set per-chunk seed for reproducibility across resumes
                chunk_seed = _qid_seed(args.seed, chunk[0]['qid'])
                hf_set_seed(chunk_seed)
                torch.manual_seed(chunk_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(chunk_seed)
                
                prompts = []
                extractors = []
                max_news = []
                
                for item in chunk:
                    if dataset_name == "gsm8k":
                        messages = get_gsm8k_messages(item['question'])
                        extractors.append(extract_gsm8k_answer)
                        max_news.append(args.max_new_tokens_gsm8k)
                    else:
                        messages = get_cqa_messages(item['question'], item['choices'])
                        extractors.append(extract_cqa_answer)
                        max_news.append(args.max_new_tokens_cqa)
                        
                    prompts.append(build_prompt(tokenizer, messages))
                
                # We use the max of max_news for the whole batch
                batch_max_new = max(max_news)
                
                # OOM-safe generation: catch CUDA OOM, clear cache, adapt batch size
                grouped_responses = None
                current_batch_size = args.batch_size
                while current_batch_size > 0:
                    try:
                        grouped_responses, tps, latency, grouped_lengths = generate_batch_prompts(
                            model, tokenizer, prompts,
                            n_samples=args.n_samples,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            max_new_tokens=batch_max_new,
                            batch_size=current_batch_size,
                        )
                        break  # Success!
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            torch.cuda.empty_cache()
                            if current_batch_size == 1:
                                raise RuntimeError(f"CUDA OOM at batch_size=1 for {chunk[0]['qid']}; stopping so the run cannot appear complete with missing questions.") from e
                            current_batch_size = max(1, current_batch_size // 2)
                            print(f"\n⚠️  OOM on chunk starting at {chunk[0]['qid']}. "
                                  f"Reducing batch_size to {current_batch_size} and retrying...")
                        else:
                            raise  # Re-raise non-OOM errors
                            
                if grouped_responses is None:
                    continue
                
                # Process metrics and log for each question
                for q_idx, item in enumerate(chunk):
                    if item['qid'] in processed_ids:
                        continue
                    responses = grouped_responses[q_idx]
                    gen_lengths = grouped_lengths[q_idx]
                    extractor = extractors[q_idx]
                    max_new = max_news[q_idx]
                    
                    extracted = [extractor(resp) for resp in responses]
                    metrics = compute_metrics(extracted, item['ground_truth'])
                    
                    cutoffs = [l >= max_new for l in gen_lengths]
                    metrics["cutoff_rate"] = sum(cutoffs) / len(cutoffs) if len(cutoffs) > 0 else 0.0

                    record = {
                        "qid": item['qid'],
                        "dataset": dataset_name,
                        "question": item['question'],
                        "ground_truth": item['ground_truth'],
                        "metrics": metrics,
                        "performance": {
                            "tokens_per_sec": tps,
                            "latency_sec": latency / len(chunk),
                            "effective_batch_size": current_batch_size,
                        },
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

                    f.write(json.dumps(record) + "\n")
                    processed_ids.add(item['qid'])
                
                f.flush()  # Ensure it writes to disk immediately after each chunk


if __name__ == "__main__":
    main()
