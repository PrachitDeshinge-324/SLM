import argparse
import json
import os
import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

import torch
from transformers import set_seed as hf_set_seed

from src.experiment_d.data_loader import load_cqa_dataset, load_gsm8k_dataset
from src.experiment_d.prompts import get_gsm8k_messages, get_cqa_messages, build_prompt
from src.experiment_d.inference import load_model_and_tokenizer, generate_n_samples, generate_batch_prompts
from src.experiment_d.extractors import extract_gsm8k_answer, extract_cqa_answer
from src.experiment_d.metrics import compute_metrics


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

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Fix #9: Set seed for reproducibility
    hf_set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

        output_file = os.path.join(args.output_dir, f"{safe_model_name}_{args.precision}_{dataset_name}.jsonl")

        # Check existing to resume
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("type") == "run_config":
                            continue  # skip config header
                        processed_ids.add(record['qid'])
                    except Exception:
                        pass

        print(f"Starting {dataset_name}. Found {len(processed_ids)} already processed items.")

        with open(output_file, 'a') as f:
            # Fix #6: Write run-level metadata header if file is new
            if len(processed_ids) == 0:
                run_config = {
                    "type": "run_config",
                    "model": args.model,
                    "precision": args.precision,
                    "dataset": dataset_name,
                    "n_samples": args.n_samples,
                    "batch_size": args.batch_size,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "total_questions": len(data),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                f.write(json.dumps(run_config) + "\n")
                f.flush()

            # Filter out already processed items
            unprocessed_data = [item for item in data if item['qid'] not in processed_ids]
            
            # Determine how many questions to process together (so total sequences = batch_size roughly)
            questions_per_batch = max(1, args.batch_size // args.n_samples)
            if questions_per_batch < 1:
                questions_per_batch = 1
                
            # Process in chunks of questions
            for i in tqdm(range(0, len(unprocessed_data), questions_per_batch), desc=f"Processing {dataset_name}"):
                chunk = unprocessed_data[i:i + questions_per_batch]
                
                prompts = []
                extractors = []
                max_news = []
                
                for item in chunk:
                    if dataset_name == "gsm8k":
                        messages = get_gsm8k_messages(item['question'])
                        extractors.append(extract_gsm8k_answer)
                        max_news.append(512)
                    else:
                        messages = get_cqa_messages(item['question'], item['choices'])
                        extractors.append(extract_cqa_answer)
                        max_news.append(256)
                        
                    prompts.append(build_prompt(tokenizer, messages))
                
                # We use the max of max_news for the whole batch
                batch_max_new = max(max_news)
                
                # Generate samples for the entire chunk of questions at once
                grouped_responses, tps, latency, grouped_lengths = generate_batch_prompts(
                    model, tokenizer, prompts,
                    n_samples=args.n_samples,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    max_new_tokens=batch_max_new,
                    batch_size=args.batch_size,
                )
                
                # Process metrics and log for each question
                for q_idx, item in enumerate(chunk):
                    responses = grouped_responses[q_idx]
                    gen_lengths = grouped_lengths[q_idx]
                    extractor = extractors[q_idx]
                    max_new = max_news[q_idx]
                    
                    extracted = [extractor(resp) for resp in responses]
                    metrics = compute_metrics(extracted, item['ground_truth'])
                    
                    cutoffs = [l >= max_new for l in gen_lengths]
                    metrics["cutoff_rate"] = sum(cutoffs) / len(cutoffs) if len(cutoffs) > 0 else 0.0

                    # We divide the total latency by the number of questions in the chunk to get per-question latency approx
                    record = {
                        "qid": item['qid'],
                        "dataset": dataset_name,
                        "question": item['question'],
                        "ground_truth": item['ground_truth'],
                        "metrics": metrics,
                        "performance": {
                            "tokens_per_sec": tps,  # Overall TPS for the batch
                            "latency_sec": latency / len(chunk),
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
                
                f.flush()  # Ensure it writes to disk immediately after each chunk


if __name__ == "__main__":
    main()
