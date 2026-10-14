"""
Self-training, one round, preference objective: train a LoRA adapter with length-normalised DPO on
pairs of the model's own solutions.

Reads the pool answers the previous round's model wrote, labels each question by majority vote,
and pairs a complete majority solution (chosen) with a complete minority one (rejected). The frozen
base scores every pair once as the reference; then a fresh adapter is trained. No reference answer
is read anywhere: the pool run file holds none, and the evaluation set is opened only to check that
no pool question is also an evaluation question.

Input : results/self_training/<experiment>/round<k-1>/<pool dataset>/generator/...
Output: results/self_training/<experiment>/round<k>/adapters/<model>_<precision>/
        (adapter + preference_pairs.jsonl with the reference scores + training_log.json)
"""
import argparse
import datetime
import json
import math
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.warning_filters import silence_harmless_warnings

silence_harmless_warnings()
load_dotenv()

import torch

from scripts.train_self_labels import load_pool
from src.experiment_d.benchmarks import load_benchmark
from src.experiment_d.lora_training import (
    attach_lora,
    build_preference_example,
    end_token_id,
    reference_logps,
    train_lora_dpo,
)
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.run_files import adapter_dir, parse_n_samples
from src.experiment_d.self_training import PAIRINGS, assert_disjoint, select_preference_pairs

LOG_NAME = "training_log.json"
PAIRS_NAME = "preference_pairs.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter with length-normalised DPO on a model's own solutions (one self-training round)")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="4bit")
    parser.add_argument("--dataset", type=str, default="gsm8k_train", help="Unlabelled pool the model answered")
    parser.add_argument("--eval_dataset", type=str, default="gsm8k", help="Evaluation set; read only to check the pool shares no question with it")
    parser.add_argument("--n_samples", type=str, default="8", help="Samples per pool question")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the self-training run")
    parser.add_argument("--round", type=int, required=True, help="Round to train (1 = first adapter, trained on the untrained model's answers)")
    parser.add_argument("--threshold", type=float, default=0.625, help="Share of all samples that must agree for a question to be paired (0.625 = 5 of 8)")
    parser.add_argument("--pairing", type=str, choices=PAIRINGS, default="similar_length",
                        help="similar_length: the chosen/rejected pair closest in length; longest_shortest: the length-confounded ablation")
    parser.add_argument("--epochs", type=float, default=2.0, help="Passes over the pairs; sets the number of updates")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Pairs per update")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Pairs per forward pass (memory only; does not change the update)")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=2.0, help="Scale of the per-token log-ratio margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=768, help="Longest prompt + solution in tokens; pairs with a longer side are skipped")
    parser.add_argument("--train_seed", type=int, default=0)

    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round starts at 1; round 0 is the untrained model")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def main():
    args = parse_args()
    settings = {key: value for key, value in vars(args).items() if key != "results_dir"}
    settings["objective"] = "dpo"
    output_dir = adapter_dir(args.results_dir, args.experiment, args.round, args.model, args.precision)
    log_file = output_dir / LOG_NAME

    if log_file.exists():
        previous = json.loads(log_file.read_text())["settings"]
        if previous != settings:
            changed = sorted(key for key in set(settings) | set(previous) if previous.get(key) != settings.get(key))
            raise RuntimeError(f"{output_dir} holds an adapter trained with other settings ({', '.join(changed)}). Use a new --experiment.")
        print(f"Round {args.round} adapter already trained at {output_dir}; nothing to do.")
        return

    pool_file, pool_header, records = load_pool(args)
    assert_disjoint([record["question"] for record in records],
                    [item["question"] for item in load_benchmark(args.eval_dataset)])
    print(f"Pool: {len(records)} questions from {pool_file}; none is an evaluation question or the prompt's example.")

    selected = select_preference_pairs(records, args.dataset, args.threshold, args.pairing)
    print(f"Paired by vote (agreement >= {args.threshold}, {args.pairing}): {len(selected)} questions.")
    if not selected:
        raise RuntimeError("No question has both a confident majority and a complete minority solution, so there is nothing to train on.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    compute_dtype = model.dtype
    end_token = end_token_id(model, tokenizer)

    kept, pairs = [], []
    for item in selected:
        example = build_preference_example(tokenizer, args.dataset, item["question"], item["chosen"], item["rejected"],
                                           end_token, args.max_length)
        if example is not None:
            kept.append(item)
            pairs.append(example)
    if not pairs:
        raise RuntimeError(f"Every pair has a solution longer than --max_length {args.max_length}.")

    # The reference is the frozen base, scored before the adapter exists.
    print(f"Scoring {len(pairs)} pairs with the frozen {args.precision} base (reference)...")
    for pair, (chosen, rejected) in zip(pairs, reference_logps(model, tokenizer, pairs, args.micro_batch_size, compute_dtype)):
        pair["ref_chosen"], pair["ref_rejected"] = chosen, rejected

    steps = math.ceil(args.epochs * len(pairs) / args.train_batch_size)
    model, target_modules = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, seed=args.train_seed)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Training {len(pairs)} pairs for {steps} steps ({args.epochs} epochs) on the {args.precision} base "
          f"({trainable:,} adapter parameters, {compute_dtype}, beta {args.beta}).")

    history = train_lora_dpo(model, tokenizer, pairs, steps=steps, batch_size=args.train_batch_size,
                             micro_batch_size=args.micro_batch_size, learning_rate=args.learning_rate,
                             beta=args.beta, seed=args.train_seed, compute_dtype=compute_dtype,
                             max_grad_norm=args.max_grad_norm)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    with (output_dir / PAIRS_NAME).open("w") as handle:
        for item, pair in zip(kept, pairs):
            handle.write(json.dumps({**item, "ref_chosen": pair["ref_chosen"], "ref_rejected": pair["ref_rejected"]}) + "\n")
    lengths = lambda side: [item[f"{side}_length"] for item in kept]
    log = {
        "settings": settings,
        "timestamp": datetime.datetime.now().isoformat(),
        "pool_file": str(pool_file),
        # Which model wrote the pairs: None for the untrained model, else the previous round's adapter.
        "pool_adapter_sha256": pool_header.get("adapter_sha256"),
        "pool_questions": len(records),
        "labelled_questions": len(selected),
        "labelled_qids": sorted(item["qid"] for item in selected),
        "selected_pairs": len(selected),
        "skipped_too_long": len(selected) - len(pairs),
        # Named as in the SFT log so the self-training report reads both; here it counts pairs.
        "trained_solutions": len(pairs),
        "train_steps": steps,
        "mean_chosen_length": sum(lengths("chosen")) / len(kept),
        "mean_rejected_length": sum(lengths("rejected")) / len(kept),
        "chosen_longer_share": sum(c > r for c, r in zip(lengths("chosen"), lengths("rejected"))) / len(kept),
        "agreement_counts": dict(sorted(Counter(f"{item['agreement']:.3f}" for item in kept).items())),
        "target_modules": target_modules,
        "trainable_parameters": trainable,
        "compute_dtype": str(compute_dtype),
        "end_token": tokenizer.convert_ids_to_tokens(end_token),
        "losses": history["losses"],
        "margins": history["margins"],
        "preferred": history["preferred"],
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
    }
    # Written last: its presence marks the adapter as complete.
    log_file.write_text(json.dumps(log, indent=2))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
