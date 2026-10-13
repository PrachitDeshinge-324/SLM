"""
Self-training, one round: train a LoRA adapter on the model's own vote-filtered solutions.

Reads the pool answers the previous round's model wrote, labels each question by majority vote,
keeps the confident ones and trains a fresh adapter on the frozen base. No reference answer is
read anywhere: the pool run file holds none, and the evaluation set is opened only to check that
no pool question is also an evaluation question.

Input : results/self_training/<experiment>/round<k-1>/<pool dataset>/generator/...
Output: results/self_training/<experiment>/round<k>/adapters/<model>_<precision>/  (adapter + training_log.json)
"""
import argparse
import datetime
import json
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

from src.experiment_d.benchmarks import load_benchmark
from src.experiment_d.lora_training import attach_lora, build_training_example, end_token_id, train_lora
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.run_files import (
    adapter_dir,
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    self_training_round_dir,
)
from src.experiment_d.self_training import assert_disjoint, select_pseudo_labels

LOG_NAME = "training_log.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on a model's own vote-filtered solutions (one self-training round)")
    parser.add_argument("--results_dir", "--output_dir", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--precision", type=str, choices=PRECISIONS, default="4bit")
    parser.add_argument("--dataset", type=str, default="gsm8k_train", help="Unlabelled pool the model answered")
    parser.add_argument("--eval_dataset", type=str, default="gsm8k", help="Evaluation set; read only to check the pool shares no question with it")
    parser.add_argument("--n_samples", type=str, default="8", help="Samples per pool question")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the self-training run")
    parser.add_argument("--round", type=int, required=True, help="Round to train (1 = first adapter, trained on the untrained model's answers)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Share of all samples that must agree for a question to be labelled")
    parser.add_argument("--train_steps", type=int, default=200, help="Optimizer updates (fixed, so every run gets the same number)")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Solutions per update")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Solutions per forward pass (memory only; does not change the update)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=768, help="Longest prompt + solution in tokens; longer ones are skipped")
    parser.add_argument("--train_seed", type=int, default=0)

    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round starts at 1; round 0 is the untrained model")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def load_pool(args):
    """Pool records the previous round's model wrote, and that run's header. The pool must be complete."""
    pool_file = generator_run_path(self_training_round_dir(args.results_dir, args.experiment, args.round - 1),
                                   args.model, args.precision, args.dataset, args.n_samples)
    if not pool_file.exists():
        raise FileNotFoundError(f"No pool answers at {pool_file}. Generate them before training round {args.round}.")
    header, records = read_run_config(pool_file), read_question_records(pool_file)
    if len(records) != header.get("total_questions"):
        raise RuntimeError(f"Pool run {pool_file} is incomplete: {len(records)} of {header.get('total_questions')} questions.")
    if not header.get("answers_hidden") or any(record.get("ground_truth") is not None for record in records):
        raise RuntimeError(f"Pool run {pool_file} carries reference answers. Self-training must use an unlabelled pool.")
    return pool_file, header, records


def main():
    args = parse_args()
    settings = {key: value for key, value in vars(args).items() if key != "results_dir"}
    output_dir = adapter_dir(args.results_dir, args.experiment, args.round, args.model, args.precision)
    log_file = output_dir / LOG_NAME

    if log_file.exists():
        previous = json.loads(log_file.read_text())["settings"]
        if previous != settings:
            changed = sorted(key for key in settings if previous.get(key) != settings[key])
            raise RuntimeError(f"{output_dir} holds an adapter trained with other settings ({', '.join(changed)}). Use a new --experiment.")
        print(f"Round {args.round} adapter already trained at {output_dir}; nothing to do.")
        return

    pool_file, pool_header, records = load_pool(args)
    assert_disjoint([record["question"] for record in records],
                    [item["question"] for item in load_benchmark(args.eval_dataset)])
    print(f"Pool: {len(records)} questions from {pool_file}; none is an evaluation question or the prompt's example.")

    selected = select_pseudo_labels(records, args.dataset, args.threshold)
    labelled = sorted({item["qid"] for item in selected})
    print(f"Labelled by vote (agreement >= {args.threshold}): {len(labelled)} questions, {len(selected)} solutions.")
    if not selected:
        raise RuntimeError("No question reached the agreement threshold, so there is nothing to train on.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    compute_dtype = model.dtype
    end_token = end_token_id(model, tokenizer)

    examples = [build_training_example(tokenizer, args.dataset, item["question"], item["trace"], end_token, args.max_length)
                for item in selected]
    skipped_too_long = sum(example is None for example in examples)
    examples = [example for example in examples if example is not None]
    if not examples:
        raise RuntimeError(f"Every selected solution is longer than --max_length {args.max_length}.")

    model, target_modules = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, seed=args.train_seed)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Training {len(examples)} solutions for {args.train_steps} steps on the {args.precision} base "
          f"({trainable:,} adapter parameters, {compute_dtype}).")

    losses = train_lora(model, tokenizer, examples, steps=args.train_steps, batch_size=args.train_batch_size,
                        micro_batch_size=args.micro_batch_size, learning_rate=args.learning_rate,
                        seed=args.train_seed, compute_dtype=compute_dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    log = {
        "settings": settings,
        "timestamp": datetime.datetime.now().isoformat(),
        "pool_file": str(pool_file),
        # Which model wrote the labels: None for the untrained model, else the previous round's adapter.
        "pool_adapter_sha256": pool_header.get("adapter_sha256"),
        "pool_questions": len(records),
        "labelled_questions": len(labelled),
        "labelled_qids": labelled,
        "selected_solutions": len(selected),
        "skipped_too_long": skipped_too_long,
        "trained_solutions": len(examples),
        "agreement_counts": dict(sorted(Counter(
            f"{item['agreement']:.3f}" for item in {item["qid"]: item for item in selected}.values()).items())),
        "target_modules": target_modules,
        "trainable_parameters": trainable,
        "compute_dtype": str(compute_dtype),
        "end_token": tokenizer.convert_ids_to_tokens(end_token),
        "losses": losses,
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
