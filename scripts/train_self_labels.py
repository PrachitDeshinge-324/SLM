"""
Self-training, one round: train a LoRA adapter on the model's own vote-filtered solutions.

Reads the pool answers the previous round's model wrote, labels each question by majority vote
(--labels vote) or by critic + vote (--labels critic, from a critic run over the pool by
scripts/run_signals.py), keeps the confident ones and trains a fresh adapter on the frozen base.
--critiques also trains on the critic's own critiques that agree with the labels, as a second task
with the critic's prompt. No reference answer is read anywhere: the pool run and signals files hold
none, and the evaluation set is opened only to check that no pool question is also an evaluation
question.

--pool_experiment reads the pool (and its critic run) from another experiment's round, so several
arms can share one round 0.

Input : results/self_training/<pool experiment>/round<k-1>/<pool dataset>/generator/...
        results/self_training/<pool experiment>/round<k-1>/<pool dataset>/signals/... (--labels critic)
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
from src.experiment_d.lora_training import (
    END_FORMATS,
    answer_end,
    attach_lora,
    build_example,
    build_training_example,
    enable_gradient_checkpointing,
    train_lora,
)
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.run_files import (
    adapter_dir,
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    self_training_round_dir,
    signals_path,
)
from src.experiment_d.prompts import build_prompt
from src.experiment_d.self_training import (
    LABEL_RULES,
    assert_disjoint,
    select_critic_labels,
    select_critiques,
    select_pseudo_labels,
)
from src.experiment_d.verifier_variants import CRITICS, answer_prefix, verification_messages

LOG_NAME = "training_log.json"
# What a training log written before these settings existed was trained with.
LEGACY_SETTINGS = {"end_format": "legacy", "labels": "vote", "pool_experiment": None, "critic": "critic_forced",
                   "signals_tag": None, "critiques": False, "critique_max_length": 1536,
                   "gradient_checkpointing": False, "chunked_loss": False}


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
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Solutions per forward pass (memory; same update in expectation, but LoRA dropout draws other masks)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_length", type=int, default=768, help="Longest prompt + solution in tokens; longer ones are skipped")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--end_format", choices=END_FORMATS, default="chat",
                        help="chat: end each answer as the base model does (<|im_end|> \\n <|endoftext|>); "
                             "legacy: the stop token alone, as every adapter before 21 Oct 2026 (to resume those)")
    parser.add_argument("--pool_experiment", type=str, default=None,
                        help="Experiment whose round k-1 holds the pool answers (default: --experiment)")
    parser.add_argument("--labels", choices=LABEL_RULES, default="vote",
                        help="vote: majority vote; critic: critic + vote from the critic run over the pool")
    parser.add_argument("--critic", choices=CRITICS, default="critic_forced", help="Critic of the signals run")
    parser.add_argument("--signals_tag", type=str, default=None,
                        help="Tag of the pool's signals file, e.g. critic_forced_v2_t704_pilot (for --labels critic / --critiques)")
    parser.add_argument("--critiques", action="store_true",
                        help="Also train on the critic's critiques that agree with the labels (critic prompt -> critique)")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations in the backward pass: much less memory, about a third slower, same gradients")
    parser.add_argument("--chunked_loss", action="store_true",
                        help="Apply the output head to a few hundred answer tokens at a time: far less memory, same loss")
    parser.add_argument("--critique_max_length", type=int, default=1536,
                        help="Longest critic prompt + critique in tokens; longer ones are skipped")

    args = parser.parse_args()
    if args.round < 1:
        parser.error("--round starts at 1; round 0 is the untrained model")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if (args.labels == "critic" or args.critiques) and not args.signals_tag:
        parser.error("--labels critic and --critiques need --signals_tag")
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def load_pool(args):
    """Pool records the previous round's model wrote, and that run's header. The pool must be complete."""
    pool_file = generator_run_path(self_training_round_dir(args.results_dir, args.pool_experiment or args.experiment, args.round - 1),
                                   args.model, args.precision, args.dataset, args.n_samples)
    if not pool_file.exists():
        raise FileNotFoundError(f"No pool answers at {pool_file}. Generate them before training round {args.round}.")
    header, records = read_run_config(pool_file), read_question_records(pool_file)
    if len(records) != header.get("total_questions"):
        raise RuntimeError(f"Pool run {pool_file} is incomplete: {len(records)} of {header.get('total_questions')} questions.")
    if not header.get("answers_hidden") or any(record.get("ground_truth") is not None for record in records):
        raise RuntimeError(f"Pool run {pool_file} carries reference answers. Self-training must use an unlabelled pool.")
    return pool_file, header, records


def load_critic(args, pool_file, records):
    """{(qid, sample index): critic cell} of the critic run over the pool; it must cover every pool question."""
    path = signals_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples,
                        args.pool_experiment or args.experiment, args.round - 1, tag=args.signals_tag)
    if not path.exists():
        raise FileNotFoundError(f"No critic run over the pool at {path}. Run scripts/run_signals.py on it first.")
    header, cells, seen = {}, {}, set()
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "run_config":
            header = row
        elif "qid" in row:
            seen.add(row["qid"])
            if row.get(args.critic) is not None:
                cells[(row["qid"], row["trace_index"])] = row[args.critic]
    if Path(header.get("generator_run", "")).name != pool_file.name or args.critic not in header.get("critics", []):
        raise RuntimeError(f"{path} is not a {args.critic} run over {pool_file.name} (header: {header.get('generator_run')}, "
                           f"critics {header.get('critics')}).")
    missing = [record["qid"] for record in records if record["qid"] not in seen]
    if missing:
        raise RuntimeError(f"The critic run {path} has not reached {len(missing)} pool questions (e.g. {missing[0]}); finish it first.")
    return path, header, cells


def main():
    args = parse_args()
    settings = {key: value for key, value in vars(args).items() if key != "results_dir"}
    output_dir = adapter_dir(args.results_dir, args.experiment, args.round, args.model, args.precision)
    log_file = output_dir / LOG_NAME

    if log_file.exists():
        previous = {**LEGACY_SETTINGS, **json.loads(log_file.read_text())["settings"]}
        if previous != settings:
            changed = sorted(key for key in settings if previous.get(key) != settings[key])
            raise RuntimeError(f"{output_dir} holds an adapter trained with other settings ({', '.join(changed)}). Use a new --experiment.")
        print(f"Round {args.round} adapter already trained at {output_dir}; nothing to do.")
        return

    pool_file, pool_header, records = load_pool(args)
    assert_disjoint([record["question"] for record in records],
                    [item["question"] for item in load_benchmark(args.eval_dataset)])
    print(f"Pool: {len(records)} questions from {pool_file}; none is an evaluation question or the prompt's example.")

    critic_file, cells = None, {}
    if args.labels == "critic" or args.critiques:
        critic_file, critic_header, cells = load_critic(args, pool_file, records)
        print(f"Critic run: {critic_file} ({len(cells)} judged solutions, {critic_header.get('verdicts')} verdicts each).")
    if args.labels == "critic":
        verdicts = {key: cell.get("forced_decision") is True for key, cell in cells.items()}
        selected = select_critic_labels(records, verdicts, args.dataset, args.threshold)
    else:
        selected = select_pseudo_labels(records, args.dataset, args.threshold)
    labelled = sorted({item["qid"] for item in selected})
    contested = sorted({item["qid"] for item in selected if item.get("contested")})
    print(f"Labelled by {args.labels} (share >= {args.threshold}): {len(labelled)} questions "
          f"({len(contested)} contested), {len(selected)} solutions.")
    if not selected:
        raise RuntimeError("No question reached the agreement threshold, so there is nothing to train on.")
    critiques = []
    if args.critiques:
        labels = {item["qid"]: item["label"] for item in selected}
        critiques = select_critiques(records, cells, labels, args.dataset)
        print(f"Critiques agreeing with the labels: {sum(c['verdict'] for c in critiques)} approving, "
              f"{sum(not c['verdict'] for c in critiques)} rejecting.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    compute_dtype = model.dtype
    end = answer_end(model, tokenizer, args.end_format)

    examples = [build_training_example(tokenizer, args.dataset, item["question"], item["trace"], end, args.max_length)
                for item in selected]
    skipped_too_long = sum(example is None for example in examples)
    examples = [example for example in examples if example is not None]
    if not examples:
        raise RuntimeError(f"Every selected solution is longer than --max_length {args.max_length}.")
    solution_examples = len(examples)
    critique_examples = [build_example(tokenizer,
                                       build_prompt(tokenizer, verification_messages(args.critic, c["question"], c["solution"]))
                                       + answer_prefix(args.critic),
                                       c["critique"], end, args.critique_max_length) for c in critiques]
    skipped_critiques = sum(example is None for example in critique_examples)
    examples += [example for example in critique_examples if example is not None]
    if critiques:
        print(f"Training examples: {solution_examples} solutions + {len(examples) - solution_examples} critiques "
              f"({skipped_critiques} critiques longer than {args.critique_max_length} tokens skipped).")

    model, target_modules = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, seed=args.train_seed)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Training {len(examples)} examples for {args.train_steps} steps on the {args.precision} base "
          f"({trainable:,} adapter parameters, {compute_dtype}).")

    losses = train_lora(model, tokenizer, examples, steps=args.train_steps, batch_size=args.train_batch_size,
                        micro_batch_size=args.micro_batch_size, learning_rate=args.learning_rate,
                        seed=args.train_seed, compute_dtype=compute_dtype, chunked_loss=args.chunked_loss)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    log = {
        "settings": settings,
        "timestamp": datetime.datetime.now().isoformat(),
        "pool_file": str(pool_file),
        # Which model wrote the labels: None for the untrained model, else the previous round's adapter.
        "pool_adapter_sha256": pool_header.get("adapter_sha256"),
        "pool_questions": len(records),
        "critic_file": str(critic_file) if critic_file else None,
        "labelled_questions": len(labelled),
        "labelled_qids": labelled,
        "contested_labelled_qids": contested,
        "selected_solutions": len(selected),
        "skipped_too_long": skipped_too_long,
        "trained_solutions": solution_examples,
        "selected_critiques": {"approving": sum(c["verdict"] for c in critiques),
                               "rejecting": sum(not c["verdict"] for c in critiques)},
        "trained_critiques": len(examples) - solution_examples,
        "skipped_critiques_too_long": skipped_critiques,
        "agreement_counts": dict(sorted(Counter(
            f"{item['agreement']:.3f}" for item in {item["qid"]: item for item in selected}.values()).items())),
        "target_modules": target_modules,
        "trainable_parameters": trainable,
        "compute_dtype": str(compute_dtype),
        "end_token": tokenizer.convert_ids_to_tokens(end if isinstance(end, int) else list(end)),
        "losses": losses,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
    }
    # Written last: its presence marks the adapter as complete.
    log_file.write_text(json.dumps(log, indent=2))
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
