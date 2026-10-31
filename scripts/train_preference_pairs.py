"""
Self-training, one round, preference objective: train a LoRA adapter with length-normalised DPO on
pairs of the model's own solutions.

Reads the pool answers the previous round's model wrote, labels each question by majority vote,
and pairs a complete majority solution (chosen) with a complete minority one (rejected). The frozen
base scores every pair once as the reference; then a fresh adapter is trained. No reference answer
is read anywhere: the pool run file holds none, and the evaluation set is opened only to check that
no pool question is also an evaluation question.

The loss per pair is weight * (DPO + nll_weight * NLL of the chosen solution). The weight is the vote
margin (--weighting margin), scaled to mean 1. With --preference_end exclude the end tokens are left
out of the DPO score of both solutions, so the rejected one's stop is never pushed down; the chosen
one's NLL still includes them. Each epoch uses another distinct pair of a question where it has one
(--pairs_per_question). The vote runs on the pool of --pool_experiment (default --experiment).

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
    END_FORMATS,
    answer_end,
    attach_lora,
    build_preference_example,
    enable_gradient_checkpointing,
    reference_logps,
    train_lora_dpo,
)
from src.experiment_d.model_loading import PRECISIONS, load_model_and_tokenizer
from src.experiment_d.run_files import adapter_dir, parse_n_samples
from src.experiment_d.self_training import (
    PAIRINGS,
    assert_disjoint,
    curriculum_pairs,
    rotate_pairs,
    select_preference_pairs,
)

LOG_NAME = "training_log.json"
PAIRS_NAME = "preference_pairs.jsonl"
WEIGHTINGS = ("margin", "none")
PREFERENCE_ENDS = ("exclude", "include")
# What a training log written before these settings existed was trained with.
LEGACY_SETTINGS = {"end_format": "legacy", "pool_experiment": None, "pairs_per_question": 1, "skip_ties": False,
                   "nll_weight": 0.0, "weighting": "none", "preference_end": "include", "curriculum": None,
                   "gradient_checkpointing": False, "chunked_loss": False}
# Settings that change memory only, never the trained adapter: an adapter trained with or without them
# counts as the same run.
MEMORY_ONLY = ("gradient_checkpointing", "chunked_loss")


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
    parser.add_argument("--pool_experiment", type=str, default=None,
                        help="Experiment whose round-(k-1) pool answers are used (default: --experiment), so a new adapter can read an existing pool")
    parser.add_argument("--pairs_per_question", type=int, default=1,
                        help="Distinct (chosen, rejected) pairs kept per question; epoch e uses its (e mod n)-th")
    parser.add_argument("--skip_ties", action="store_true", help="Drop questions where another answer ties the majority one")
    parser.add_argument("--epochs", type=float, default=2.0, help="Passes over the pairs; sets the number of updates")
    parser.add_argument("--curriculum", type=str, default=None,
                        help="Comma-separated agreement thresholds, one stage each, e.g. 0.75,0.625,0.5: a pass over the questions "
                             "reaching each one, in order. Replaces --epochs; the last must equal --threshold")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Pairs per update")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="Pairs per forward pass (memory only; does not change the update)")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=2.0, help="Scale of the per-token log-ratio margin")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--nll_weight", type=float, default=0.0,
                        help="Weight of the chosen solution's NLL (per token, end included) next to the DPO loss; ScPO uses 1")
    parser.add_argument("--weighting", choices=WEIGHTINGS, default="none",
                        help="margin: weight each pair by its question's vote margin (scaled to mean 1); none: equal")
    parser.add_argument("--preference_end", choices=PREFERENCE_ENDS, default="include",
                        help="exclude: leave the end tokens out of the DPO score, so stopping is never penalised directly")
    parser.add_argument("--end_format", choices=END_FORMATS, default="chat",
                        help="chat: end each solution as the base model does (<|im_end|> \\n <|endoftext|>); legacy: <|endoftext|> only")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute activations in the backward pass: much less memory, about a third more time")
    parser.add_argument("--chunked_loss", action="store_true",
                        help="Score tokens with a chunked output head: much less memory, same values")
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
    if args.pairs_per_question < 1 or (args.pairing == "longest_shortest" and args.pairs_per_question != 1):
        parser.error("--pairs_per_question must be at least 1, and 1 with --pairing longest_shortest")
    if args.nll_weight < 0:
        parser.error("--nll_weight must not be negative")
    if args.curriculum is not None:
        stages = [float(value) for value in args.curriculum.split(",")]
        if sorted(stages, reverse=True) != stages or abs(stages[-1] - args.threshold) > 1e-9:
            parser.error("--curriculum thresholds must not rise, and the last must equal --threshold")
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def main():
    args = parse_args()
    settings = {key: value for key, value in vars(args).items() if key != "results_dir"}
    settings["objective"] = "dpo"
    output_dir = adapter_dir(args.results_dir, args.experiment, args.round, args.model, args.precision)
    log_file = output_dir / LOG_NAME

    if log_file.exists():
        previous = {**LEGACY_SETTINGS, **json.loads(log_file.read_text())["settings"]}
        changed = sorted(key for key in settings if key not in MEMORY_ONLY and previous.get(key) != settings[key])
        if changed:
            raise RuntimeError(f"{output_dir} holds an adapter trained with other settings ({', '.join(changed)}). Use a new --experiment.")
        print(f"Round {args.round} adapter already trained at {output_dir}; nothing to do.")
        return

    pool_file, pool_header, records = load_pool(args)
    assert_disjoint([record["question"] for record in records],
                    [item["question"] for item in load_benchmark(args.eval_dataset)])
    print(f"Pool: {len(records)} questions from {pool_file}; none is an evaluation question or the prompt's example.")

    selected = select_preference_pairs(records, args.dataset, args.threshold, args.pairing,
                                       args.pairs_per_question, args.skip_ties)
    questions = sorted({item["qid"] for item in selected})
    print(f"Paired by vote (agreement >= {args.threshold}, {args.pairing}"
          f"{', ties skipped' if args.skip_ties else ''}): {len(questions)} questions, {len(selected)} distinct pairs.")
    if not selected:
        raise RuntimeError("No question has both a confident majority and a complete minority solution, so there is nothing to train on.")

    model, tokenizer = load_model_and_tokenizer(args.model, args.precision)
    compute_dtype = model.dtype
    end = answer_end(model, tokenizer, args.end_format)

    kept, pairs = [], []
    for item in selected:
        example = build_preference_example(tokenizer, args.dataset, item["question"], item["chosen"], item["rejected"],
                                           end, args.max_length, mask_end=args.preference_end == "exclude")
        if example is not None:
            kept.append(item)
            pairs.append(example)
    if not pairs:
        raise RuntimeError(f"Every pair has a solution longer than --max_length {args.max_length}.")
    for item, pair in zip(kept, pairs):
        pair["qid"] = item["qid"]

    # One entry per question and pass; entries are shared dicts, so each pair is scored once as reference.
    stage_sizes = None
    if args.curriculum is not None:
        for item, pair in zip(kept, pairs):
            pair["agreement"] = item["agreement"]
        schedule, stage_sizes = curriculum_pairs(pairs, [float(value) for value in args.curriculum.split(",")])
        print(f"Curriculum: stages of {stage_sizes} questions at agreement {args.curriculum}.")
    else:
        schedule = rotate_pairs(pairs, math.ceil(args.epochs))
    for pair in pairs:
        pair["weight"] = 1.0
    if args.weighting == "margin":
        margins = {id(pair): item["margin"] for item, pair in zip(kept, pairs)}
        mean_margin = sum(margins[id(pair)] for pair in schedule) / len(schedule)
        if mean_margin <= 0:
            raise RuntimeError("Every pair has margin 0 (a tie), so margin weights are all zero. Use --skip_ties or --weighting none.")
        for pair in pairs:
            pair["weight"] = margins[id(pair)] / mean_margin

    # The reference is the frozen base, scored before the adapter exists.
    print(f"Scoring {len(pairs)} pairs with the frozen {args.precision} base (reference)...")
    for pair, (chosen, rejected) in zip(pairs, reference_logps(model, tokenizer, pairs, args.micro_batch_size, compute_dtype,
                                                               chunked=args.chunked_loss)):
        pair["ref_chosen"], pair["ref_rejected"] = chosen, rejected

    passes = len(schedule) if args.curriculum is not None else args.epochs * len(questions)
    steps = math.ceil(passes / args.train_batch_size)
    model, target_modules = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout, seed=args.train_seed)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Training {len(schedule)} pair-epochs ({len(pairs)} distinct pairs, {len(questions)} questions) for {steps} steps "
          f"on the {args.precision} base ({trainable:,} adapter parameters, {compute_dtype}, "
          f"beta {args.beta}, nll weight {args.nll_weight}, weighting {args.weighting}, preference end {args.preference_end}).")

    history = train_lora_dpo(model, tokenizer, schedule, steps=steps, batch_size=args.train_batch_size,
                             micro_batch_size=args.micro_batch_size, learning_rate=args.learning_rate,
                             beta=args.beta, seed=args.train_seed, compute_dtype=compute_dtype,
                             max_grad_norm=args.max_grad_norm, nll_weight=args.nll_weight,
                             chunked_loss=args.chunked_loss, stage_sizes=stage_sizes)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    with (output_dir / PAIRS_NAME).open("w") as handle:
        for item, pair in zip(kept, pairs):
            handle.write(json.dumps({**item, "ref_chosen": pair["ref_chosen"], "ref_rejected": pair["ref_rejected"],
                                     "weight": pair["weight"]}) + "\n")
    lengths = lambda side: [item[f"{side}_length"] for item in kept]
    log = {
        "settings": settings,
        "timestamp": datetime.datetime.now().isoformat(),
        "pool_file": str(pool_file),
        # Which model wrote the pairs: None for the untrained model, else the previous round's adapter.
        "pool_adapter_sha256": pool_header.get("adapter_sha256"),
        "pool_questions": len(records),
        "labelled_questions": len(questions),
        "labelled_qids": questions,
        "selected_pairs": len(selected),
        "skipped_too_long": len(selected) - len(pairs),
        "pair_epochs": len(schedule),
        "stage_sizes": stage_sizes,
        # Named as in the SFT log so the self-training report reads both; here it counts pairs.
        "trained_solutions": len(pairs),
        "train_steps": steps,
        "mean_chosen_length": sum(lengths("chosen")) / len(kept),
        "mean_rejected_length": sum(lengths("rejected")) / len(kept),
        "chosen_longer_share": sum(c > r for c, r in zip(lengths("chosen"), lengths("rejected"))) / len(kept),
        "agreement_counts": dict(sorted(Counter(f"{item['agreement']:.3f}" for item in kept).items())),
        "mean_weight": sum(pair["weight"] for pair in schedule) / len(schedule),
        "target_modules": target_modules,
        "trainable_parameters": trainable,
        "compute_dtype": str(compute_dtype),
        "end_token": tokenizer.convert_ids_to_tokens(end if isinstance(end, int) else list(end)),
        **{key: history[key] for key in ("losses", "margins", "preferred", "nll", "chosen_shift", "rejected_shift",
                                         "end_chosen", "end_rejected")},
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
