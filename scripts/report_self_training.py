"""
Self-training report for one model at one precision.

For every round found under results/self_training/<experiment>/ it re-scores the test answers
from their raw samples and compares each trained round with round 0 on the same questions. It also
shows how many pool questions each round's model labelled and, after the fact, how many of those
labels were right. That last column is the only place the train answers are read; nothing in the
training loop sees this report.

Output: printed, and saved to results/self_training/<experiment>/report_<model>_<precision>.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure the repository root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.answer_matching import answers_match
from src.experiment_d.benchmarks import load_benchmark, load_gsm8k_train_answers
from src.experiment_d.evaluation_stats import bootstrap_interval
from src.experiment_d.rescoring import rescore_record
from src.experiment_d.run_files import (
    SELF_TRAINING_DIR,
    adapter_dir,
    adapter_fingerprint,
    generator_run_path,
    parse_n_samples,
    read_question_records,
    read_run_config,
    safe_name,
    self_training_round_dir,
)
from src.experiment_d.self_training import select_pseudo_labels, shared_questions


def parse_args():
    parser = argparse.ArgumentParser(description="Report on a self-training run (Experiment D)")
    parser.add_argument("--results_dir", "--directory", dest="results_dir", type=str, default="results", help="Root directory of the run files")
    parser.add_argument("--model", "--model_id", dest="model", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--precision", type=str, default="4bit", help="16bit, 8bit or 4bit")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Evaluation set")
    parser.add_argument("--pool_dataset", type=str, default="gsm8k_train", help="Unlabelled pool")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--experiment", type=str, required=True, help="Name of the self-training run")
    parser.add_argument("--threshold", type=float, default=0.75, help="Agreement needed for a label")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def score_run(path, dataset: str, threshold: float) -> dict:
    """qid -> per-question numbers of one run file, rebuilt from its raw samples."""
    scored = {}
    for record in read_question_records(path):
        metrics = rescore_record(record, dataset)
        samples = record["raw_samples"]
        scored[record["qid"]] = {
            "single": metrics["sample_accuracy"],
            "voted": float(bool(metrics["correctness"])),
            "cutoff": metrics["cutoff_rate"],
            "confident": float(metrics["confidence_all"] >= threshold - 1e-9),
            "distinct": metrics["diversity"],
            "length": float(np.mean([sample["length"] for sample in samples])),
        }
    return scored


def column(scored: dict, qids: list, key: str) -> np.ndarray:
    return np.array([scored[qid][key] for qid in qids], float)


def paired_difference(after: np.ndarray, before: np.ndarray) -> str:
    """Mean difference in points with a 95% interval from resampling questions."""
    difference = after - before
    low, high = bootstrap_interval(lambda index: difference[index].mean(), len(difference))
    return f"{100 * difference.mean():+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]"


def test_table(rounds: dict, qids: list) -> list:
    lines = ["round | single sample | cut off | complete samples | voted | confident | confident and right | distinct answers | tokens"]
    for index, scored in rounds.items():
        single, cutoff = column(scored, qids, "single"), column(scored, qids, "cutoff")
        voted, confident = column(scored, qids, "voted"), column(scored, qids, "confident") > 0
        complete_accuracy = single.sum() / (1 - cutoff).sum() if (1 - cutoff).sum() else float("nan")
        confident_right = voted[confident].mean() if confident.any() else float("nan")
        lines.append(f"{index:5d} | {single.mean():.3f} | {cutoff.mean():.3f} | {complete_accuracy:.3f} | {voted.mean():.3f} | "
                     f"{confident.mean():.3f} | {confident_right:.3f} | {column(scored, qids, 'distinct').mean():.2f} | "
                     f"{column(scored, qids, 'length').mean():.0f}")
    return lines


def pool_table(args, round_indices: list) -> list:
    """Labels each round's model produced on the pool, and how the next round trained on them."""
    answers = load_gsm8k_train_answers() if args.pool_dataset == "gsm8k_train" else {}
    lines = ["round | pool questions | labelled | labels right (checked afterwards) | kept from previous round | solutions | trained | loss first -> last"]
    previous_labels = None
    for index in round_indices:
        pool_file = generator_run_path(self_training_round_dir(args.results_dir, args.experiment, index),
                                       args.model, args.precision, args.pool_dataset, args.n_samples)
        if not pool_file.exists():
            continue
        records = read_question_records(pool_file)
        selected = select_pseudo_labels(records, args.pool_dataset, args.threshold)
        labels = {item["qid"]: item["label"] for item in selected}
        right = [answers_match(label, answers[qid]) for qid, label in labels.items() if qid in answers]
        kept = "-" if previous_labels is None else f"{len(set(labels) & set(previous_labels))}/{len(previous_labels)}"
        previous_labels = labels

        log_file = adapter_dir(args.results_dir, args.experiment, index + 1, args.model, args.precision) / "training_log.json"
        trained, loss = "-", "-"
        if log_file.exists():
            log = json.loads(log_file.read_text())
            trained = str(log["trained_solutions"])
            if log["losses"]:
                loss = f"{np.mean(log['losses'][:10]):.3f} -> {np.mean(log['losses'][-10:]):.3f}"
        lines.append(f"{index:5d} | {len(records)} | {len(labels)} ({len(labels) / len(records):.3f}) | "
                     f"{f'{np.mean(right):.3f}' if right else '-'} | {kept} | {len(selected)} | {trained} | {loss}")
    return lines


def leak_checks(args, round_indices: list) -> list:
    """The facts that make the comparison fair, re-derived from the files themselves."""
    lines = []
    test_questions = [item["question"] for item in load_benchmark(args.dataset)]
    for index in round_indices:
        round_dir = self_training_round_dir(args.results_dir, args.experiment, index)
        pool_file = generator_run_path(round_dir, args.model, args.precision, args.pool_dataset, args.n_samples)
        if pool_file.exists():
            records = read_question_records(pool_file)
            shared = shared_questions([record["question"] for record in records], test_questions)
            with_answers = sum(record.get("ground_truth") is not None for record in records)
            lines.append(f"round {index} pool: {len(shared)} questions shared with the test set or the prompt; "
                         f"{with_answers} records carry a reference answer")
        eval_file = generator_run_path(round_dir, args.model, args.precision, args.dataset, args.n_samples)
        if eval_file.exists() and index > 0:
            recorded = read_run_config(eval_file).get("adapter_sha256")
            actual = adapter_fingerprint(adapter_dir(args.results_dir, args.experiment, index, args.model, args.precision))
            lines.append(f"round {index} test run used this round's adapter: {'yes' if recorded == actual else 'NO'}")
    return lines


def main():
    args = parse_args()
    experiment_dir = Path(args.results_dir) / SELF_TRAINING_DIR / args.experiment
    round_indices = sorted(int(path.name[len("round"):]) for path in experiment_dir.glob("round*")
                           if path.is_dir() and path.name[len("round"):].isdigit())

    rounds = {}
    for index in round_indices:
        eval_file = generator_run_path(self_training_round_dir(args.results_dir, args.experiment, index),
                                       args.model, args.precision, args.dataset, args.n_samples)
        if eval_file.exists():
            rounds[index] = score_run(eval_file, args.dataset, args.threshold)
    if 0 not in rounds:
        print(f"No round 0 test run for {args.model} {args.precision} under {experiment_dir}")
        return

    # Compare rounds only on questions every round has answered.
    qids = sorted(set.intersection(*(set(scored) for scored in rounds.values())))
    out = [f"Self-training report: {args.model} {args.precision}, experiment '{args.experiment}'",
           f"Test set {args.dataset}: {len(qids)} questions answered in every round, {args.n_samples} samples each. "
           f"Confident = agreement >= {args.threshold}.", ""]

    out += ["Fairness checks"] + leak_checks(args, round_indices) + [""]
    out += ["Test questions (accuracies; a cut-off sample counts as wrong in 'single sample')"] + test_table(rounds, qids) + [""]

    out.append("Change against round 0, in points, same questions (95% interval over questions)")
    for index, scored in rounds.items():
        if index == 0:
            continue
        out.append(f"round {index}: single sample {paired_difference(column(scored, qids, 'single'), column(rounds[0], qids, 'single'))}"
                   f" | voted {paired_difference(column(scored, qids, 'voted'), column(rounds[0], qids, 'voted'))}")
    out.append("")

    # Two untrained runs of the same model: how far the numbers move with no training at all.
    stored_file = generator_run_path(args.results_dir, args.model, args.precision, args.dataset, args.n_samples)
    if stored_file.exists():
        stored = score_run(stored_file, args.dataset, args.threshold)
        common = sorted(set(qids) & set(stored))
        if common:
            out += [f"Noise floor: round 0 minus the stored untrained run, {len(common)} shared questions",
                    f"single sample {paired_difference(column(rounds[0], common, 'single'), column(stored, common, 'single'))}"
                    f" | voted {paired_difference(column(rounds[0], common, 'voted'), column(stored, common, 'voted'))}", ""]

    out += ["Pool labels (the model's own votes on unlabelled train questions)"] + pool_table(args, round_indices)

    text = "\n".join(out)
    print(text)
    report_file = experiment_dir / f"report_{safe_name(args.model)}_{args.precision}.txt"
    report_file.write_text(text + "\n")
    print(f"\nSaved to {report_file}")


if __name__ == "__main__":
    main()
