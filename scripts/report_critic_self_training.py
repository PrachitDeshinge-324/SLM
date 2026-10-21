"""
Report on critic self-training: one shared round 0, then one round of training per arm.

  arm vote           labels from the majority vote (the earlier self-training rule)
  arm critic         labels from critic + vote (self_training.select_critic_labels)
  arm critic_reason  the critic labels, plus the critic's critiques that agree with them as a second task

Section 1, labels (round 0 pool): for each rule, how many questions it labels, how many of them are contested,
  and how many labels are right. The reference answers of the pool are read here, after the fact, and only
  here: nothing that selects labels or trains sees this report. Also: the questions the critic adds to the
  vote's and the ones it drops, with their accuracy.
Section 2, test: each arm's round 1 against round 0 on the same test questions, re-scored from the raw samples,
  with paired 95% intervals over questions; and each critic arm against the vote arm.

--labels_only prints section 1 alone (it needs only round 0 and the critic run, not the trained arms).
Output: printed, and saved to results/self_training/<base>/report_critic_<model>_<precision>.txt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_self_training import column, paired_difference, score_run  # noqa: E402

from src.experiment_d.answer_matching import answers_match
from src.experiment_d.benchmarks import load_gsm8k_train_answers
from src.experiment_d.run_files import (
    adapter_dir,
    generator_run_path,
    parse_n_samples,
    read_question_records,
    self_training_round_dir,
    signals_path,
)
from src.experiment_d.self_training import select_critic_labels, select_critiques, select_pseudo_labels

ARMS = ("vote", "critic", "critic_reason")


def parse_args():
    parser = argparse.ArgumentParser(description="Report on critic self-training arms sharing one round 0")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--precision", type=str, default="4bit")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--pool_dataset", type=str, default="gsm8k_train")
    parser.add_argument("--n_samples", type=str, default="8")
    parser.add_argument("--base", type=str, required=True, help="Experiment holding round 0; arm <base>_<arm> holds its round 1")
    parser.add_argument("--signals_tag", type=str, required=True, help="Tag of the critic run over the pool")
    parser.add_argument("--critic", type=str, default="critic_forced")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--labels_only", action="store_true")
    args = parser.parse_args()
    args.n_samples = parse_n_samples(args.n_samples)
    return args


def critic_cells(path: Path, critic: str) -> dict:
    cells = {}
    for line in path.open():
        row = json.loads(line) if line.strip() else {}
        if "qid" in row and row.get(critic) is not None:
            cells[(row["qid"], row["trace_index"])] = row[critic]
    return cells


def accuracy(labels: dict, answers: dict) -> str:
    right = [answers_match(label, answers[qid]) for qid, label in labels.items() if qid in answers]
    return f"{sum(right)}/{len(right)} ({100 * np.mean(right):.1f}%)" if right else "-"


def label_section(args) -> list:
    pool_file = generator_run_path(self_training_round_dir(args.results_dir, args.base, 0), args.model, args.precision,
                                   args.pool_dataset, args.n_samples)
    records = read_question_records(pool_file)
    cells = critic_cells(signals_path(args.results_dir, args.model, args.precision, args.pool_dataset, args.n_samples,
                                      args.base, 0, tag=args.signals_tag), args.critic)
    answers = load_gsm8k_train_answers()
    vote_items = select_pseudo_labels(records, args.pool_dataset, args.threshold)
    critic_items = select_critic_labels(records, {k: c.get("forced_decision") is True for k, c in cells.items()},
                                        args.pool_dataset, args.threshold)
    vote = {item["qid"]: item["label"] for item in vote_items}
    critic = {item["qid"]: item["label"] for item in critic_items}
    contested = {item["qid"] for item in critic_items if item["contested"]}
    critiques = select_critiques(records, cells, critic, args.pool_dataset)

    lines = [f"Pool: {len(records)} questions ({pool_file.name}); critic run: {len(cells)} judged solutions.",
             "Reference answers below are read only for this report.", "",
             "rule | labelled questions | labels right | training solutions",
             f"vote | {len(vote)} | {accuracy(vote, answers)} | {len(vote_items)}",
             f"critic + vote | {len(critic)} ({len(contested)} contested) | {accuracy(critic, answers)} | {len(critic_items)}",
             "", "Where the rules differ:",
             f"  labelled by critic only: {len(set(critic) - set(vote))} | right "
             f"{accuracy({q: critic[q] for q in set(critic) - set(vote)}, answers)}",
             f"  labelled by vote only (dropped by the critic): {len(set(vote) - set(critic))} | right "
             f"{accuracy({q: vote[q] for q in set(vote) - set(critic)}, answers)}",
             f"  labelled by both, different label: "
             f"{sum(not answers_match(critic[q], vote[q]) for q in set(critic) & set(vote))}"]
    agree = [answers_match(critic[c["qid"]], answers[c["qid"]]) for c in critiques if c["qid"] in answers]
    lines += ["", f"Critiques for arm critic_reason: {sum(c['verdict'] for c in critiques)} approving, "
              f"{sum(not c['verdict'] for c in critiques)} rejecting; on questions whose label is right: "
              f"{sum(agree)}/{len(agree)}" if agree else "", ""]
    return lines


def test_section(args) -> list:
    round0 = generator_run_path(self_training_round_dir(args.results_dir, args.base, 0), args.model, args.precision,
                                args.dataset, args.n_samples)
    runs = {"round 0": score_run(round0, args.dataset, args.threshold)}
    for arm in ARMS:
        path = generator_run_path(self_training_round_dir(args.results_dir, f"{args.base}_{arm}", 1), args.model,
                                  args.precision, args.dataset, args.n_samples)
        if path.exists():
            runs[arm] = score_run(path, args.dataset, args.threshold)
    qids = sorted(set.intersection(*(set(scored) for scored in runs.values())))
    lines = [f"Test: {len(qids)} questions answered by every run.", "",
             "run | single sample | cut off | right when finished | voted | trained examples (solutions + critiques)"]
    for name, scored in runs.items():
        single, cutoff = column(scored, qids, "single"), column(scored, qids, "cutoff")
        finished = single.sum() / (1 - cutoff).sum() if (1 - cutoff).sum() else float("nan")
        trained = "-"
        if name != "round 0":
            log = adapter_dir(args.results_dir, f"{args.base}_{name}", 1, args.model, args.precision) / "training_log.json"
            if log.exists():
                data = json.loads(log.read_text())
                trained = f"{data['trained_solutions']} + {data.get('trained_critiques', 0)}"
        lines.append(f"{name} | {single.mean():.3f} | {cutoff.mean():.3f} | {finished:.3f} | "
                     f"{column(scored, qids, 'voted').mean():.3f} | {trained}")
    lines += ["", "difference in points [95%] | single sample | voted | cut off"]
    pairs = [(arm, "round 0") for arm in ARMS if arm in runs] + [(arm, "vote") for arm in ARMS[1:] if arm in runs and "vote" in runs]
    for after, before in pairs:
        lines.append(f"{after} - {before} | " + " | ".join(
            paired_difference(column(runs[after], qids, key), column(runs[before], qids, key))
            for key in ("single", "voted", "cutoff")))
    return lines


def main():
    args = parse_args()
    lines = [f"Critic self-training: {args.model} {args.precision}, base {args.base}, critic run {args.signals_tag}", ""]
    lines += ["=== 1. Labels on the round 0 pool"] + label_section(args)
    if not args.labels_only:
        lines += ["=== 2. Test answers after one round"] + test_section(args)
    text = "\n".join(lines)
    print(text)
    if not args.labels_only:
        out = self_training_round_dir(args.results_dir, args.base, 0).parent / f"report_critic_{args.model.replace('/', '_')}_{args.precision}.txt"
        out.write_text(text + "\n")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
