"""
Report on critic self-training: one shared round 0, then one or more rounds of training per arm.

  arm vote           labels from the majority vote (the earlier self-training rule)
  arm critic         labels from critic + vote (self_training.select_critic_labels)
  arm critic_reason  the critic labels, plus the critic's critiques that agree with them as a second task

Section 1, labels (round 0 pool): for each rule, how many questions it labels, how many of them are contested,
  and how many labels are right. The reference answers of the pool are read here, after the fact, and only
  here: nothing that selects labels or trains sees this report. Also: the questions the critic adds to the
  vote's and the ones it drops, with their accuracy.
Section 2, test: each arm's round 1 against round 0 on the same test questions, re-scored from the raw samples,
  with paired 95% intervals over questions; and each critic arm against the vote arm. Arms trained again with
  another seed (<base>_<arm>_s<k>) are rows of their own; the difference between two seeds of one arm is the
  training-noise floor. The intervals resample questions only, so they do not include that noise.
  Later rounds (run_critic_self_training.sh --rounds N) are rows '<arm> r<k>', compared with round 0, with the
  arm's previous round, and (critic arms) with the vote arm of the same round.
Section 3, labels: for every trained run, the labels it learned from and how many are right; after round 1 they
  are re-made from the arm's own previous round, so this shows label drift.

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

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match
from src.experiment_d.self_consistency import majority_vote
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
from src.experiment_d.signals import with_judge

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
    extractor = get_extractor(args.pool_dataset)
    plurality = {r["qid"]: majority_vote([None if s.get("cutoff", False) else extractor(s["response"])
                                          for s in r["raw_samples"]])[0] for r in records}
    critiques = select_critiques(records, cells, critic, args.pool_dataset)

    lines = [f"Pool: {len(records)} questions ({pool_file.name}); critic run: {len(cells)} judged solutions.",
             "Reference answers below are read only for this report.", "",
             "rule | labelled questions | labels right | training solutions",
             f"vote | {len(vote)} | {accuracy(vote, answers)} | {len(vote_items)}",
             f"critic + vote | {len(critic)} ({len(contested)} contested) | {accuracy(critic, answers)} | {len(critic_items)}",
             "", "Where the rules differ:",
             f"  labelled by critic only: {len(set(critic) - set(vote))} | right "
             f"{accuracy({q: critic[q] for q in set(critic) - set(vote)}, answers)} | the plain vote's top answer there "
             f"right {accuracy({q: plurality[q] for q in set(critic) - set(vote) if plurality.get(q)}, answers)}",
             f"  labelled by vote only (dropped by the critic): {len(set(vote) - set(critic))} | right "
             f"{accuracy({q: vote[q] for q in set(vote) - set(critic)}, answers)}",
             f"  labelled by both, different label: "
             f"{sum(not answers_match(critic[q], vote[q]) for q in set(critic) & set(vote))}"]
    agree = [answers_match(critic[c["qid"]], answers[c["qid"]]) for c in critiques if c["qid"] in answers]
    lines += ["", f"Critiques for arm critic_reason: {sum(c['verdict'] for c in critiques)} approving, "
              f"{sum(not c['verdict'] for c in critiques)} rejecting; on questions whose label is right: "
              f"{sum(agree)}/{len(agree)}" if agree else "", ""]
    return lines


def run_name(arm_seed: str, round_index: int) -> str:
    """Row name of a trained run: '<arm>[_s<k>]' for round 1 (the names of the one-round reports), else '... r<round>'."""
    return arm_seed if round_index == 1 else f"{arm_seed} r{round_index}"


def trained_runs(args) -> list:
    """(arm[_s<k>], round) of every trained run whose test answers exist, rounds counted up from 1 until one is missing."""
    root = self_training_round_dir(args.results_dir, args.base, 0).parent.parent
    out = []
    for arm in ARMS:
        seeds = sorted((int(d.name.rsplit("_s", 1)[1]) for d in root.glob(f"{args.base}_{arm}_s*")
                        if d.name.rsplit("_s", 1)[1].isdigit()))
        for arm_seed in [arm] + [f"{arm}_s{k}" for k in seeds]:
            round_index = 1
            while generator_run_path(self_training_round_dir(args.results_dir, f"{args.base}_{arm_seed}", round_index),
                                     args.model, args.precision, args.dataset, args.n_samples).exists():
                out.append((arm_seed, round_index))
                round_index += 1
    return out


def test_section(args) -> list:
    round0 = generator_run_path(self_training_round_dir(args.results_dir, args.base, 0), args.model, args.precision,
                                args.dataset, args.n_samples)
    runs = {"round 0": score_run(round0, args.dataset, args.threshold)}
    where = {}
    for arm_seed, round_index in trained_runs(args):
        name = run_name(arm_seed, round_index)
        where[name] = (arm_seed, round_index)
        path = generator_run_path(self_training_round_dir(args.results_dir, f"{args.base}_{arm_seed}", round_index),
                                  args.model, args.precision, args.dataset, args.n_samples)
        runs[name] = score_run(path, args.dataset, args.threshold)
    # A run that has not answered every round 0 question yet is listed, not compared.
    incomplete = {name: len(scored) for name, scored in runs.items()
                  if name != "round 0" and not set(runs["round 0"]) <= set(scored)}
    runs = {name: scored for name, scored in runs.items() if name not in incomplete}
    qids = sorted(set.intersection(*(set(scored) for scored in runs.values())))
    lines = [f"Test: {len(qids)} questions answered by every run."]
    lines += [f"Left out, not finished: {name} ({n}/{len(runs['round 0'])} questions)" for name, n in incomplete.items()]
    lines += ["",
             "run | single sample | cut off | right when finished | voted | trained examples (solutions + critiques)"]
    for name, scored in runs.items():
        single, cutoff = column(scored, qids, "single"), column(scored, qids, "cutoff")
        finished = single.sum() / (1 - cutoff).sum() if (1 - cutoff).sum() else float("nan")
        trained = "-"
        if name != "round 0":
            arm_seed, round_index = where[name]
            log = adapter_dir(args.results_dir, f"{args.base}_{arm_seed}", round_index, args.model,
                              args.precision) / "training_log.json"
            if log.exists():
                data = json.loads(log.read_text())
                trained = f"{data['trained_solutions']} + {data.get('trained_critiques', 0)}"
        lines.append(f"{name} | {single.mean():.3f} | {cutoff.mean():.3f} | {finished:.3f} | "
                     f"{column(scored, qids, 'voted').mean():.3f} | {trained}")
    lines += ["", "difference in points [95%] | single sample | voted | cut off"]
    trained = [name for name in runs if name != "round 0"]
    pairs = [(name, "round 0") for name in trained]
    for name in trained:
        arm_seed, round_index = where[name]
        arm = arm_seed.rsplit("_s", 1)[0] if "_s" in arm_seed and arm_seed.rsplit("_s", 1)[1].isdigit() else arm_seed
        # a later round against the arm's previous round; a critic run against the vote arm (seed 0) of the same
        # round; a seed k > 0 against seed 0 of its own arm in the same round
        candidates = [run_name(arm_seed, round_index - 1) if round_index > 1 else None,
                      run_name("vote", round_index) if arm != "vote" else None,
                      run_name(arm, round_index) if arm != arm_seed else None]
        pairs += [(name, before) for before in candidates if before in runs]
    for after, before in pairs:
        lines.append(f"{after} - {before} | " + " | ".join(
            paired_difference(column(runs[after], qids, key), column(runs[before], qids, key))
            for key in ("single", "voted", "cutoff")))
    return lines


def pool_critic_file(args, experiment: str, round_index: int):
    """The critic run over a round's pool: judged by that round's model (--round_critic self), else by the untrained one."""
    tags = [args.signals_tag]
    if round_index > 0:
        tags.insert(0, with_judge(args.signals_tag, adapter_dir(args.results_dir, experiment, round_index, args.model,
                                                                 args.precision)))
    for tag in tags:
        path = signals_path(args.results_dir, args.model, args.precision, args.pool_dataset, args.n_samples,
                            experiment, round_index, tag=tag)
        if path.exists():
            return path
    return None


def round_labels_section(args) -> list:
    """The labels every trained run learned from, re-made each round from the previous round's pool (drift)."""
    answers = load_gsm8k_train_answers()
    lines = ["run | learned from | rule | labelled questions | labels right | training solutions"]
    for arm_seed, round_index in trained_runs(args):
        experiment = args.base if round_index == 1 else f"{args.base}_{arm_seed}"
        pool_file = generator_run_path(self_training_round_dir(args.results_dir, experiment, round_index - 1),
                                       args.model, args.precision, args.pool_dataset, args.n_samples)
        if not pool_file.exists():
            continue
        records = read_question_records(pool_file)
        source = "round 0 (shared)" if round_index == 1 else f"its round {round_index - 1}"
        if arm_seed.startswith("vote"):
            rule, items = "vote", select_pseudo_labels(records, args.pool_dataset, args.threshold)
        else:
            critic_file = pool_critic_file(args, experiment, round_index - 1)
            if critic_file is None:
                lines.append(f"{run_name(arm_seed, round_index)} | {source} | critic + vote | no critic run found")
                continue
            cells = critic_cells(critic_file, args.critic)
            rule = "critic + vote" + (" (judged by that round's model)" if "_by_" in critic_file.name else "")
            items = select_critic_labels(records, {k: c.get("forced_decision") is True for k, c in cells.items()},
                                         args.pool_dataset, args.threshold)
        labels = {item["qid"]: item["label"] for item in items}
        lines.append(f"{run_name(arm_seed, round_index)} | {source} | {rule} | {len(labels)} | "
                     f"{accuracy(labels, answers)} | {len(items)}")
    return lines


def main():
    args = parse_args()
    lines = [f"Critic self-training: {args.model} {args.precision}, base {args.base}, critic run {args.signals_tag}", ""]
    lines += ["=== 1. Labels on the round 0 pool"] + label_section(args)
    if not args.labels_only:
        lines += ["=== 2. Test answers after each round"] + test_section(args)
        lines += ["", "=== 3. Labels each run learned from (reference answers read for this report only)"]
        lines += round_labels_section(args)
    text = "\n".join(lines)
    print(text)
    if not args.labels_only:
        out = self_training_round_dir(args.results_dir, args.base, 0).parent / f"report_critic_{args.model.replace('/', '_')}_{args.precision}.txt"
        out.write_text(text + "\n")
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
