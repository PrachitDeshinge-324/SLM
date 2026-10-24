"""
Where do the Qwen3.5-0.8B self-training gains fall: on randomly damaged, systematically damaged or preserved
questions (classes from 01_damage_types.py, GSM8K)?

Prediction: training on the 4-bit model's own votes repairs random damage (its plurality is still right, so
the labels are right) and not systematic damage (its plurality is wrong, so the label is wrong or missing).

For each self-training experiment, on its own test questions: per class, the round 0 accuracy and the gain of
the trained round, per sample (a cut-off sample counts as wrong) and voted (ties split). The classes come from
the main runs (outputs/01_question_types.csv), which share no samples with the self-training runs, so the gains
are not biased by the classification.

Room to improve: a question already right in every sample cannot gain, and the classes differ in how often the
model is right. So the class contrasts are also taken inside bins of the trained precision's own accuracy in the
classification runs (p4 for a 4-bit experiment, p16 for 16-bit): within each bin, the class's mean gain minus
the preserved questions' mean gain, averaged over bins weighted by the class's questions; 95% interval from
resampling questions inside bins. The 16-bit experiments are the control: there the classes describe what
4-bit does to a question, which 16-bit training should not care about.
Reference answers are read only to score.
"""
import csv
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.run_files import generator_run_path, read_question_records, self_training_round_dir

types = import_module("01_damage_types")

MODEL = "Qwen/Qwen3.5-0.8B"
# name -> (precision, [(experiment, round) of round 0, then of each trained round to compare])
EXPERIMENTS = {
    "4-bit, 4 rounds, old end token (q08_sft_4bit_p500)":
        ("4bit", [("q08_sft_4bit_p500", 0), ("q08_sft_4bit_p500", 1), ("q08_sft_4bit_p500", 4)]),
    "16-bit, 4 rounds, old end token (q08_sft_16bit_p500)":
        ("16bit", [("q08_sft_16bit_p500", 0), ("q08_sft_16bit_p500", 1), ("q08_sft_16bit_p500", 4)]),
    "4-bit, 1 round, fixed end token (q0.8b_4bit_critic_p500_vote)":
        ("4bit", [("q0.8b_4bit_critic_p500", 0), ("q0.8b_4bit_critic_p500_vote", 1)]),
    "16-bit, 1 round, fixed end token (q0.8b_16bit_critic_p500_vote)":
        ("16bit", [("q0.8b_16bit_critic_p500", 0), ("q0.8b_16bit_critic_p500_vote", 1)]),
}
SHOWN = ("preserved", "random", "systematic", "helped", "both fail")
BINS = (0.0, 0.25, 0.5, 0.75, 1.0001)


def per_question(experiment: str, round_index: int, precision: str) -> dict:
    """qid -> (per-sample accuracy, voted accuracy with ties split) of one test run."""
    extractor = get_extractor("gsm8k")
    path = generator_run_path(self_training_round_dir(ROOT / "results", experiment, round_index), MODEL, precision, "gsm8k", 8)
    return {r["qid"]: types.summary(types.scored_samples(r, extractor)) for r in read_question_records(path)}


def bin_matched(gain: np.ndarray, kind: np.ndarray, pre: np.ndarray, target: str, rng, resamples: int = 2000):
    """Gain of `target` minus gain of 'preserved' inside accuracy bins, weighted by the target's questions."""
    def estimate(index):
        g, k, p = gain[index], kind[index], pre[index]
        diffs, weights = [], []
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            inside = (p >= lo) & (p < hi)
            t, c = inside & (k == target), inside & (k == "preserved")
            if t.any() and c.any():
                diffs.append(g[t].mean() - g[c].mean())
                weights.append(t.sum())
        return np.average(diffs, weights=weights) if weights else np.nan

    point = estimate(np.arange(len(gain)))
    bin_of = np.digitize(pre, BINS[1:-1])
    groups = [np.flatnonzero(bin_of == b) for b in np.unique(bin_of)]
    draws = [estimate(np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(resamples)]
    low, high = np.nanpercentile(draws, [2.5, 97.5])
    return point, low, high


def main():
    rows = {r["qid"]: r for r in csv.DictReader((HERE / "outputs" / "01_question_types.csv").open()) if r["dataset"] == "gsm8k"}
    rng = np.random.default_rng(0)
    for name, (precision, runs) in EXPERIMENTS.items():
        base = per_question(*runs[0], precision)
        print(f"\n=== {name}")
        for experiment, round_index in runs[1:]:
            after = per_question(experiment, round_index, precision)
            qids = [q for q in base if q in after and q in rows]
            kind = np.array([rows[q]["type"] for q in qids])
            pre = np.array([float(rows[q]["p4" if precision == "4bit" else "p16"]) for q in qids])
            single = np.array([after[q][0] - base[q][0] for q in qids])
            voted = np.array([after[q][1] - base[q][1] for q in qids])
            print(f"round {round_index} against round 0, {len(qids)} test questions")
            print("class | n | round 0 per sample | gain per sample | gain voted")
            for c in SHOWN + ("all",):
                m = np.ones(len(qids), bool) if c == "all" else kind == c
                if not m.any():
                    continue
                r0 = np.mean([base[q][0] for q, keep in zip(qids, m) if keep])
                print(f"{c} | {m.sum()} | {100 * r0:.1f} | {100 * single[m].mean():+.1f} | {100 * voted[m].mean():+.1f}")
            for target in ("random", "systematic"):
                point, low, high = bin_matched(single, kind, pre, target, rng)
                print(f"  per-sample gain, {target} minus preserved, matched on accuracy: "
                      f"{100 * point:+.1f} [{100 * low:+.1f}, {100 * high:+.1f}]")


if __name__ == "__main__":
    main()
