"""
B1 across the four cells: Qwen3.5-0.8B and 2B at 16-bit and 4-bit, first 200 GSM8K test questions, 512 tokens.

Not pre-registered: written on 30 Oct after seeing every result of 01_program_agreement.py, so everything here is
exploratory. 01 stays the record of the pre-registered tests. Programs are not run again: answers come from the
execution caches that 01 wrote (outputs/exec_cache_*.json).

Section 1: per sample. Programs minus prose, failed programs and cut-off prose samples counted as wrong.
Section 2: is the tie-break gain real? 01 scores the prose vote with ties going to the first answer seen, so a
  tie-break selector can win by beating an arbitrary rule. Here the baseline splits ties (the share of the tied top
  answers that are right), the convention of the damage-type analysis, and each selector's own remaining ties are
  split the same way. Also counts the questions with a tied prose vote and where S3 moves the 01 score.
Section 3: does 4-bit hurt programs as much as prose? 4-bit minus 16-bit per sample and on the vote (ties split),
  and the difference between the two.
Intervals are 95% from 2,000 resamples of questions. Reference answers are read only to score.
"""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
spec = importlib.util.spec_from_file_location("b1", HERE / "01_program_agreement.py")
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

from src.experiment_d.answer_matching import answers_match  # noqa: E402
from src.experiment_d.run_files import safe_name  # noqa: E402

MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B")
PRECISIONS = ("16bit", "4bit")


def top_answers(keys: list) -> list:
    counts = Counter(k for k in keys if k is not None)
    if not counts:
        return []
    top = max(counts.values())
    return [k for k, c in counts.items() if c == top]


def split_score(tied: list, truth) -> float:
    return float(np.mean([answers_match(k, truth) for k in tied])) if tied else 0.0


def s3_tied(prose: list, program: list) -> list:
    """The prose top answers that survive the tie-break by program counts."""
    tied = top_answers(prose)
    if len(tied) < 2:
        return tied if tied else top_answers(program)
    counts = Counter(k for k in program if k is not None)
    best = max(counts[k] for k in tied)
    return [k for k in tied if counts[k] == best]


def cell(model: str, precision: str) -> dict:
    prose = b1.load(b1.run_path("results", model, precision, 8))
    program = b1.load(b1.run_path("results/prompt_variants/program", model, precision, 8))
    qids = sorted(set(prose) & set(program))
    keys, _ = b1.program_keys({q: program[q] for q in qids}, f"{safe_name(model)}_{precision}", 8)
    out = {key: [] for key in ("prose", "program", "vote_split", "vote_first", "s3_split", "s3_first", "cross_split",
                               "program_vote_split", "tied")}
    for q in qids:
        truth = prose[q]["ground_truth"]
        p, g = b1.prose_keys(prose[q]), keys[q]
        out["prose"].append(np.mean([k is not None and answers_match(k, truth) for k in p]))
        out["program"].append(np.mean([k is not None and answers_match(k, truth) for k in g]))
        out["vote_split"].append(split_score(top_answers(p), truth))
        first = b1.vote(p)
        out["vote_first"].append(float(first is not None and answers_match(first, truth)))
        out["s3_split"].append(split_score(s3_tied(p, g), truth))
        chosen = b1.tiebreak(p, g)
        out["s3_first"].append(float(chosen is not None and answers_match(chosen, truth)))
        out["cross_split"].append(split_score(top_answers(p + g), truth))
        out["program_vote_split"].append(split_score(top_answers(g), truth))
        out["tied"].append(len(top_answers(p)) > 1)
    return {key: np.array(value, float) for key, value in out.items()}


def interval(difference: np.ndarray) -> str:
    index = np.random.default_rng(0).integers(0, len(difference), (b1.B, len(difference)))
    means = difference[index].mean(1)
    return (f"{100 * difference.mean():+.1f} [{100 * np.percentile(means, 2.5):+.1f}, "
            f"{100 * np.percentile(means, 97.5):+.1f}]")


def main():
    cells = {(m, p): cell(m, p) for m in MODELS for p in PRECISIONS}
    print("=== 1. Per sample, 200 questions")
    print("model | precision | prose | program | program - prose")
    for (m, p), c in cells.items():
        print(f"{m} | {p} | {100 * c['prose'].mean():.1f} | {100 * c['program'].mean():.1f} | "
              f"{interval(c['program'] - c['prose'])}")

    print("\n=== 2. Vote selectors against prose vote")
    print("model | precision | tied prose votes | prose vote: first answer / ties split | S3 - prose, as in 01 "
          "(both first answer) | S3 - prose, ties split | cross-modal - prose, ties split | S3 wins, losses (as in 01)")
    for (m, p), c in cells.items():
        move = c["s3_first"] - c["vote_first"]
        print(f"{m} | {p} | {int(c['tied'].sum())} | {100 * c['vote_first'].mean():.1f} / "
              f"{100 * c['vote_split'].mean():.1f} | {interval(move)} | {interval(c['s3_split'] - c['vote_split'])} | "
              f"{interval(c['cross_split'] - c['vote_split'])} | {int((move > 0).sum())}, {int((move < 0).sum())}")

    print("\n=== 3. 4-bit minus 16-bit, same 200 questions")
    print("model | metric | prose | program | program damage - prose damage")
    for m in MODELS:
        low, high = cells[(m, "4bit")], cells[(m, "16bit")]
        for name, prose_key, program_key in (("per sample", "prose", "program"),
                                             ("vote, ties split", "vote_split", "program_vote_split")):
            a = low[prose_key] - high[prose_key]
            b = low[program_key] - high[program_key]
            print(f"{m} | {name} | {interval(a)} | {interval(b)} | {interval(b - a)}")


if __name__ == "__main__":
    main()
