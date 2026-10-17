"""
Generator against verifier on the same questions and the same traces, for the Generator vs Verifier page.

Every verifier run judges the traces of one generator run, so both are scored on identical data: the first
VERIFIER_QUESTIONS questions every kept precision of a model completed (standard_questions, as on the Phase 2
page), all N traces each, cut-off traces counted as wrong (they never vote).

  single      share of all traces that are right: what one sampled answer scores
  ver_single  one answer picked among the traces the verifier approved, per question (a random trace when it
              approved none), averaged over questions. Per question on purpose: pooling approved traces over
              questions favours easy questions
  voted       the majority answer among finished traces is right
  ver_voted   the majority among the approved finished traces (all finished traces when it approved none)

Reference answers are used only to score.
"""
from collections import Counter
from pathlib import Path

import numpy as np

from .rescoring import load_verifier_traces
from .run_files import is_legacy_file, parse_run_name
from .web_export import PRECISIONS, VERIFIER_QUESTIONS, standard_questions

FIELDS = ("single", "ver_single", "voted", "ver_voted")


def question_scores(traces: list) -> dict:
    """The four accuracies of one question, plus how many verdicts were parsed and approved."""
    right = [bool(t["correct"]) and not t["truncated"] for t in traces]
    finished = [t for t in traces if t["answer"] is not None and not t["truncated"]]
    voted = ver_voted = 0.0
    if finished:
        top = Counter(t["answer"] for t in finished).most_common(1)[0][0]
        voted = float(any(t["correct"] for t in finished if t["answer"] == top))
        pool = [t for t in finished if t["approved"] is True] or finished
        top_v = Counter(t["answer"] for t in pool).most_common(1)[0][0]
        ver_voted = float(any(t["correct"] for t in pool if t["answer"] == top_v))
    approved_right = [r for r, t in zip(right, traces) if t["approved"] is True]
    pick_from = approved_right or right
    return {"single": sum(right) / len(right), "ver_single": sum(pick_from) / len(pick_from),
            "voted": voted, "ver_voted": ver_voted,
            "n": len(traces), "parsed": sum(t["approved"] is not None for t in traces),
            "approved": len(approved_right)}


def paired_delta(after: np.ndarray, before: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict:
    """Mean of after - before over questions, with a 95% bootstrap interval over questions."""
    diff = after - before
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(resamples, len(diff)))].mean(axis=1)
    return {"mean": float(diff.mean()), "lo": float(np.quantile(means, 0.025)), "hi": float(np.quantile(means, 0.975))}


def score_run(question_traces: list) -> dict:
    per_q = [question_scores(ts) for ts in question_traces]
    col = {f: np.array([q[f] for q in per_q]) for f in FIELDS}
    n_traces = sum(q["n"] for q in per_q)
    row = {f: float(col[f].mean()) for f in FIELDS}
    row.update({
        "n_questions": len(per_q),
        "parse_coverage": sum(q["parsed"] for q in per_q) / n_traces,
        "approval_rate": sum(q["approved"] for q in per_q) / n_traces,
        "delta_single": paired_delta(col["ver_single"], col["single"]),
        "delta_voted": paired_delta(col["ver_voted"], col["voted"]),
    })
    return row


def collect(results_dir, questions: int = VERIFIER_QUESTIONS) -> list:
    """One row per (dataset, model, N, precision) with a verifier run, on its model's standard questions."""
    groups = {}
    for path in sorted(Path(results_dir).glob("*/verifier_output/*/verifier_output_*.jsonl")):
        run = parse_run_name(path.name)
        if is_legacy_file(path) or run is None:
            continue
        groups.setdefault((run["dataset"], run["model"], run["n_samples"]), {})[run["precision"]] = path

    rows = []
    for (dataset, model, n_samples), files in sorted(groups.items()):
        traces = {p: load_verifier_traces(path, dataset) for p, path in files.items()}
        selected, short = standard_questions(traces, n_samples, questions)
        if selected is None:
            continue
        for precision in sorted(set(files) - set(short), key=PRECISIONS.index):
            row = score_run([traces[precision][q] for q in selected])
            row.update({"dataset": dataset, "model": model, "n_samples": n_samples, "precision": precision})
            rows.append(row)
    return rows
