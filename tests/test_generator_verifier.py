"""Tests for generator_verifier: per-question accuracies of the generator and of its verifier."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_d.generator_verifier import paired_delta, question_scores, score_run


def t(answer, correct, approved, truncated=False):
    return {"answer": answer, "correct": correct, "approved": approved, "truncated": truncated}


def test_verifier_breaks_a_wrong_vote():
    traces = [t("5", False, False), t("5", False, False), t("5", False, None), t("7", True, True)]
    s = question_scores(traces)
    assert s["single"] == 0.25 and s["voted"] == 0.0
    assert s["ver_voted"] == 1.0 and s["ver_single"] == 1.0
    assert s["parsed"] == 3 and s["approved"] == 1


def test_nothing_approved_falls_back():
    traces = [t("5", True, False), t("6", False, False)]
    s = question_scores(traces)
    assert s["ver_single"] == s["single"] == 0.5 and s["ver_voted"] == s["voted"]


def test_cut_off_never_counts():
    traces = [t("7", True, True, truncated=True), t("5", False, False)]
    s = question_scores(traces)
    assert s["single"] == 0.0 and s["voted"] == 0.0 and s["ver_single"] == 0.0


def test_delta_and_run():
    d = paired_delta(np.array([1.0, 1.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert abs(d["mean"] - 0.25) < 1e-12 and d["lo"] <= d["mean"] <= d["hi"]
    row = score_run([[t("1", True, True), t("2", False, False)], [t("3", False, True), t("3", False, True)]])
    assert row["n_questions"] == 2 and abs(row["ver_single"] - 0.5) < 1e-12 and row["delta_single"]["mean"] == 0.25


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("All generator_verifier tests passed!")
