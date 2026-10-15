import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.verifier_metrics import metrics_from_counts, question_counts, summarize
from src.experiment_d.web_export import (
    build_web_data,
    calibration_curve,
    collect_verifier_runs,
    paired_intervals,
    standard_questions,
    token_limit,
    write_web_data,
)


def _trace(answer, correct, approved, truncated=False):
    return {"answer": answer, "correct": correct, "approved": approved, "truncated": truncated}


def test_question_counts_and_metrics():
    traces = [
        _trace("18", True, True),        # tp
        _trace("18", True, False),       # fn
        _trace("20", False, True),       # fp
        _trace("20", False, False),      # tn
        _trace("18", True, None),        # no verdict
        _trace("18", True, True, truncated=True),   # dropped, never votes
        _trace("7", False, True, truncated=None),   # truncation unknown: kept and counted
    ]
    counts = question_counts(traces)
    assert (counts["tp"], counts["fn"], counts["fp"], counts["tn"]) == (1, 1, 2, 1)
    assert counts["unparseable"] == 1 and counts["truncated"] == 1 and counts["truncation_unknown"] == 1
    assert counts["total"] == 6
    assert counts["gen_correct"] == 1       # "18" has three votes against two for "20"

    metrics = metrics_from_counts([counts, question_counts([])])   # a question with no traces still counts
    assert metrics["Total_Questions"] == 2
    assert metrics["Gen_Maj_Acc"] == 0.5
    assert metrics["Trace_FPR"] == 2 / 3
    assert metrics["Trace_Recall"] == 0.5
    assert summarize({"q": traces}, ["q", "missing"]) == metrics


def test_question_counts_can_keep_truncated_traces():
    traces = [
        _trace("18", True, True),                    # tp
        _trace("18", True, True, truncated=True),    # cut off: wrong whatever the extractor found, so fp
        _trace("20", False, False, truncated=True),  # tn
        _trace("20", False, None, truncated=True),   # no verdict
        _trace("20", False, True, truncated=True),   # fp
    ]
    dropped, kept = question_counts(traces), question_counts(traces, keep_truncated=True)
    assert dropped["total"] == 1 and kept["total"] == 5
    assert dropped["truncated"] == kept["truncated"] == 4
    assert (kept["tp"], kept["fp"], kept["fn"], kept["tn"], kept["unparseable"]) == (1, 2, 0, 1, 1)
    # the three cut-off "20" traces never vote, in either view
    assert kept["gen_correct"] == dropped["gen_correct"] == 1
    assert kept["ver_correct"] == 1


def test_verifier_guided_majority_falls_back_to_generator_majority():
    # the verifier approves only the wrong answer: the guided vote follows it
    traces = [_trace("18", True, False), _trace("18", True, False), _trace("20", False, True)]
    counts = question_counts(traces)
    assert counts["gen_correct"] == 1 and counts["ver_correct"] == 0
    # it rejects everything: fall back to the generator's majority
    rejects = [_trace("18", True, False), _trace("20", False, False)]
    assert question_counts(rejects)["ver_correct"] == question_counts(rejects)["gen_correct"]


def test_calibration_curve_bins_and_min_count():
    confidences = [1.0] * 6 + [0.7] * 5 + [0.2] * 4        # 0.2 bin has too few questions
    correct = [1] * 6 + [1, 1, 0, 0, 0] + [1, 0, 0, 0]
    curve = calibration_curve(confidences, correct)
    assert [point["x"] for point in curve] == [0.7, 1.0]    # 0.7 lands in the (0.6, 0.7] bin, not the next
    assert curve[0]["accuracy"] == 0.4 and curve[0]["count"] == 5
    assert curve[1]["accuracy"] == 1.0


def test_token_limit_key_spellings():
    assert token_limit({"max_new_tokens": 1024}, "gsm8k") == 1024
    assert token_limit({"max_new_tokens_gsm8k": 512, "max_new_tokens_cqa": 384}, "cqa") == 384
    assert token_limit({}, "gsm8k") is None


def test_paired_intervals_contain_the_mean():
    import numpy as np
    diff = np.array([[1.0, 0.0]] * 30 + [[0.0, 0.0]] * 70)
    mean, low, high = paired_intervals(diff, resamples=500)[0]
    assert abs(mean - 0.3) < 1e-9 and low < 0.3 < high
    assert paired_intervals(diff, resamples=500)[1].tolist() == [0.0, 0.0, 0.0]


def _write_run(root, model, precision, dataset, n, header, questions, folder="generator"):
    name = f"{model.replace('/', '_')}_{precision}_{dataset}_n{n}.jsonl"
    target = Path(root) / dataset / folder / model.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    with (target / name).open("w") as handle:
        handle.write(json.dumps({"type": "run_config", "model": model, "precision": precision, "dataset": dataset,
                                 "n_samples": n, "timestamp": "2026-09-20T08:48:15", **header}) + "\n")
        for qid, correct_flags in questions.items():
            samples = [{"response": f"#### {'18' if ok else '5'}", "cutoff": False} for ok in correct_flags]
            handle.write(json.dumps({"qid": qid, "dataset": dataset, "question": qid, "ground_truth": "18",
                                     "raw_samples": samples}) + "\n")


def test_build_web_data_pairs_runs_and_flags_protocol_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        full = {f"q{i}": [True] * 4 for i in range(6)}
        full.update({f"q{i}": [True, True, False, False] for i in range(6, 12)})
        worse = {q: ([True, False, False, False] if i % 2 else flags) for i, (q, flags) in enumerate(full.items())}

        _write_run(root, "org/m", "16bit", "gsm8k", 4, {"max_new_tokens": 512, "total_questions": 12}, full)
        _write_run(root, "org/m", "4bit", "gsm8k", 4, {"max_new_tokens": 512, "total_questions": 12}, worse)
        _write_run(root, "org/m", "8bit", "gsm8k", 4, {"max_new_tokens": 2048, "total_questions": 12}, full)   # other limit
        _write_run(root, "org/m", "4bit", "cqa", 4, {"max_new_tokens_cqa": 384, "total_questions": 12}, full)  # no 16-bit

        data = build_web_data(root)
        runs = {(r["dataset"], r["precision"]): r for r in data["generator"]}

        base = runs[("gsm8k", "16bit")]
        assert base["accuracy"] == 1.0 and base["delta"] is None and base["n_questions"] == 12
        assert abs(base["sample_accuracy"] - 0.75) < 1e-9
        assert base["max_new_tokens"] == 512

        four = runs[("gsm8k", "4bit")]
        assert four["delta"]["n_paired"] == 12
        assert four["delta"]["sample_accuracy"]["mean"] < 0
        assert four["delta"]["sample_accuracy"]["lo"] <= four["delta"]["sample_accuracy"]["mean"] <= four["delta"]["sample_accuracy"]["hi"]

        eight = runs[("gsm8k", "8bit")]
        assert eight["delta"] is None and "token limits differ" in eight["delta_note"]
        assert runs[("cqa", "4bit")]["delta_note"] == "no 16-bit run to compare with"
        assert runs[("cqa", "4bit")]["max_new_tokens"] == 384

        # datasets keep their preferred order and carry labels
        assert [d["id"] for d in data["datasets"]] == ["gsm8k", "cqa"]
        assert data["datasets"][0]["label"] == "GSM8K"
        assert data["models"] == [{"id": "org/m", "label": "m"}]

        out = write_web_data(data, root / "web" / "results.js")
        text = out.read_text()
        assert text.startswith("/* Generated by") and "window.RESULTS = " in text
        assert json.loads(text[text.index("{\n"):text.rindex("}") + 1])["source"]["generator_runs"] == 4


def _write_verifier(root, model, precision, dataset, n, questions):
    """questions: {question: [(solution text, verdict text), ...]}"""
    name = f"verifier_output_{model.replace('/', '_')}_{precision}_{dataset}_n{n}.jsonl"
    target = Path(root) / dataset / "verifier_output" / model.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    with (target / name).open("w") as handle:
        for question, traces in questions.items():
            for index, (solution, verdict) in enumerate(traces):
                handle.write(json.dumps({"question": question, "gt_answer": "18", "trace_index": index,
                                         "student_solution": solution, "verifier_raw_response": verdict}) + "\n")


def _write_generator(root, model, precision, dataset, n, limit, questions):
    """questions: {question: [(solution text, cut off?), ...]}, the traces a verifier run then judges."""
    name = f"{model.replace('/', '_')}_{precision}_{dataset}_n{n}.jsonl"
    target = Path(root) / dataset / "generator" / model.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    with (target / name).open("w") as handle:
        handle.write(json.dumps({"type": "run_config", "model": model, "precision": precision, "dataset": dataset,
                                 "n_samples": n, "max_new_tokens": limit}) + "\n")
        for question, samples in questions.items():
            handle.write(json.dumps({"qid": question, "dataset": dataset, "question": question, "ground_truth": "18",
                                     "raw_samples": [{"response": text, "cutoff": cut} for text, cut in samples]}) + "\n")


def test_standard_questions_keep_run_order_and_leave_short_runs_out():
    full = [0, 0]
    runs = {
        "16bit": {"q3": full, "q1": full, "q2": full, "q4": [0]},     # q4 is missing a trace
        "8bit": {"q1": full, "q2": full, "q3": full, "q5": full},
        "4bit": {"q1": full},
    }
    questions, left_out = standard_questions(runs, n_samples=2, limit=2)
    assert questions == ["q3", "q1"]                 # 16-bit order, not alphabetical
    assert left_out == {"4bit": 1}

    questions, left_out = standard_questions({"16bit": {"q1": full, "q2": full}, "8bit": {"q3": full, "q4": full}}, 2, 2)
    assert questions is None and left_out == {"16bit": 0, "8bit": 0}      # nothing in common


def test_verifier_rows_cover_the_same_questions_and_every_trace():
    ok, bad = "#### 18", "#### 5"
    yes, no = "Final Conclusion: Correct", "Final Conclusion: Incorrect"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # 16-bit: three complete questions; the second trace of q1 was cut off after writing the right number
        _write_generator(root, "org/m", "16bit", "gsm8k", 2, 512,
                         {"q1": [(ok, False), (ok, True)], "q2": [(ok, False), (bad, False)], "q3": [(ok, False), (ok, False)]})
        _write_verifier(root, "org/m", "16bit", "gsm8k", 2,
                        {"q1": [(ok, yes), (ok, yes)], "q2": [(ok, yes), (bad, no)], "q3": [(ok, yes), (ok, yes)]})
        _write_generator(root, "org/m", "4bit", "gsm8k", 2, 512, {"q1": [(ok, False), (bad, False)], "q2": [(bad, False), (bad, False)]})
        _write_verifier(root, "org/m", "4bit", "gsm8k", 2, {"q1": [(ok, yes), (bad, yes)], "q2": [(bad, yes), (bad, no)]})
        _write_verifier(root, "org/m", "8bit", "gsm8k", 2, {"q1": [(ok, yes), (ok, yes)], "q2": [(ok, yes)]})   # q2 incomplete

        rows, left_out = collect_verifier_runs(root, questions=2)
        rows = {r["precision"]: r for r in rows}
        assert set(rows) == {"16bit", "4bit"}
        assert [(r["precision"], r["n_questions"]) for r in left_out] == [("8bit", 1)]

        base, four = rows["16bit"], rows["4bit"]
        assert base["n_questions"] == four["n_questions"] == 2            # q1 and q2: q3 is not in the 4-bit run
        assert base["total_traces"] == four["total_traces"] == 4          # every trace, the cut-off one included
        assert base["truncated"] == 1 and (base["tp"], base["fp"], base["tn"]) == (2, 1, 1)
        assert base["trace_fpr"] == 0.5                                   # the approved cut-off trace is a false positive
        assert base["gen_maj_acc"] == 1.0 and base["max_new_tokens"] == 512
        assert four["trace_fpr"] == 2 / 3 and four["tp"] == 1
        delta = four["delta"]
        assert delta["n_paired"] == 2 and abs(delta["trace_fpr"]["mean"] - (2 / 3 - 0.5)) < 1e-9
        assert base["delta"] is None and four["delta_note"] is None

        data = build_web_data(root)
        assert data["verifier_questions"] == 100 and data["verifier"] == []     # no run reaches the real slice
        assert {r["model"] for r in data["verifier_left_out"]} == {"org/m"}     # file-safe name resolved through the generator header


def test_verifier_rows_are_not_compared_when_cut_off_status_or_token_limit_differs():
    ok, yes = "#### 18", "Final Conclusion: Correct"
    judged = {"q1": [(ok, yes)], "q2": [(ok, yes)]}
    samples = {"q1": [(ok, False)], "q2": [(ok, False)]}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for precision, limit in (("16bit", 512), ("8bit", 2048)):
            _write_generator(root, "org/m", precision, "gsm8k", 1, limit, samples)
        for precision in ("16bit", "8bit", "4bit"):                       # the 4-bit generator run no longer exists
            _write_verifier(root, "org/m", precision, "gsm8k", 1, judged)

        rows = {r["precision"]: r for r in collect_verifier_runs(root, questions=2)[0]}
        assert rows["8bit"]["delta"] is None and rows["8bit"]["delta_note"] == "token limits differ: 512 vs 2048"
        assert rows["4bit"]["truncation_unknown"] == 2 and rows["4bit"]["max_new_tokens"] is None
        assert rows["4bit"]["delta"] is None and "cut-off solutions are unknown" in rows["4bit"]["delta_note"]


def test_vote_against_verifier_compares_scores_at_the_votes_number_of_labels():
    from src.experiment_d.web_export import vote_against_verifier, vote_and_verdict

    q1 = [_trace("18", True, True), _trace("18", True, False), _trace("5", False, True), _trace("18", True, True, truncated=True)]
    q2 = [_trace("7", False, True)] * 4
    q3 = [_trace("9", True, False)] * 3 + [_trace("9", True, None)]
    q4 = [_trace("3", False, True, truncated=True)] * 4
    # the cut-off "18" never votes but stays in the denominator; only verdicts on finished majority traces count
    assert vote_and_verdict(q1) == (0.5, True, 0.5)
    assert vote_and_verdict(q2) == (1.0, False, 1.0)
    assert vote_and_verdict(q3) == (1.0, True, 0.0)
    assert vote_and_verdict(q4) == (0.0, False, None)

    out = vote_against_verifier([q1, q2, q3, q4], threshold=0.75)
    assert out["labels_k"] == 2 and out["questions_without_verdict"] == 1
    assert out["vote_label_accuracy"] == 0.5          # q2 and q3
    assert out["verifier_label_accuracy"] == 0.5      # q2 (1.0) and q1 (0.5, ahead of q4's 0.5 by order)
    assert abs(out["vote_auroc"] - 0.625) < 1e-9


def _write_round_run(root, experiment, k, dataset, total, records, adapter=None):
    """records: {qid: (reference answer, [answer per sample, None = cut off])}"""
    from src.experiment_d.run_files import self_training_round_dir

    folder = self_training_round_dir(root, experiment, k) / dataset / "generator" / "org_m"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"org_m_4bit_{dataset}_n8.jsonl").open("w") as handle:
        handle.write(json.dumps({"type": "run_config", "model": "org/m", "precision": "4bit", "dataset": dataset, "n_samples": 8,
                                 "total_questions": total, "max_new_tokens": 512, "gpu": "Tesla T4", "batch_size": 128,
                                 "adapter_sha256": adapter}) + "\n")
        for qid, (truth, answers) in records.items():
            samples = [{"response": "unfinished" if a is None else f"Working {i}.\n#### {a}", "cutoff": a is None,
                        "length": 512 if a is None else 100} for i, a in enumerate(answers)]
            handle.write(json.dumps({"qid": qid, "dataset": dataset, "question": qid, "ground_truth": truth,
                                     "raw_samples": samples, "performance": {"effective_batch_size": 128}}) + "\n")


def test_self_training_experiment_rounds_labels_and_progress():
    from src.experiment_d.run_files import adapter_dir
    from src.experiment_d.web_export import score_self_training_experiment

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        right, wrong, cut = "18", "5", None
        _write_round_run(root, "exp", 0, "gsm8k", 4, {
            "t0": ("18", [right] * 8), "t1": ("18", [right] * 4 + [cut] * 4),
            "t2": ("18", [wrong] * 8), "t3": ("18", [cut] * 8)})
        _write_round_run(root, "exp", 1, "gsm8k", 4, {
            "t0": ("18", [right] * 8), "t1": ("18", [right] * 8),
            "t2": ("18", [wrong] * 8), "t3": ("18", [right] * 4 + [cut] * 4)}, adapter="abc")
        _write_round_run(root, "exp", 0, "gsm8k_train", 3, {
            "gsm8k_train_1": (None, ["10"] * 8),                 # labelled, right
            "gsm8k_train_2": (None, ["3"] * 6 + ["4"] * 2),      # labelled, wrong
            "gsm8k_train_3": (None, ["7"] * 5 + ["8"] * 3)})     # 5 of 8: not labelled at 0.75
        _write_round_run(root, "exp", 1, "gsm8k_train", 3, {"gsm8k_train_1": (None, ["10"] * 8)})   # still running
        log_dir = adapter_dir(root, "exp", 1, "org/m", "4bit")
        log_dir.mkdir(parents=True)
        (log_dir / "training_log.json").write_text(json.dumps({
            "settings": {"objective": "sft", "threshold": 0.75, "train_steps": 200, "learning_rate": 1e-4},
            "trained_solutions": 13, "losses": [0.2] * 10 + [0.1] * 20 + [0.05] * 10}))

        spec = {"id": "exp", "part": "main", "objective": "sft", "precision": "4bit", "label": "SFT", "model": "org/m"}
        entry, inner = score_self_training_experiment(root, spec, {"gsm8k_train_1": "10", "gsm8k_train_2": "4", "gsm8k_train_3": "7"})

        assert entry["status"] == "in progress"
        assert entry["in_progress"] == [{"stage": "round 1 pool", "done": 1, "total": 3}]
        r0, r1 = entry["test"]
        assert r0["change"] is None and r0["unfinished_questions"] == 1
        assert abs(r0["single"] - 0.375) < 1e-9 and abs(r1["single"] - 0.625) < 1e-9
        assert r0["voted"] == 0.5 and r1["voted"] == 0.75
        assert r1["gpu"] == "Tesla T4" and r1["batches"] == [128] and r1["adapter"] is True
        change = r1["change"]
        assert abs(change["single"]["mean"] - 0.25) < 1e-9 and abs(change["cutoff"]["mean"] + 0.25) < 1e-9
        assert (change["voted_lost"], change["voted_gained"]) == (0, 1)
        same = change["finished_accuracy_same_questions"]
        assert same["n"] == 3 and same["mean"] == 0.0        # t3 never finished in round 0, so it is left out

        (labels,) = entry["labels"]
        assert labels["round"] == 0 and labels["labelled"] == 2 and labels["label_accuracy"] == 0.5 and labels["wrong_labels"] == 1
        assert labels["training"]["trained"] == 13 and labels["training"]["loss_first"] == 0.2 and labels["training"]["loss_last"] == 0.05
        assert set(inner["tests"]) == {0, 1}

        missing, none = score_self_training_experiment(root, {**spec, "id": "absent"}, {})
        assert missing["status"] == "not synced" and none is None


if __name__ == "__main__":
    test_question_counts_and_metrics()
    test_question_counts_can_keep_truncated_traces()
    test_verifier_guided_majority_falls_back_to_generator_majority()
    test_calibration_curve_bins_and_min_count()
    test_token_limit_key_spellings()
    test_paired_intervals_contain_the_mean()
    test_build_web_data_pairs_runs_and_flags_protocol_changes()
    test_standard_questions_keep_run_order_and_leave_short_runs_out()
    test_verifier_rows_cover_the_same_questions_and_every_trace()
    test_verifier_rows_are_not_compared_when_cut_off_status_or_token_limit_differs()
    test_vote_against_verifier_compares_scores_at_the_votes_number_of_labels()
    test_self_training_experiment_rounds_labels_and_progress()
    print("All web export tests passed!")
