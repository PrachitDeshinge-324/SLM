"""
Everything the project web pages show, computed from the run files in results/.

Nothing is read from stored metrics: generator runs are re-scored from `raw_samples` and verifier
runs are relabelled from the stored solutions (see rescoring.py), so the pages always reflect the
current extractor, matcher and vote rule. The result is one plain dict; scripts/export_web_data.py
writes it to web/assets/data/results.js, which the Phase 1 and Phase 2 pages draw their tables from.

Comparisons follow the report scripts: a quantized run is compared with the 16-bit run of the same
model, dataset and sample count on the questions both contain, and only when both used the same
token limit. Intervals are 95% paired bootstrap percentiles over questions.

The verifier page differs from the verifier reports in one way: every row is scored on the same
VERIFIER_QUESTIONS questions and on all N traces of each, so every row holds the same number of
traces. Truncated traces are therefore kept, as wrong traces, instead of dropped.
"""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.experiment_d.answer_matching import answers_match
from src.experiment_d.benchmarks import GSM8K_TRAIN_PARQUET, load_gsm8k_train_answers
from src.experiment_d.evaluation_stats import auroc
from src.experiment_d.rescoring import load_verifier_traces, rescore_record
from src.experiment_d.run_files import (
    SELF_TRAINING_DIR, adapter_dir, generator_path_for_verifier_output, generator_run_path, is_legacy_file,
    parse_run_name, read_run_config, safe_name, self_training_round_dir, token_limit,
)
from src.experiment_d.self_training import select_preference_pairs, select_pseudo_labels
from src.experiment_d.verifier_metrics import COUNT_FIELDS, metrics_from_totals, question_counts

PRECISIONS = ("16bit", "8bit", "4bit")
BASELINE = "16bit"

DATASET_LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500", "ChilleD_SVAMP": "SVAMP", "cqa": "CommonsenseQA"}

# Questions behind every verifier row: 100 x 8 samples = 800 traces. Runs with fewer are left out.
VERIFIER_QUESTIONS = 100

RESAMPLES = 2000
CALIBRATION_BINS = 10
CALIBRATION_MIN_COUNT = 5

# The vote rule of the self-training loop: a question is labelled when at least this share of ALL its
# samples give the majority answer (6 of 8). Cut-off samples never vote but stay in the denominator.
LABEL_THRESHOLD = 0.75

GENERATOR_DELTAS = (
    ("accuracy", "correctness"),
    ("sample_accuracy", "sample_accuracy"),
    ("pass_at_k", "oracle_correctness"),
    ("label_share", "labelled"),
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Verifier metrics that get a paired interval against the 16-bit run: (output key, metrics_from_totals key).
VERIFIER_DELTAS = (
    ("gen_maj_acc", "Gen_Maj_Acc"),
    ("ver_maj_acc", "Ver_Maj_Acc"),
    ("trace_fpr", "Trace_FPR"),
    ("mcc", "MCC"),
    ("balanced_accuracy", "Balanced_Accuracy"),
)

VERIFIER_FIELDS = (
    ("gen_maj_acc", "Gen_Maj_Acc"),
    ("ver_maj_acc", "Ver_Maj_Acc"),
    ("ver_minus_gen", "Delta_Ver_Gen"),
    ("trace_accuracy", "Trace_Accuracy"),
    ("trace_precision", "Trace_Precision"),
    ("trace_recall", "Trace_Recall"),
    ("trace_specificity", "Trace_Specificity"),
    ("trace_fpr", "Trace_FPR"),
    ("balanced_accuracy", "Balanced_Accuracy"),
    ("mcc", "MCC"),
    ("approval_rate", "Approval_Rate"),
    ("parse_coverage", "Parse_Coverage"),
    ("correct_prevalence", "Correct_Trace_Prevalence"),
    ("total_traces", "Total_Traces"),
    ("truncation_unknown", "Truncation_Unknown"),
    ("unparseable", "Unparseable_Traces"),
    ("tp", "TP"),
    ("fp", "FP"),
    ("fn", "FN"),
    ("tn", "TN"),
)


# ---------------------------------------------------------------- helpers

def model_label(model_id: str) -> str:
    """'Qwen/Qwen3.5-0.8B' -> 'Qwen3.5-0.8B'."""
    return model_id.split("/")[-1]


def paired_intervals(per_question: np.ndarray, resamples: int = RESAMPLES, seed: int = 0):
    """
    Mean and 95% percentile interval of each column of an (n_questions, k) array, resampling
    questions with replacement. Returns an array of shape (k, 3): mean, low, high.
    """
    n, k = per_question.shape
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, k))
    for i in range(resamples):
        means[i] = per_question[rng.integers(0, n, n)].mean(axis=0)
    low, high = np.percentile(means, [2.5, 97.5], axis=0)
    return np.stack([per_question.mean(axis=0), low, high], axis=1)


def _interval(row) -> dict:
    return {"mean": float(row[0]), "lo": float(row[1]), "hi": float(row[2])}


def calibration_curve(confidences, correct, bins: int = CALIBRATION_BINS, min_count: int = CALIBRATION_MIN_COUNT) -> list:
    """
    Accuracy of the majority answer by agreement bin, as in the generator report: right-closed bins
    (0, 0.1], (0.1, 0.2] ... with 0 in the first, and bins with fewer than `min_count` questions dropped.
    """
    confidences, correct = np.asarray(confidences, float), np.asarray(correct, float)
    edges = np.arange(1, bins + 1) / bins
    index = np.clip(np.searchsorted(edges, confidences, side="left"), 0, bins - 1)
    curve = []
    for b in range(bins):
        mask = index == b
        if mask.sum() >= min_count:
            curve.append({"x": float(edges[b]), "accuracy": float(correct[mask].mean()), "count": int(mask.sum())})
    return curve


# ---------------------------------------------------------------- generator runs

def read_generator_run(path) -> tuple:
    """(run_config, per-question records) of one generator file; unreadable lines are skipped."""
    run_config, records = {}, []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "run_config":
                run_config = obj
            elif "raw_samples" in obj:
                records.append(obj)
    return run_config, records


def score_generator_run(path):
    """
    One generator run as {summary fields, 'per_question'} or None if it holds no records.
    `per_question` maps qid to the three paired metrics and is not written to the output.
    """
    run_config, records = read_generator_run(path)
    if not records:
        return None
    name = parse_run_name(Path(path).name) or {}
    dataset_id = run_config.get("dataset") or name.get("dataset", "")
    model = run_config.get("model") or name.get("model", "")
    precision = run_config.get("precision") or name.get("precision", "")
    n_samples = int(run_config.get("n_samples") or name.get("n_samples") or 0)

    per_question, confidences, correctness, labelled_right = {}, [], [], []
    sums = defaultdict(float)
    for record in records:
        metrics = rescore_record(record, dataset_id)
        metrics["labelled"] = float(metrics["confidence_all"] >= LABEL_THRESHOLD - 1e-9)
        if metrics["labelled"]:
            labelled_right.append(float(metrics["correctness"]))
        key = record.get("qid", record["question"])
        per_question[key] = [float(metrics[source]) for _, source in GENERATOR_DELTAS]
        confidences.append(metrics["confidence"])
        correctness.append(float(metrics["correctness"]))
        for field in ("confidence", "confidence_all", "diversity", "cutoff_rate", "failure_rate"):
            sums[field] += float(metrics[field])

    n = len(per_question)
    return {
        "dataset": safe_name(dataset_id),
        "model": model,
        "precision": precision,
        "n_samples": n_samples,
        "n_questions": n,
        "expected_questions": run_config.get("total_questions"),
        "max_new_tokens": token_limit(run_config, dataset_id),
        "run_date": (run_config.get("timestamp") or "")[:10] or None,
        "file": Path(path).name,
        "accuracy": float(np.mean([v[0] for v in per_question.values()])),
        "sample_accuracy": float(np.mean([v[1] for v in per_question.values()])),
        "pass_at_k": float(np.mean([v[2] for v in per_question.values()])),
        "label_share": len(labelled_right) / n,
        "label_accuracy": float(np.mean(labelled_right)) if labelled_right else None,
        "n_labelled": len(labelled_right),
        "confidence": sums["confidence"] / len(records),
        "confidence_all": sums["confidence_all"] / len(records),
        "diversity": sums["diversity"] / len(records),
        "cutoff_rate": sums["cutoff_rate"] / len(records),
        "failure_rate": sums["failure_rate"] / len(records),
        "calibration": calibration_curve(confidences, correctness),
        "per_question": per_question,
        "_timestamp": run_config.get("timestamp"),
    }


def generator_deltas(runs: list) -> None:
    """Fill in `delta` and `delta_note` on every run, against the 16-bit run of the same model, dataset and N."""
    baselines = {(r["dataset"], r["model"], r["n_samples"]): r for r in runs if r["precision"] == BASELINE}
    for run in runs:
        run["delta"], run["delta_note"] = None, None
        if run["precision"] == BASELINE:
            continue
        base = baselines.get((run["dataset"], run["model"], run["n_samples"]))
        if base is None:
            run["delta_note"] = "no 16-bit run to compare with"
            continue
        limits = {run["max_new_tokens"], base["max_new_tokens"]} - {None}
        if len(limits) > 1:
            run["delta_note"] = "token limits differ: " + " vs ".join(str(x) for x in (base["max_new_tokens"], run["max_new_tokens"]))
            continue
        common = sorted(set(run["per_question"]) & set(base["per_question"]))
        if not common:
            run["delta_note"] = "no questions in common with the 16-bit run"
            continue
        diff = np.array([np.subtract(run["per_question"][q], base["per_question"][q]) for q in common])
        intervals = paired_intervals(diff)
        run["delta"] = {"n_paired": len(common)}
        for (name, _), row in zip(GENERATOR_DELTAS, intervals):
            run["delta"][name] = _interval(row)


def collect_generator_runs(results_dir) -> list:
    files = sorted(Path(results_dir).glob("*/generator/*/*.jsonl"))
    runs = []
    for path in files:
        if is_legacy_file(path) or parse_run_name(path.name) is None:
            continue
        run = score_generator_run(path)
        if run is not None:
            runs.append(run)
    generator_deltas(runs)
    return runs


# ---------------------------------------------------------------- verifier runs

def standard_questions(traces_by_precision: dict, n_samples: int, limit: int = VERIFIER_QUESTIONS) -> tuple:
    """
    (questions, left_out) for the verifier runs of one model, dataset and N.

    A question is complete in a run that holds all N of its traces. `left_out` maps each precision
    with fewer than `limit` complete questions to how many it has. `questions` are the first `limit`,
    in the order of the highest-precision run kept, that are complete in every kept run; None when
    no run is kept or the kept runs share fewer than that (they are then all in `left_out`).
    """
    complete = {
        precision: [question for question, traces in by_question.items() if len(traces) == n_samples]
        for precision, by_question in traces_by_precision.items()
    }
    left_out = {precision: len(found) for precision, found in complete.items() if len(found) < limit}
    kept = [precision for precision in PRECISIONS if precision in complete and precision not in left_out]
    if not kept:
        return None, left_out
    shared = set.intersection(*[set(complete[precision]) for precision in kept])
    questions = [question for question in complete[kept[0]] if question in shared][:limit]
    if len(questions) < limit:
        left_out.update({precision: len(questions) for precision in kept})
        return None, left_out
    return questions, left_out


def vote_and_verdict(traces: list) -> tuple:
    """
    (agreement, majority answer right, verifier score) of one question, from its relabelled traces.

    Agreement is the majority count over ALL traces; cut-off traces never vote. The verifier score is
    the approval rate among the finished traces that carry the majority answer, None without verdicts.
    """
    finished = [t for t in traces if t["answer"] is not None and t["truncated"] is not True]
    counts = Counter(t["answer"] for t in finished)
    if not counts:
        return 0.0, False, None
    majority, votes = counts.most_common(1)[0]
    right = any(t["correct"] for t in finished if t["answer"] == majority)
    verdicts = [bool(t["approved"]) for t in finished if t["answer"] == majority and t["approved"] is not None]
    return votes / len(traces), right, (sum(verdicts) / len(verdicts) if verdicts else None)


def vote_against_verifier(questions: list, threshold: float = LABEL_THRESHOLD) -> dict:
    """
    How well three label-free scores pick out questions whose majority answer is right: the vote's
    agreement, the verifier's approval of the majority answer, and their product.

    `questions` holds the relabelled traces of each question. For each score: AUROC, and the accuracy
    of the k labels it ranks highest, with k = the number of questions the vote labels (agreement >=
    threshold), so all three are compared at the same number of labels. A question without verdicts
    gets a verifier score of 0.5 (no information).
    """
    rows = [vote_and_verdict(traces) for traces in questions]
    agreement = np.array([r[0] for r in rows])
    right = np.array([r[1] for r in rows], dtype=bool)
    verdict = np.array([0.5 if r[2] is None else r[2] for r in rows])
    k = int((agreement >= threshold - 1e-9).sum())
    out = {"labels_k": k, "questions_without_verdict": sum(r[2] is None for r in rows)}
    for name, score in (("vote", agreement), ("verifier", verdict), ("both", agreement * verdict)):
        out[f"{name}_auroc"] = auroc(score, right)
        top = np.argsort(-score, kind="stable")[:k]
        out[f"{name}_label_accuracy"] = float(right[top].mean()) if k else None
    return out


def verifier_pairing_problem(base, row):
    """Why a verifier row cannot be compared with its 16-bit row, or None when it can."""
    if base is None:
        return "no complete 16-bit run to compare with"
    if base["truncation_unknown"]:
        return "no comparison: the 16-bit run's cut-off solutions are unknown"
    if row["truncation_unknown"]:
        return "no comparison: this run's cut-off solutions are unknown"
    limits = {row["max_new_tokens"], base["max_new_tokens"]} - {None}
    if len(limits) > 1:
        return "token limits differ: " + " vs ".join(str(x) for x in (base["max_new_tokens"], row["max_new_tokens"]))
    return None


def collect_verifier_runs(results_dir, questions: int = VERIFIER_QUESTIONS) -> tuple:
    """
    (rows, left_out). One row per (dataset, model, N, precision), each scored on the same `questions`
    questions of its model (see standard_questions) and on all N traces of each, truncated ones
    included as wrong traces. `left_out` lists the runs that hold too few complete questions.
    """
    groups = defaultdict(dict)
    for path in sorted(Path(results_dir).glob("*/verifier_output/*/verifier_output_*.jsonl")):
        run = parse_run_name(path.name)
        if is_legacy_file(path) or run is None:
            continue
        groups[(run["dataset"], run["model"], run["n_samples"])][run["precision"]] = path

    def metrics_of(array, n):
        totals = dict(zip(COUNT_FIELDS, array.sum(axis=0)))
        return metrics_from_totals({k: int(v) for k, v in totals.items()}, n)

    rows, left_out = [], []
    for (dataset, model, n_samples), files in groups.items():
        traces = {precision: load_verifier_traces(path, dataset) for precision, path in files.items()}
        selected, short = standard_questions(traces, n_samples, questions)
        for precision in sorted(short, key=PRECISIONS.index):
            left_out.append({"dataset": dataset, "model": model, "n_samples": n_samples, "precision": precision,
                             "file": files[precision].name, "n_questions": short[precision]})
        if selected is None:
            continue

        counts, group_rows = {}, {}
        for precision in sorted(set(files) - set(short), key=PRECISIONS.index):
            counts[precision] = np.array(
                [[question_counts(traces[precision][q], keep_truncated=True)[f] for f in COUNT_FIELDS] for q in selected],
                dtype=float,
            )
            metrics = metrics_of(counts[precision], len(selected))
            row = {
                "dataset": dataset,
                "model": model,
                "n_samples": n_samples,
                "precision": precision,
                "n_questions": len(selected),
                "file": files[precision].name,
                "truncated": int(counts[precision][:, COUNT_FIELDS.index("truncated")].sum()),
                "max_new_tokens": None,
                "delta": None,
                "delta_note": None,
            }
            for out_key, key in VERIFIER_FIELDS:
                row[out_key] = metrics[key]
            row["vote_vs_verifier"] = vote_against_verifier([traces[precision][q] for q in selected])
            if not row["truncation_unknown"]:     # the traces are the ones in the generator file, so its limit applies
                row["max_new_tokens"] = token_limit(read_run_config(generator_path_for_verifier_output(files[precision])), dataset)

            if precision != BASELINE:
                row["delta_note"] = verifier_pairing_problem(group_rows.get(BASELINE), row)
                if row["delta_note"] is None:
                    row["delta"] = verifier_delta(counts[BASELINE], counts[precision], metrics_of)
            group_rows[precision] = row
            rows.append(row)
    return rows, left_out


def verifier_delta(base: np.ndarray, other: np.ndarray, metrics_of, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """Change in each VERIFIER_DELTAS metric against the baseline, paired over questions, with 95% interval."""
    n = len(base)

    def vector(idx):
        a, b = metrics_of(base[idx], len(idx)), metrics_of(other[idx], len(idx))
        return np.array([b[key] - a[key] for _, key in VERIFIER_DELTAS])

    point = vector(np.arange(n))
    rng = np.random.default_rng(seed)
    draws = np.array([vector(rng.integers(0, n, n)) for _ in range(resamples)])
    low, high = np.percentile(draws, [2.5, 97.5], axis=0)
    delta = {"n_paired": n}
    for i, (name, _) in enumerate(VERIFIER_DELTAS):
        delta[name] = _interval((point[i], low[i], high[i]))
    return delta


# ---------------------------------------------------------------- self-training runs

SELF_TRAINING_MODEL = "Qwen/Qwen3.5-0.8B"

# The experiments the Phase 3 page shows, in page order. Any not found under results/self_training/
# are listed as not synced yet. `control`: an untrained run (a results root, relative to the results
# directory) that replaces round 0 as the reference, for an experiment whose round 0 ran at another
# batch size than its trained rounds.
SELF_TRAINING_EXPERIMENTS = (
    {"id": "q08_sft_16bit_p500", "part": "main", "objective": "sft", "precision": "16bit", "label": "SFT, 16-bit",
     "note": "500 unlabelled train questions, 200 test questions, batch 128 on a T4."},
    {"id": "q08_sft_4bit_p500", "part": "main", "objective": "sft", "precision": "4bit", "label": "SFT, 4-bit",
     "note": "500 unlabelled train questions, 200 test questions, batch 128 on a T4."},
    {"id": "qwen08_16bit_pilot", "part": "pilot", "objective": "sft", "precision": "16bit", "label": "SFT, 16-bit (pilot)",
     "note": "200 unlabelled train questions, 100 test questions, batch 256 on an L4."},
    {"id": "qwen08_4bit_pilot", "part": "pilot", "objective": "sft", "precision": "4bit", "label": "SFT, 4-bit (pilot)",
     "control": "self_training/qwen08_4bit_batch_check/b256",
     "note": "200 unlabelled train questions, 100 test questions on an L4. Round 0 ran at batch 64 and the trained "
             "rounds at batch 256, so every round is compared with an untrained run at batch 256."},
    {"id": "qwen08_4bit_dpo_similar1", "part": "dpo", "objective": "dpo", "precision": "4bit", "label": "DPO, 4-bit, 512 tokens",
     "note": "One round of length-normalised DPO on (majority, minority) pairs of similar length; 200 unlabelled "
             "train questions, 100 test questions, batch 128 on a T4."},
    {"id": "qwen08_4bit_dpo_similar1_t1024", "part": "dpo", "objective": "dpo", "precision": "4bit",
     "label": "DPO, 4-bit, 1,024 tokens",
     "note": "The DPO test repeated with a 1,024-token limit, against an untrained run at the same limit."},
)

# 16-bit and 4-bit arms trained the same way, compared round by round.
SELF_TRAINING_COMPARISONS = (
    {"label": "Main run", "full": "q08_sft_16bit_p500", "quantized": "q08_sft_4bit_p500"},
    {"label": "Pilot", "full": "qwen08_16bit_pilot", "quantized": "qwen08_4bit_pilot"},
)

# Untrained 4-bit runs of the same model on the same test questions: how far the numbers move with no training.
SELF_TRAINING_CONTROLS = (
    {"label": "Stored full-test run, older session", "root": ""},
    {"label": "4-bit pilot, round 0", "root": "self_training/qwen08_4bit_pilot/round0"},
    {"label": "Check run", "root": "self_training/qwen08_4bit_batch_check/b128"},
    {"label": "Check run", "root": "self_training/qwen08_4bit_batch_check/b256"},
    {"label": "Check run, 1,024 tokens", "root": "self_training/qwen08_4bit_tokens1024/untrained"},
)

TEST_DATASET, POOL_DATASET, ST_SAMPLES = "gsm8k", "gsm8k_train", 8


def test_run_scores(path) -> tuple:
    """(header, qid -> per-question numbers) of one test run, re-scored from its raw samples."""
    header, records = read_generator_run(path)
    scores = {}
    for record in records:
        metrics = rescore_record(record, header.get("dataset") or TEST_DATASET)
        n = len(record["raw_samples"])
        scores[record.get("qid", record["question"])] = {
            "single": metrics["sample_accuracy"],
            "cutoff": metrics["cutoff_rate"],
            "voted": float(bool(metrics["correctness"])),
            "right": metrics["sample_accuracy"] * n,
            "finished": (1 - metrics["cutoff_rate"]) * n,
            "confident": float(metrics["confidence_all"] >= LABEL_THRESHOLD - 1e-9),
            "distinct": metrics["diversity"],
            "tokens": float(np.mean([sample.get("length", 0) for sample in record["raw_samples"]])),
            "batch": (record.get("performance") or {}).get("effective_batch_size") or header.get("batch_size"),
        }
    return header, scores


def _shared(*score_sets) -> list:
    return sorted(set.intersection(*[set(s) for s in score_sets])) if score_sets else []


def _column(scores: dict, qids: list, key: str) -> np.ndarray:
    return np.array([scores[q][key] for q in qids], float)


def test_summary(scores: dict, qids: list) -> dict:
    """Mean of each per-question number over `qids`; accuracy among finished answers is pooled."""
    finished_per_question = _column(scores, qids, "finished")
    finished = finished_per_question.sum()
    return {
        "unfinished_questions": int((finished_per_question == 0).sum()),
        "single": _column(scores, qids, "single").mean(),
        "cutoff": _column(scores, qids, "cutoff").mean(),
        "finished_accuracy": _column(scores, qids, "right").sum() / finished if finished else None,
        "voted": _column(scores, qids, "voted").mean(),
        "confident": _column(scores, qids, "confident").mean(),
        "distinct": _column(scores, qids, "distinct").mean(),
        "tokens": _column(scores, qids, "tokens").mean(),
    }


def test_change(after: dict, before: dict, qids: list, resamples: int = RESAMPLES, seed: int = 0) -> dict:
    """
    `after` minus `before` on the same questions, each with a 95% paired bootstrap interval. Accuracy
    among finished answers is a pooled ratio, so it is recomputed on every resample of the questions.
    """
    columns = ("single", "cutoff", "voted")
    diff = np.array([[after[q][c] - before[q][c] for c in columns] for q in qids])
    change = {"n_paired": len(qids)}
    for name, row in zip(columns, paired_intervals(diff, resamples, seed)):
        change[name] = _interval(row)

    right_a, finished_a = _column(after, qids, "right"), _column(after, qids, "finished")
    right_b, finished_b = _column(before, qids, "right"), _column(before, qids, "finished")

    def ratio_change(index):
        da, db = finished_a[index].sum(), finished_b[index].sum()
        return right_a[index].sum() / da - right_b[index].sum() / db if da and db else np.nan

    rng = np.random.default_rng(seed)
    draws = [ratio_change(rng.integers(0, len(qids), len(qids))) for _ in range(resamples)]
    change["finished_accuracy"] = _interval((ratio_change(np.arange(len(qids))), *np.nanpercentile(draws, [2.5, 97.5])))

    # The pooled ratio shifts when questions drop out of it (every sample cut off), so also compare it
    # within each question, on the questions where both runs finished at least one sample.
    both = [q for q in qids if after[q]["finished"] > 0 and before[q]["finished"] > 0]
    if both:
        within = np.array([[after[q]["right"] / after[q]["finished"] - before[q]["right"] / before[q]["finished"]] for q in both])
        change["finished_accuracy_same_questions"] = {**_interval(paired_intervals(within, resamples, seed)[0]), "n": len(both)}
    else:
        change["finished_accuracy_same_questions"] = None

    voted_a, voted_b = _column(after, qids, "voted"), _column(before, qids, "voted")
    change["voted_lost"] = int(((voted_b == 1) & (voted_a == 0)).sum())
    change["voted_gained"] = int(((voted_b == 0) & (voted_a == 1)).sum())
    return change


def _run_facts(header: dict, scores: dict) -> dict:
    """Where and how a test run was made: the facts two compared runs must share."""
    return {
        "gpu": header.get("gpu"),
        "max_new_tokens": token_limit(header, TEST_DATASET),
        "batches": sorted({s["batch"] for s in scores.values() if s["batch"] is not None}),
        "adapter": bool(header.get("adapter_sha256")),
    }


def _round_indices(experiment_dir: Path) -> list:
    return sorted(int(p.name[len("round"):]) for p in experiment_dir.glob("round*")
                  if p.is_dir() and p.name[len("round"):].isdigit())


def _training_facts(log: dict) -> dict:
    settings = log.get("settings", {})
    losses = log.get("losses") or []
    window = 10 if len(losses) >= 20 else 1     # a short run (DPO) has too few steps to average over
    return {
        "objective": settings.get("objective", "sft"),
        "trained": log.get("trained_solutions", log.get("selected_pairs")),
        "steps": log.get("train_steps", settings.get("train_steps")),
        "learning_rate": settings.get("learning_rate"),
        "loss_first": float(np.mean(losses[:window])) if losses else None,
        "loss_last": float(np.mean(losses[-window:])) if losses else None,
    }


def pool_labels(records: list, objective: str, threshold: float, pairing: str, train_answers: dict) -> dict:
    """
    The labels one round's pool answers give the next round, re-selected with the current code, and how
    many were right. Answers are read here only to score the labels after the run, never to choose them.
    """
    if objective == "dpo":
        items = select_preference_pairs(records, POOL_DATASET, threshold, pairing)
        labels = {item["qid"]: item["label"] for item in items}
        rejected = [answers_match(item["rejected_answer"], train_answers[item["qid"]]) for item in items if item["qid"] in train_answers]
    else:
        items = select_pseudo_labels(records, POOL_DATASET, threshold)
        labels = {item["qid"]: item["label"] for item in items}
        rejected = []
    right = [answers_match(label, train_answers[qid]) for qid, label in labels.items() if qid in train_answers]
    return {
        "labelled": len(labels),
        "label_accuracy": float(np.mean(right)) if right else None,
        "wrong_labels": len(right) - int(sum(right)),
        "items": len(items),
        "rejected_right": float(np.mean(rejected)) if rejected else None,
    }


def score_self_training_experiment(results_dir, spec: dict, train_answers: dict) -> tuple:
    """(page entry, internals for the cross-precision comparison) of one self-training experiment."""
    results_dir = Path(results_dir)
    model, precision = spec.get("model", SELF_TRAINING_MODEL), spec["precision"]
    entry = {key: spec.get(key) for key in ("id", "part", "label", "note", "precision", "objective")}
    entry["model"] = model
    experiment_dir = results_dir / SELF_TRAINING_DIR / spec["id"]
    if not experiment_dir.is_dir():
        entry["status"] = "not synced"
        return entry, None

    rounds = _round_indices(experiment_dir)
    tests, facts, in_progress = {}, {}, []
    for k in rounds:
        path = generator_run_path(self_training_round_dir(results_dir, spec["id"], k), model, precision, TEST_DATASET, ST_SAMPLES)
        if not path.exists():
            continue
        header, scores = test_run_scores(path)
        total = header.get("total_questions")
        if total and len(scores) < total:
            in_progress.append({"stage": f"round {k} test", "done": len(scores), "total": total})
            continue
        tests[k], facts[k] = scores, _run_facts(header, scores)

    reference_label, reference, reference_facts = "round 0", tests.get(0), facts.get(0)
    if spec.get("control"):
        path = generator_run_path(results_dir / spec["control"], model, precision, TEST_DATASET, ST_SAMPLES)
        if path.exists():
            header, reference = test_run_scores(path)
            reference_label, reference_facts = "untrained control", _run_facts(header, reference)

    rows = []
    if reference is not None:
        qids = _shared(reference, *tests.values())
        entry["test_questions"] = len(qids)
        if spec.get("control"):
            rows.append({"round": None, "name": "untrained control", **reference_facts, **test_summary(reference, qids), "change": None})
        for k in sorted(tests):
            is_reference = k == 0 and not spec.get("control")
            rows.append({"round": k, "name": f"round {k}", **facts[k], **test_summary(tests[k], qids),
                         "change": None if is_reference else test_change(tests[k], reference, qids)})
    entry["reference"] = reference_label
    entry["test"] = rows

    labels = []
    for k in rounds:
        path = generator_run_path(self_training_round_dir(results_dir, spec["id"], k), model, precision, POOL_DATASET, ST_SAMPLES)
        if not path.exists():
            continue
        header, records = read_generator_run(path)
        total = header.get("total_questions")
        if total and len(records) < total:
            in_progress.append({"stage": f"round {k} pool", "done": len(records), "total": total})
            continue
        log_file = adapter_dir(results_dir, spec["id"], k + 1, model, precision) / "training_log.json"
        log = json.loads(log_file.read_text()) if log_file.exists() else {}
        settings = log.get("settings", {})
        objective = settings.get("objective", spec.get("objective") or "sft")
        threshold = settings.get("threshold", 0.625 if objective == "dpo" else LABEL_THRESHOLD)
        row = {"round": k, "pool_questions": len(records), "objective": objective, "threshold": threshold,
               **pool_labels(records, objective, threshold, settings.get("pairing", "similar_length"), train_answers),
               "training": _training_facts(log) if log else None}
        labels.append(row)
    entry["labels"] = labels
    entry["in_progress"] = in_progress
    entry["status"] = "in progress" if in_progress else "synced"
    return entry, {"tests": tests, "reference": reference}


def compare_arms(full: dict, quantized: dict) -> dict:
    """
    16-bit against 4-bit, round by round, each against its own untrained reference on shared questions:
    the gap before training, the difference of the two changes, and the gap left to the untrained 16-bit model.
    """
    def per_question(scores, qids, key):
        return _column(scores, qids, key)

    out = {"before": None, "rounds": []}
    if not full or not quantized or full["reference"] is None or quantized["reference"] is None:
        return out
    full_ref, quant_ref = full["reference"], quantized["reference"]
    qids = _shared(full_ref, quant_ref)
    if qids:
        gap = np.stack([per_question(full_ref, qids, k) - per_question(quant_ref, qids, k) for k in ("single", "voted")], axis=1)
        intervals = paired_intervals(gap)
        out["before"] = {"n_paired": len(qids), "single": _interval(intervals[0]), "voted": _interval(intervals[1])}
    for k in sorted((set(full["tests"]) & set(quantized["tests"])) - {0}):
        qids = _shared(full_ref, quant_ref, full["tests"][k], quantized["tests"][k])
        if not qids:
            continue
        columns = []
        for key in ("single", "voted"):
            full_change = per_question(full["tests"][k], qids, key) - per_question(full_ref, qids, key)
            quant_change = per_question(quantized["tests"][k], qids, key) - per_question(quant_ref, qids, key)
            columns.append(full_change - quant_change)
        for key in ("single", "voted"):
            columns.append(per_question(full_ref, qids, key) - per_question(quantized["tests"][k], qids, key))
        intervals = paired_intervals(np.stack(columns, axis=1))
        out["rounds"].append({
            "round": k, "n_paired": len(qids),
            "difference_single": _interval(intervals[0]), "difference_voted": _interval(intervals[1]),
            "gap_single": _interval(intervals[2]), "gap_voted": _interval(intervals[3]),
        })
    return out


def untrained_controls(results_dir) -> dict:
    """The untrained 4-bit runs of SELF_TRAINING_CONTROLS on the questions they share."""
    scored = []
    for spec in SELF_TRAINING_CONTROLS:
        path = generator_run_path(Path(results_dir) / spec["root"], SELF_TRAINING_MODEL, "4bit", TEST_DATASET, ST_SAMPLES)
        if path.exists():
            header, scores = test_run_scores(path)
            scored.append((spec, header, scores))
    if not scored:
        return {"questions": 0, "rows": []}
    qids = _shared(*[scores for _, _, scores in scored])
    rows = [{"name": spec["label"], **_run_facts(header, scores), **test_summary(scores, qids)} for spec, header, scores in scored]
    return {"questions": len(qids), "rows": rows}


def collect_self_training(results_dir) -> dict:
    parquet = REPO_ROOT / GSM8K_TRAIN_PARQUET
    train_answers = load_gsm8k_train_answers(str(parquet)) if parquet.exists() else {}
    experiments, internals = [], {}
    for spec in SELF_TRAINING_EXPERIMENTS:
        entry, inner = score_self_training_experiment(results_dir, spec, train_answers)
        experiments.append(entry)
        internals[spec["id"]] = inner
    comparisons = [{"label": c["label"], "full": c["full"], "quantized": c["quantized"],
                    **compare_arms(internals.get(c["full"]), internals.get(c["quantized"]))}
                   for c in SELF_TRAINING_COMPARISONS]
    return {
        "label_threshold": LABEL_THRESHOLD,
        "experiments": experiments,
        "comparisons": comparisons,
        "controls": untrained_controls(results_dir),
    }


# ---------------------------------------------------------------- assembly and output

def _order(values, preferred=()) -> list:
    rest = sorted(v for v in set(values) if v not in preferred)
    return [v for v in preferred if v in set(values)] + rest


def build_web_data(results_dir) -> dict:
    results_dir = Path(results_dir)
    generator = collect_generator_runs(results_dir)
    verifier, verifier_left_out = collect_verifier_runs(results_dir)

    models = sorted({r["model"] for r in generator})
    model_ids = {safe_name(m): m for m in models}
    for row in verifier + verifier_left_out:   # verifier files carry only the file-safe model name
        row["model"] = model_ids.get(row["model"], row["model"])
    everything = generator + verifier + verifier_left_out
    all_models = _order([r["model"] for r in everything])
    datasets = _order([r["dataset"] for r in everything], tuple(DATASET_LABELS))

    def sort_key(r):
        return (datasets.index(r["dataset"]), model_label(r["model"]).lower(), r["n_samples"], PRECISIONS.index(r["precision"]))

    generator.sort(key=sort_key)
    verifier.sort(key=sort_key)
    verifier_left_out.sort(key=sort_key)
    for run in generator:
        del run["per_question"]

    stamps = [r.pop("_timestamp") for r in generator if r.get("_timestamp")]
    for run in generator:
        run.pop("_timestamp", None)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "source": {
            "results_dir": results_dir.name,
            "generator_runs": len(generator),
            "verifier_runs": len(verifier),
            "latest_generator_run": max(stamps)[:10] if stamps else None,
        },
        "datasets": [{"id": d, "label": DATASET_LABELS.get(d, d)} for d in datasets],
        "models": [{"id": m, "label": model_label(m)} for m in all_models],
        "precisions": [{"id": p, "label": p.replace("bit", "-bit")} for p in PRECISIONS],
        "generator": generator,
        "verifier": verifier,
        "verifier_questions": VERIFIER_QUESTIONS,
        "verifier_left_out": verifier_left_out,
        "label_threshold": LABEL_THRESHOLD,
        "self_training": collect_self_training(results_dir),
    }


def _clean(value, digits: int = 5):
    """JSON-safe copy: NaN and inf become null, floats are rounded."""
    if isinstance(value, dict):
        return {k: _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_web_data(data: dict, path) -> Path:
    """Write `window.RESULTS = {...};` so pages load it with a plain <script> tag, also from file://."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(_clean(data), indent=1, ensure_ascii=False)
    path.write_text(
        "/* Generated by scripts/export_web_data.py from the run files in results/. Do not edit by hand. */\n"
        f"window.RESULTS = {body};\n"
    )
    return path
