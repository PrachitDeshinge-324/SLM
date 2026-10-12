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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.experiment_d.rescoring import load_verifier_traces, rescore_record
from src.experiment_d.run_files import (
    generator_path_for_verifier_output, is_legacy_file, parse_run_name, read_run_config, safe_name, token_limit,
)
from src.experiment_d.verifier_metrics import COUNT_FIELDS, metrics_from_totals, question_counts

PRECISIONS = ("16bit", "8bit", "4bit")
BASELINE = "16bit"

DATASET_LABELS = {"gsm8k": "GSM8K", "math500": "MATH-500", "ChilleD_SVAMP": "SVAMP", "cqa": "CommonsenseQA"}

# Questions behind every verifier row: 100 x 8 samples = 800 traces. Runs with fewer are left out.
VERIFIER_QUESTIONS = 100

RESAMPLES = 2000
CALIBRATION_BINS = 10
CALIBRATION_MIN_COUNT = 5

GENERATOR_DELTAS = (
    ("accuracy", "correctness"),
    ("sample_accuracy", "sample_accuracy"),
    ("pass_at_k", "oracle_correctness"),
)

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

    per_question, confidences, correctness = {}, [], []
    sums = defaultdict(float)
    for record in records:
        metrics = rescore_record(record, dataset_id)
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
