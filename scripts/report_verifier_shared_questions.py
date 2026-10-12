"""
Report: verifier metrics for every model on one shared slice of questions (default 100).

Same re-scoring as report_verifier_metrics.py, with more metrics per row and a combined CSV:
results/standardized_verifier_metrics/metrics_shared<K>_n<N>.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.rescoring import load_verifier_traces
from src.experiment_d.run_files import is_legacy_file, parse_n_samples, parse_run_name, safe_name
from src.experiment_d.verifier_metrics import summarize


PRECISION_ORDER = {"16bit": 0, "8bit": 1, "4bit": 2}


def find_output_files(results_dir: Path, dataset: str, n_samples: str) -> dict[tuple[str, str], Path]:
    pattern = f"{dataset}/verifier_output/*/verifier_output_*_{n_samples}.jsonl"
    files = {}
    for path in results_dir.glob(pattern):
        if is_legacy_file(path):
            continue
        run = parse_run_name(path.name)
        if run:
            files[(run["model"], run["precision"])] = path
    return files


def choose_shared_questions(traces_by_file: dict[str, dict[str, list[dict]]], limit: int) -> list[str]:
    shared = set.intersection(*[set(traces) for traces in traces_by_file.values()])
    if len(shared) < limit:
        raise ValueError(f"Only {len(shared)} questions are shared by every file; need {limit}.")
    return sorted(shared)[:limit]


def write_report(results_dir: Path, output_dir: Path, dataset: str, n_samples: str, limit: int) -> list[dict]:
    files = find_output_files(results_dir, dataset, n_samples)
    grouped: dict[str, dict[str, Path]] = {}
    all_rows = []
    for (model, precision), path in files.items():
        grouped.setdefault(model, {})[precision] = path

    for model, precision_paths in grouped.items():
        if not {"16bit", "8bit", "4bit"}.issubset(precision_paths):
            continue
        traces_by_file = {key: load_verifier_traces(path, dataset) for key, path in precision_paths.items()}
        try:
            shared_questions = choose_shared_questions(traces_by_file, limit)
        except ValueError as error:
            print(f"Skipping {model} on {dataset}: {error}")
            continue
        rows = []
        for precision in sorted(precision_paths, key=PRECISION_ORDER.get):
            row = {"Model": model, "Dataset": dataset, "Precision": precision}
            row.update(summarize(traces_by_file[precision], set(shared_questions)))
            rows.append(row)
        all_rows.extend(rows)

        report_dir = output_dir / dataset
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / f"metrics_{safe_name(model)}_{dataset}_{n_samples}_shared{limit}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        questions_path = report_dir / f"questions_{safe_name(model)}_{dataset}_{n_samples}_shared{limit}.jsonl"
        questions_path.write_text("".join(json.dumps(question) + "\n" for question in shared_questions))
        print(csv_path)
        print(questions_path)
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", "--results-dir", dest="results_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, default=None,
                        help="Default: <results_dir>/standardized_verifier_metrics")
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default=None)
    parser.add_argument("--n_samples", "--n-samples", dest="n_samples", default="n8")
    parser.add_argument("--questions", type=int, default=100)
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.results_dir / "standardized_verifier_metrics"
    args.n_samples = f"n{parse_n_samples(args.n_samples)}"

    datasets = [args.dataset] if args.dataset else ["gsm8k", "math500"]
    all_rows = []
    for dataset in datasets:
        all_rows.extend(write_report(args.results_dir, args.output_dir, dataset, args.n_samples, args.questions))

    if all_rows:
        combined_path = args.output_dir / f"metrics_shared{args.questions}_{args.n_samples}.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with combined_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(combined_path)


if __name__ == "__main__":
    main()