"""The question sets that analyses share, so that every run is scored on the same questions."""
from pathlib import Path

from src.experiment_d.rescoring import load_verifier_traces
from src.experiment_d.run_files import parse_run_name, safe_name
from src.experiment_d.web_export import standard_questions


def phase2_questions(results_dir, model: str, dataset: str, n_samples: int, limit: int = 100) -> list:
    """
    The question texts the Phase 2 page scores for this model and dataset: the first `limit` questions that
    every kept precision's verifier run completed (all N traces), in the order of the highest kept precision.
    Empty when the verifier runs do not share that many.
    """
    files = {}
    for path in sorted(Path(results_dir).glob(f"{dataset}/verifier_output/*/verifier_output_*.jsonl")):
        run = parse_run_name(path.name)
        if run and run["model"] == safe_name(model) and run["n_samples"] == n_samples:
            files[run["precision"]] = path
    traces = {precision: load_verifier_traces(path, dataset) for precision, path in files.items()}
    selected, _ = standard_questions(traces, n_samples, limit)
    return list(selected or [])


def records_in(records: list, questions: list) -> list:
    """The generator records whose question is in `questions`, in that order (a missing question is an error)."""
    by_question = {record["question"]: record for record in records}
    missing = [q for q in questions if q not in by_question]
    if missing:
        raise ValueError(f"{len(missing)} question(s) are not in the generator run, e.g. {missing[0][:60]!r}")
    return [by_question[q] for q in questions]
