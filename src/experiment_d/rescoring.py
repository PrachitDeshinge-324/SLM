"""
Re-score stored runs from their raw outputs with the current extractor and matcher.

A run file keeps, next to every raw generation, the `metrics`, `extracted` and `is_correct`
values computed by whatever code existed on the day of the run. Report scripts must not read
those: runs made before and after a scoring fix would be mixed in one table. Everything here
rebuilds the numbers from `raw_samples` / `student_solution`.
"""
import json
from pathlib import Path

from src.experiment_d.answer_extraction import get_extractor
from src.experiment_d.answer_matching import answers_match, normalize_answer
from src.experiment_d.run_files import generator_path_for_verifier_output
from src.experiment_d.self_consistency import summarize_votes
from src.experiment_d.verifier import parse_verdict


def rescore_record(record: dict, dataset_name: str = None) -> dict:
    """
    Metrics for one generator record, rebuilt from its `raw_samples`.

    Truncated samples (hit max_new_tokens) do not vote. On top of summarize_votes this adds
    `cutoff_rate`, `sample_accuracy` (share of the N samples that are complete and correct) and
    redefines `failure_rate` as the share of samples that finished but gave no extractable answer,
    so that truncation and extraction failure stay separate columns.
    """
    extractor = get_extractor(dataset_name or record.get("dataset", ""))
    samples = record["raw_samples"]
    extracted = [extractor(sample["response"]) for sample in samples]
    cutoffs = [bool(sample.get("cutoff", False)) for sample in samples]
    total = len(samples)

    metrics = summarize_votes(extracted, record["ground_truth"], cutoffs=cutoffs, drop_truncated=True)
    complete = [answer for answer, cut in zip(extracted, cutoffs) if not cut]
    metrics["cutoff_rate"] = sum(cutoffs) / total if total else 0.0
    metrics["failure_rate"] = sum(answer is None for answer in complete) / total if total else 1.0
    metrics["sample_accuracy"] = (
        sum(answers_match(answer, record["ground_truth"]) for answer in complete) / total if total else 0.0
    )
    return metrics


def load_generator_samples(path) -> dict:
    """question -> raw_samples of a generator run file ({} if the file does not exist)."""
    path = Path(path)
    samples = {}
    if not path.exists():
        return samples
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if "raw_samples" in record:
                samples[record["question"]] = record["raw_samples"]
    return samples


def relabel_trace(record: dict, extractor, generator_samples: dict = None) -> dict:
    """
    Label one verifier record from the trace text it stores.

    `truncated` is True/False when it can be established: the generator file still holds this exact
    trace, or the record carries `trace_cutoff`. It is None when the generator run was overwritten
    after the verifier ran. The record's own `cutoff` is not used: older verifier runs stored the
    verifier's truncation under that name.
    """
    solution = record["student_solution"]
    answer = extractor(solution)

    truncated = None
    samples = (generator_samples or {}).get(record["question"])
    index = record.get("trace_index")
    if samples is not None and index is not None and index < len(samples) and samples[index]["response"] == solution:
        truncated = bool(samples[index].get("cutoff", False))
    elif "trace_cutoff" in record:
        truncated = bool(record["trace_cutoff"])

    if "verifier_raw_response" in record:
        approved = parse_verdict(record["verifier_raw_response"])
    else:
        approved = record.get("verifier_final_conclusion")

    return {
        "answer": None if answer is None else normalize_answer(answer),
        "correct": answer is not None and answers_match(answer, record["gt_answer"]),
        "truncated": truncated,
        "approved": approved,
    }


def load_verifier_traces(path, dataset_name: str) -> dict:
    """question -> relabelled traces (see relabel_trace) of one verifier output file, in file order."""
    extractor = get_extractor(dataset_name)
    generator_samples = load_generator_samples(generator_path_for_verifier_output(path))
    traces = {}
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            traces.setdefault(record["question"], []).append(relabel_trace(record, extractor, generator_samples))
    return traces
