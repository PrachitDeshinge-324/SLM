"""Generate verifier metrics on a shared 100-question evaluation slice."""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_verifier_output(output_text: str, student_answer=None):
    """Parse the constrained verifier response without importing plotting dependencies."""
    if not output_text:
        return [], None
    text = re.sub(r"<think>.*?(?:</think>|$)", "", output_text, flags=re.DOTALL).strip()

    patterns = (
        r"(?i)(?:\*{1,2})?Final\s+Conclusion(?:\*{1,2})?\s*:\s*(?:\*{1,2})?\s*(Correct|Incorrect)\b",
        r"(?i)<final_conclusion>\s*(Correct|Incorrect)\s*</final_conclusion>",
        r"(?i)(?:\*{1,2})?Conclusion(?:\*{1,2})?\s*:\s*(?:the\s+student'?s?\s+solution\s+is\s+)?(?:\*{1,2})?\s*(Correct|Incorrect)\b",
        r"(?i)\b(?:the\s+student'?s?\s+|this\s+)?solution\s+is\s+(?:\*{1,2})?\s*(correct|incorrect)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return [], match.group(1).lower() == "correct"

    tail = text[-250:]
    if re.search(r"(?i)\b(?:is\s+|judged\s+as\s+|marked\s+as\s+)(?:\*{1,2})?incorrect\b", tail):
        return [], False
    if re.search(r"(?i)\b(?:is\s+|judged\s+as\s+|marked\s+as\s+)(?:\*{1,2})?correct\b", tail):
        return [], True

    if student_answer is not None:
        try:
            from src.experiment_d.extractors import extract_gsm8k_answer
            from src.experiment_d.utils import answers_match

            verifier_answer = extract_gsm8k_answer(text)
            if verifier_answer:
                return [], answers_match(verifier_answer, str(student_answer))
        except Exception:
            pass
    return [], None


PRECISION_ORDER = {"16bit": 0, "8bit": 1, "4bit": 2}


def safe_name(value: str) -> str:
    return value.replace("/", "_")


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def find_output_files(results_dir: Path, dataset: str, n_samples: str) -> dict[tuple[str, str], Path]:
    pattern = f"{dataset}/verifier_output/*/verifier_output_*_{n_samples}.jsonl"
    files = {}
    for path in results_dir.glob(pattern):
        match = re.match(r"verifier_output_(.+)_(16bit|8bit|4bit)_", path.name)
        if match:
            files[(match.group(1), match.group(2))] = path
    return files


def choose_shared_questions(records_by_file: dict[tuple[str, str], list[dict]], limit: int) -> list[str]:
    question_sets = [set(record["question"] for record in records) for records in records_by_file.values()]
    shared = set.intersection(*question_sets)
    if len(shared) < limit:
        raise ValueError(f"Only {len(shared)} questions are shared by every file; need {limit}.")
    return sorted(shared)[:limit]


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(records: list[dict], selected_questions: set[str]) -> dict:
    records = [record for record in records if record.get("question") in selected_questions]
    by_question: dict[str, list[dict]] = {}
    tp = fp = fn = tn = unparseable = 0

    for record in records:
        question = record["question"]
        generated_answer = str(record.get("generated_answer"))
        conclusion = record.get("verifier_final_conclusion")
        if "verifier_raw_response" in record:
            _, conclusion = parse_verifier_output(
                record["verifier_raw_response"], student_answer=generated_answer
            )

        if conclusion is None:
            unparseable += 1
        elif conclusion and record["is_correct"]:
            tp += 1
        elif conclusion and not record["is_correct"]:
            fp += 1
        elif not conclusion and record["is_correct"]:
            fn += 1
        else:
            tn += 1

        by_question.setdefault(question, []).append(
            {
                "answer": generated_answer,
                "correct": record["is_correct"],
                "approved": conclusion is True,
            }
        )

    generator_correct = verifier_correct = 0
    for traces in by_question.values():
        all_answers = [trace["answer"] for trace in traces]
        generator_answer = Counter(all_answers).most_common(1)[0][0]
        if any(trace["correct"] for trace in traces if trace["answer"] == generator_answer):
            generator_correct += 1

        approved_answers = [trace["answer"] for trace in traces if trace["approved"]]
        selected_answers = approved_answers or all_answers
        verifier_answer = Counter(selected_answers).most_common(1)[0][0]
        if any(trace["correct"] for trace in traces if trace["answer"] == verifier_answer):
            verifier_correct += 1

    total = len(records)
    parsed = total - unparseable
    correct_traces = tp + fn
    incorrect_traces = tn + fp
    approved_traces = tp + fp
    rejected_traces = fn + tn
    trace_accuracy = ratio(tp + tn, parsed)
    trace_precision = ratio(tp, approved_traces)
    trace_recall = ratio(tp, correct_traces)
    specificity = ratio(tn, incorrect_traces)
    balanced_accuracy = (trace_recall + specificity) / 2
    negative_predictive_value = ratio(tn, rejected_traces)
    false_discovery_rate = ratio(fp, approved_traces)
    approval_rate = ratio(approved_traces, parsed)
    trace_f1 = (
        2 * trace_precision * trace_recall / (trace_precision + trace_recall)
        if trace_precision + trace_recall
        else 0.0
    )

    return {
        "Total_Questions": len(by_question),
        "Total_Traces": total,
        "Parseable_Traces": parsed,
        "Unparseable_Traces": unparseable,
        "Parse_Coverage": parsed / total if total else 0.0,
        "Correct_Trace_Prevalence": ratio(correct_traces, parsed),
        "Incorrect_Trace_Prevalence": ratio(incorrect_traces, parsed),
        "Gen_Maj_Acc": generator_correct / len(by_question) if by_question else 0.0,
        "Ver_Maj_Acc": verifier_correct / len(by_question) if by_question else 0.0,
        "Delta_Ver_Gen": (verifier_correct - generator_correct) / len(by_question) if by_question else 0.0,
        "Trace_Accuracy": trace_accuracy,
        "Trace_Precision": trace_precision,
        "Trace_Recall": trace_recall,
        "Trace_Specificity": specificity,
        "Balanced_Accuracy": balanced_accuracy,
        "Negative_Predictive_Value": negative_predictive_value,
        "Trace_F1": trace_f1,
        "Trace_FPR": fp / (fp + tn) if fp + tn else 0.0,
        "False_Discovery_Rate": false_discovery_rate,
        "Approval_Rate": approval_rate,
        "Selective_Risk": false_discovery_rate,
        "MCC": mcc(tp, fp, fn, tn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def write_report(results_dir: Path, output_dir: Path, dataset: str, n_samples: str, limit: int) -> list[dict]:
    files = find_output_files(results_dir, dataset, n_samples)
    grouped: dict[str, dict[str, Path]] = {}
    all_rows = []
    for (model, precision), path in files.items():
        grouped.setdefault(model, {})[precision] = path

    for model, precision_paths in grouped.items():
        if not {"16bit", "8bit", "4bit"}.issubset(precision_paths):
            continue
        records_by_file = {key: load_records(path) for key, path in precision_paths.items()}
        try:
            shared_questions = choose_shared_questions(records_by_file, limit)
        except ValueError as error:
            print(f"Skipping {model} on {dataset}: {error}")
            continue
        rows = []
        for precision in sorted(precision_paths, key=PRECISION_ORDER.get):
            row = {"Model": model, "Dataset": dataset, "Precision": precision}
            row.update(summarize(records_by_file[precision], set(shared_questions)))
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
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/standardized_verifier_metrics"))
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default=None)
    parser.add_argument("--n-samples", default="n8")
    parser.add_argument("--questions", type=int, default=100)
    args = parser.parse_args()

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