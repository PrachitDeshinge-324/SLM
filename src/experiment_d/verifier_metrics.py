"""
Verifier metrics from relabelled traces (see rescoring.load_verifier_traces).

A trace is one stored solution with its recomputed label, its voting answer, whether it was
truncated, and the verifier's verdict. Metrics are built from per-question counts, so a report can
sum them over any set of questions and a bootstrap can resample questions cheaply.
"""
import math
from collections import Counter

COUNT_FIELDS = ("total", "truncated", "truncation_unknown", "unparseable", "tp", "fp", "fn", "tn", "gen_correct", "ver_correct")


def mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0


def ratio(numerator, denominator) -> float:
    return numerator / denominator if denominator else 0.0


def question_counts(traces: list, keep_truncated: bool = False) -> dict:
    """
    Counts for the traces of one question.

    Truncated traces are partial scratchpads, so they are neither a vote nor a test of the verifier:
    they are counted in `truncated` and otherwise ignored. With `keep_truncated` they stay in `total`
    and in the verdict counts as wrong traces (an approval of one is a false positive), so every
    question contributes all its traces; they still never vote. `unparseable` traces have no verdict.
    `gen_correct` is 1 when the majority answer among the question's answered traces is correct, and
    `ver_correct` the same for the majority among the traces the verifier approved (all answered
    traces when it approved none). Both are 0 for a question with no answered trace.
    """
    counts = dict.fromkeys(COUNT_FIELDS, 0)
    answered = []
    for trace in traces:
        cut_off = bool(trace["truncated"])
        if cut_off:
            counts["truncated"] += 1
            if not keep_truncated:
                continue
        if trace["truncated"] is None:
            counts["truncation_unknown"] += 1
        counts["total"] += 1

        correct = trace["correct"] and not cut_off
        verdict = trace["approved"]
        if verdict is None:
            counts["unparseable"] += 1
        elif verdict and correct:
            counts["tp"] += 1
        elif verdict:
            counts["fp"] += 1
        elif correct:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

        if trace["answer"] is not None and not cut_off:
            answered.append({"answer": trace["answer"], "correct": correct, "approved": verdict is True})

    if answered:
        all_answers = [trace["answer"] for trace in answered]
        generator_answer = Counter(all_answers).most_common(1)[0][0]
        counts["gen_correct"] = int(any(t["correct"] for t in answered if t["answer"] == generator_answer))

        approved_answers = [trace["answer"] for trace in answered if trace["approved"]]
        verifier_answer = Counter(approved_answers or all_answers).most_common(1)[0][0]
        counts["ver_correct"] = int(any(t["correct"] for t in answered if t["answer"] == verifier_answer))
    return counts


def metrics_from_totals(total: dict, n_questions: int) -> dict:
    """Metrics from summed question_counts and the number of questions they were summed over."""
    tp, fp, fn, tn = total["tp"], total["fp"], total["fn"], total["tn"]

    parsed = total["total"] - total["unparseable"]
    correct_traces, incorrect_traces = tp + fn, tn + fp
    approved_traces, rejected_traces = tp + fp, fn + tn
    precision = ratio(tp, approved_traces)
    recall = ratio(tp, correct_traces)
    specificity = ratio(tn, incorrect_traces)

    return {
        "Total_Questions": n_questions,
        "Total_Traces": total["total"],
        "Truncated_Traces_Dropped": total["truncated"],
        "Truncation_Unknown": total["truncation_unknown"],
        "Parseable_Traces": parsed,
        "Unparseable_Traces": total["unparseable"],
        "Parse_Coverage": ratio(parsed, total["total"]),
        "Correct_Trace_Prevalence": ratio(correct_traces, parsed),
        "Incorrect_Trace_Prevalence": ratio(incorrect_traces, parsed),
        "Gen_Maj_Acc": ratio(total["gen_correct"], n_questions),
        "Ver_Maj_Acc": ratio(total["ver_correct"], n_questions),
        "Delta_Ver_Gen": ratio(total["ver_correct"] - total["gen_correct"], n_questions),
        "Trace_Accuracy": ratio(tp + tn, parsed),
        "Trace_Precision": precision,
        "Trace_Recall": recall,
        "Trace_Specificity": specificity,
        "Balanced_Accuracy": (recall + specificity) / 2,
        "Negative_Predictive_Value": ratio(tn, rejected_traces),
        "Trace_F1": ratio(2 * precision * recall, precision + recall),
        "Trace_FPR": ratio(fp, fp + tn),
        "False_Discovery_Rate": ratio(fp, approved_traces),
        "Approval_Rate": ratio(approved_traces, parsed),
        "Selective_Risk": ratio(fp, approved_traces),
        "MCC": mcc(tp, fp, fn, tn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    }


def metrics_from_counts(rows: list) -> dict:
    """Metrics over a list of question_counts dicts (one per question, empty questions included)."""
    return metrics_from_totals({field: sum(row[field] for row in rows) for field in COUNT_FIELDS}, len(rows))


def summarize(traces_by_question: dict, selected_questions) -> dict:
    """Metrics over the complete traces of the selected questions (questions with no traces count as empty)."""
    return metrics_from_counts([question_counts(traces_by_question.get(question, [])) for question in selected_questions])
