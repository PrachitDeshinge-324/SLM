from collections import Counter
from typing import List, Optional, Dict, Any
from src.experiment_d.utils import answers_match, normalize_answer


def compute_metrics(
    extracted_answers: List[str],
    ground_truth: str,
    cutoffs: Optional[List[bool]] = None,
    drop_truncated: bool = False,
) -> Dict[str, Any]:
    """
    Computes majority vote, confidence, correctness, failure rate, and diversity.

    extracted_answers: List of N extracted answers (strings or None).
    ground_truth: The correct answer (string).
    cutoffs: optional per-sample flags, True if generation hit max_new_tokens.
    drop_truncated: if True, truncated samples are treated as failed extractions
        (they never vote). Their text is a partial scratchpad, so whatever number
        the extractor finds in it is not an answer.

    Votes are counted on normalized answers ('18' == '18.00', '\\dfrac' == '\\frac'),
    and `confidence` is the majority fraction among valid samples while
    `confidence_all` divides by all N, so truncation lowers it.
    """
    total_samples = len(extracted_answers)

    if drop_truncated and cutoffs is not None:
        extracted_answers = [None if cut else ans for ans, cut in zip(extracted_answers, cutoffs)]

    # Filter out None (failed extractions)
    valid_answers = [ans for ans in extracted_answers if ans is not None]
    valid_count = len(valid_answers)

    failure_rate = (total_samples - valid_count) / total_samples if total_samples > 0 else 1.0

    if valid_count == 0:
        return {
            "majority_answer": None,
            "confidence": 0.0,
            "confidence_all": 0.0,
            "correctness": False,
            "oracle_correctness": False,
            "failure_rate": failure_rate,
            "diversity": 0,
            "valid_count": 0,
            "answer_distribution": {}
        }

    # Majority vote over normalized keys, keeping one representative raw string per key
    keys = [normalize_answer(ans) for ans in valid_answers]
    counts = Counter(keys)
    majority_key, majority_count = counts.most_common(1)[0]
    majority_answer = valid_answers[keys.index(majority_key)]

    # Confidence is the fraction of *valid* samples that agree with the majority
    confidence = majority_count / valid_count

    correctness = answers_match(majority_answer, ground_truth)

    # Oracle Correctness (pass@k): Did it get the right answer AT LEAST ONCE?
    oracle_correctness = any(answers_match(ans, ground_truth) for ans in valid_answers)

    return {
        "majority_answer": majority_answer,
        "confidence": confidence,
        "confidence_all": majority_count / total_samples,
        "correctness": correctness,
        "oracle_correctness": oracle_correctness,
        "failure_rate": failure_rate,
        "diversity": len(counts),  # number of unique answers
        "valid_count": valid_count,
        "answer_distribution": dict(counts)
    }
