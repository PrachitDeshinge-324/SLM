from collections import Counter
from typing import List, Tuple, Dict, Any

def compute_metrics(extracted_answers: List[str], ground_truth: str) -> Dict[str, Any]:
    """
    Computes majority vote, confidence, correctness, failure rate, and diversity.
    
    extracted_answers: List of N extracted answers (strings or None).
    ground_truth: The correct answer (string).
    """
    total_samples = len(extracted_answers)
    
    # Filter out None (failed extractions)
    valid_answers = [ans for ans in extracted_answers if ans is not None]
    valid_count = len(valid_answers)
    
    failure_rate = (total_samples - valid_count) / total_samples if total_samples > 0 else 1.0
    
    if valid_count == 0:
        return {
            "majority_answer": None,
            "confidence": 0.0,
            "correctness": False,
            "oracle_correctness": False,
            "failure_rate": failure_rate,
            "diversity": 0,
            "valid_count": 0,
            "answer_distribution": {}
        }
        
    # Compute majority vote
    counts = Counter(valid_answers)
    # most_common(1) returns e.g. [('72', 12)]
    majority_answer, majority_count = counts.most_common(1)[0]
    
    # Confidence is the fraction of *valid* samples that agree with the majority
    confidence = majority_count / valid_count
    
    # Check correctness
    # Ensure both are strings and stripped
    correctness = (str(majority_answer).strip() == str(ground_truth).strip())
    
    # Oracle Correctness (pass@k): Did it get the right answer AT LEAST ONCE?
    oracle_correctness = any(str(ans).strip() == str(ground_truth).strip() for ans in valid_answers)
    
    return {
        "majority_answer": majority_answer,
        "confidence": confidence,
        "correctness": correctness,
        "oracle_correctness": oracle_correctness,
        "failure_rate": failure_rate,
        "diversity": len(counts),  # number of unique answers
        "valid_count": valid_count,
        "answer_distribution": dict(counts)
    }
