"""
Do two inference engines give the same verifier verdicts on the same solutions?

Compares the stored verdicts of the one-line verifier (v0, Hugging Face, greedy) with verdicts the same
model gives today on the same prompts through another engine or dtype. Verdicts are True / False / None
(None: no verdict could be parsed). Reference answers enter only `verifier_rates`, for evaluation.

Pure functions: nothing here loads a model or reads a file.
"""


def agreement(a: list, b: list) -> dict:
    """How often two verdict lists agree: overall (None counts as its own verdict), and Cohen's kappa."""
    if len(a) != len(b) or not a:
        raise ValueError("two verdict lists of the same, non-zero length are needed")
    n = len(a)
    same = sum(x == y for x, y in zip(a, b))
    labels = (True, False, None)
    expected = sum((a.count(v) / n) * (b.count(v) / n) for v in labels)
    kappa = (same / n - expected) / (1 - expected) if expected < 1 else 1.0
    flips = {f"{x}->{y}": sum(1 for p, q in zip(a, b) if p == x and q == y)
             for x in labels for y in labels if x != y}
    return {"n": n, "agreement": same / n, "kappa": kappa, "flips": {k: v for k, v in flips.items() if v}}


def verifier_rates(verdicts: list, correct: list) -> dict:
    """Parse rate, approval rate, and the approval rate on right (TPR) and on wrong (FPR) solutions."""
    def rate(xs):
        return sum(xs) / len(xs) if xs else None
    return {"parsed": rate([v is not None for v in verdicts]),
            "approved": rate([v is True for v in verdicts]),
            "tpr": rate([v is True for v, c in zip(verdicts, correct) if c]),
            "fpr": rate([v is True for v, c in zip(verdicts, correct) if not c])}
