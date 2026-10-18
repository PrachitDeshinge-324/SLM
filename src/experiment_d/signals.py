"""
Bookkeeping of the one-pass signal run (scripts/run_signals.py): every per-solution signal in one row.

A row holds, for one stored solution: its answer and vote support (no model), the token signals of a
teacher-forced pass (token_signals.py), the mc verifier's P(Correct), and, for the solutions a pilot
selects, the verdicts of each sampled critic with its forced verdict readout (verifier_variants.py).
Reference answers are never read here.

Pure functions: nothing here loads a model or reads a file.
"""
from collections import Counter

from src.experiment_d.answer_matching import normalize_answer


def vote_support(samples: list, extractor) -> list:
    """(answer key, share of all samples that give it) per sample; (None, None) for a cut-off or unanswered one."""
    keys = []
    for sample in samples:
        answer = None if sample.get("cutoff") else extractor(sample["response"])
        keys.append(None if answer is None else normalize_answer(answer))
    counts = Counter(k for k in keys if k is not None)
    return [(k, None if k is None else counts[k] / len(samples)) for k in keys]


def verdict_rates(rows: list, variants) -> dict:
    """variant -> (verdicts that reached the closing line, all verdicts) over the rows that hold it."""
    out = {}
    for variant in variants:
        values = [v for row in rows if variant in row for v in row[variant]["values"]]
        out[variant] = (sum(v is not None for v in values), len(values))
    return out


def gate_failures(rates: dict, minimum: float) -> list:
    """The variants whose share of verdicts reached is below `minimum` (a variant with no verdicts yet passes)."""
    return [variant for variant, (reached, total) in rates.items() if total and reached / total < minimum - 1e-12]
