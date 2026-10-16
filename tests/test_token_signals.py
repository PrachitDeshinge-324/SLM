import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.experiment_d.token_signals import aggregate, position_statistics, target_ids

torch.manual_seed(0)


def _reference(logits, targets, temperature):
    """The definitions written out in float64, one position at a time."""
    rows = []
    for row, target in zip(logits.double(), targets):
        p = torch.softmax(row, -1)
        logp = torch.log(p)
        pt = torch.softmax(row / temperature, -1)
        logpt = torch.log(pt)
        rows.append({
            "logp": logp[target].item(), "entropy": -(p * logp).sum().item(),
            "selfcert": -math.log(len(row)) - logp.mean().item(),   # KL(Uniform || p)
            "logp_T": logpt[target].item(), "entropy_T": -(pt * logpt).sum().item(), "p_max": p.max().item(),
        })
    return rows


# Uniform logits: entropy log V, self-certainty 0, every token has probability 1/V.
V = 16
stats = position_statistics(torch.zeros(5, V), torch.zeros(5, dtype=torch.long), top_k=V)
assert torch.allclose(stats["entropy"], torch.full((5,), math.log(V)), atol=1e-5)
assert torch.allclose(stats["selfcert"], torch.zeros(5), atol=1e-5)
assert torch.allclose(stats["logp"], torch.full((5,), -math.log(V)), atol=1e-5)

# A peaked distribution: entropy near 0, self-certainty large and positive.
peaked = torch.zeros(1, V)
peaked[0, 3] = 30.0
stats = position_statistics(peaked, torch.tensor([3]), top_k=V)
assert stats["entropy"].item() < 1e-6 and stats["selfcert"].item() > 5 and stats["p_max"].item() > 0.999

# Random logits against the float64 definitions, at the sampling temperature and not.
logits = torch.randn(7, 40) * 3
targets = torch.randint(0, 40, (7,))
for temperature in (1.0, 0.7):
    got = position_statistics(logits, targets, temperature=temperature, top_k=10, chunk=3)
    for name in ("logp", "entropy", "selfcert", "logp_T", "entropy_T", "p_max"):
        want = torch.tensor([row[name] for row in _reference(logits, targets, temperature)], dtype=torch.float64)
        assert torch.allclose(got[name].double(), want, atol=1e-4), (name, got[name], want)

# Chunked and unchunked agree, and bfloat16 input is accepted (computed in float32).
a = position_statistics(logits, targets, chunk=2)
b = position_statistics(logits, targets, chunk=64)
assert all(torch.equal(a[name], b[name]) for name in a)
assert torch.allclose(position_statistics(logits.bfloat16(), targets)["entropy"], a["entropy"], atol=0.05)

# Support of the sampler: rank in the top-k, and the nucleus of the renormalised top-k.
flat = torch.zeros(3, 100)
flat[0, 5] = 1.0                                   # target rank 0
flat[1, :] = torch.linspace(5, 0, 100)             # target 60 places down
stats = position_statistics(flat, torch.tensor([5, 60, 0]), top_k=50, top_p=0.95)
assert stats["in_topk"].tolist() == [True, False, True]
assert stats["in_nucleus"].tolist() == [True, False, True]
dominant = torch.zeros(1, 8)
dominant[0, 0] = 10.0                              # one token holds more than 95% of the mass
assert position_statistics(dominant, torch.tensor([0]), top_k=8)["in_nucleus"].item() is True
assert position_statistics(dominant, torch.tensor([1]), top_k=8)["in_nucleus"].item() is False
assert position_statistics(dominant, torch.tensor([1]), top_k=8, top_p=1.0)["in_nucleus"].item() is True

# A shape mismatch is an error, not silent truncation.
try:
    position_statistics(torch.zeros(3, 4), torch.zeros(2, dtype=torch.long))
    raise AssertionError("expected ValueError")
except ValueError:
    pass

# Reconstructing the written tokens.
END = 99
assert target_ids([1, 2, 3, 4], "\n", True, 3, END) == ([1, 2, 3, END], [])          # trailing newline dropped
assert target_ids([1, 2, 3], "x", True, 3, END) == ([1, 2, 3, END], [])               # stored length already right
assert target_ids([1, 2, 3, 4], "x", True, 3, END) == ([1, 2, 3, 4, END], ["length_mismatch"])
assert target_ids([1, 2, 3, 4], "\n", False, 4, END) == ([1, 2, 3, 4], [])            # cut off: no end token
assert target_ids([1, 2, 3, 4, 5], "\n", False, 4, END) == ([1, 2, 3, 4, 5], ["length_mismatch"])

# Aggregation: thirds, the answer span, the end token kept apart.
close = lambda a, b: math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)   # the tensors are float32
T = 10
stats = {
    "logp": torch.arange(T, dtype=torch.float) * -0.1, "entropy": torch.arange(T, dtype=torch.float),
    "selfcert": torch.ones(T), "p_max": torch.full((T,), 0.5), "logp_T": torch.zeros(T), "entropy_T": torch.zeros(T),
    "in_topk": torch.ones(T, dtype=torch.bool), "in_nucleus": torch.ones(T, dtype=torch.bool),
}
finished = aggregate(stats, finished=True, answer_start=6)
assert finished["n_tokens"] == 9
assert close(finished["entropy_mean"], sum(range(9)) / 9)
assert close(finished["end_logp"], -0.9) and close(finished["logp_sum_with_end"], sum(-0.1 * i for i in range(10)))
assert close(finished["entropy_answer"], (6 + 7 + 8) / 3)
assert close(finished["entropy_first"], (0 + 1 + 2) / 3) and close(finished["entropy_last"], (6 + 7 + 8) / 3)
cut = aggregate(stats, finished=False, answer_start=None)
assert cut["n_tokens"] == 10 and cut["end_logp"] is None and cut["entropy_answer"] is None
assert aggregate({k: v[:1] for k, v in stats.items()}, finished=True) == {"n_tokens": 0}

print("All token signal tests passed!")
