"""Tests for engine_parity: verdict agreement between engines and the verifier's approval rates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment_d.engine_parity import agreement, verifier_rates

a = [True, True, False, None, True, False]
assert agreement(a, list(a))["agreement"] == 1.0 and agreement(a, list(a))["kappa"] == 1.0
b = [True, False, False, None, True, True]
r = agreement(a, b)
assert abs(r["agreement"] - 4 / 6) < 1e-12 and r["flips"] == {"True->False": 1, "False->True": 1}
assert -1 <= r["kappa"] < 1
assert agreement([True] * 4, [True] * 4)["kappa"] == 1.0          # one label only: no chance correction possible
try:
    agreement([True], [True, False])
    raise AssertionError("lists of different lengths must be refused")
except ValueError:
    pass

rates = verifier_rates([True, False, None, True], [True, True, False, False])
assert rates == {"parsed": 0.75, "approved": 0.5, "tpr": 0.5, "fpr": 0.5}

print("All engine_parity tests passed!")
