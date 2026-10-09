import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_d.evaluation_stats import (
    auroc,
    bootstrap_interval,
    pool_random_effects,
    sensitivity_and_criterion,
)


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5      # ties count half
    assert math.isnan(auroc([0.1, 0.2], [True, True]))                          # no negatives


def test_sensitivity_and_criterion():
    # a judge that approves 84% of correct and 16% of incorrect items: unbiased, d' about 2
    d_prime, criterion = sensitivity_and_criterion(0.84, 0.16, 100, 100)
    assert abs(d_prime - 1.989) < 0.01 and abs(criterion) < 1e-9
    # approving more of everything leaves sensitivity alone and moves the criterion down (lenient)
    lenient = sensitivity_and_criterion(0.95, 0.40, 100, 100)
    assert lenient[1] < 0
    # rates of exactly 1 and 0 stay finite
    assert all(math.isfinite(value) for value in sensitivity_and_criterion(1.0, 0.0, 50, 50))


def test_pool_random_effects():
    # identical estimates: no between-cell variation, pooled mean equals the estimate
    k, mean, se, tau, i2 = pool_random_effects([2.0, 2.0, 2.0], [1.0, 1.0, 1.0])
    assert (k, tau, i2) == (3, 0.0, 0.0) and abs(mean - 2.0) < 1e-12 and abs(se - 1 / math.sqrt(3)) < 1e-12
    # widely separated estimates with tiny standard errors: almost all variation is real
    _, mean, _, tau, i2 = pool_random_effects([-10.0, 0.0, 10.0], [0.1, 0.1, 0.1])
    assert abs(mean) < 1e-9 and tau > 9 and i2 > 0.99
    # cells with a missing estimate are left out
    assert pool_random_effects([1.0, float("nan"), 3.0], [1.0, 1.0, 1.0])[0] == 2


def test_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = [0.0] * 50 + [1.0] * 50
    statistic = lambda index: sum(values[i] for i in index) / len(index)
    low, high = bootstrap_interval(statistic, len(values), resamples=500, seed=1)
    assert low < 0.5 < high
    assert (low, high) == bootstrap_interval(statistic, len(values), resamples=500, seed=1)


if __name__ == "__main__":
    test_auroc()
    test_sensitivity_and_criterion()
    test_pool_random_effects()
    test_bootstrap_interval_is_reproducible_and_brackets_the_mean()
    print("All evaluation statistics tests passed!")
