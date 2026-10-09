"""Statistics shared by the reports and the analyses."""
from statistics import NormalDist

import numpy as np


def auroc(scores, labels) -> float:
    """Probability that a positive item outranks a negative one (ties count half). NaN if a class is empty."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def bootstrap_interval(statistic, n_items: int, resamples: int = 2000, seed: int = 0):
    """
    95% percentile interval of statistic(indices) over resamples of n_items indices drawn with
    replacement. Pass a statistic that indexes paired arrays to get a paired (clustered) interval.
    """
    rng = np.random.default_rng(seed)
    values = [statistic(rng.integers(0, n_items, n_items)) for _ in range(resamples)]
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def sensitivity_and_criterion(true_positive_rate, false_positive_rate, n_positive: int, n_negative: int):
    """
    Signal detection theory for a yes/no judge: (d', c).
    d' is how well correct and incorrect items are told apart; c is the bias, negative when the
    judge leans towards approving. Rates of exactly 0 or 1 are pulled in by half an observation.
    """
    z = NormalDist().inv_cdf
    clip = lambda rate, n: min(max(rate, 0.5 / n), 1 - 0.5 / n)
    z_hit, z_false_alarm = z(clip(true_positive_rate, n_positive)), z(clip(false_positive_rate, n_negative))
    return z_hit - z_false_alarm, -(z_hit + z_false_alarm) / 2


def pool_random_effects(estimates, standard_errors):
    """
    DerSimonian-Laird random-effects pooling of per-cell estimates.
    Returns (cells used, pooled mean, its standard error, tau, I2): tau is the standard deviation of
    the true effect between cells, I2 the share of observed variation that is not sampling error.
    """
    y, v = np.asarray(estimates, float), np.asarray(standard_errors, float) ** 2
    usable = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[usable], v[usable]
    k = len(y)
    w = 1 / v
    fixed_mean = (w * y).sum() / w.sum()
    q = (w * (y - fixed_mean) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    w_random = 1 / (v + tau2)
    mean = (w_random * y).sum() / w_random.sum()
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return k, mean, np.sqrt(1 / w_random.sum()), np.sqrt(tau2), i2
