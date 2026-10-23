"""Estimate a paired difference while keeping observations or groups together."""
import numpy as np
import pandas as pd


def paired_bootstrap(reference, candidate, *, statistic=np.mean, groups=None,
                     n_resamples=2000, confidence_level=0.95, seed=42):
    """Return candidate minus reference, a percentile interval and its draws.

    Inputs must already describe the same ordered cases. A draw samples rows,
    or whole groups with replacement, using identical indices for both methods.
    Group draws pool their rows: unequal group sizes retain row-level weighting.
    This conditions on the fitted models and does not estimate training variance.
    Undefined statistics raise an error rather than silently dropping draws.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(candidate, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("Provide at least two aligned scalar observations per method.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired observations must be finite.")
    if type(n_resamples) is not int or n_resamples < 2:
        raise ValueError("n_resamples must be an integer of at least two.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between zero and one.")
    if groups is None:
        units = None
        count = len(a)
    else:
        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != len(a) or pd.isna(g).any():
            raise ValueError("Groups must be observed and aligned with cases.")
        codes, labels = pd.factorize(g, sort=False)
        count = len(labels)
        if count < 2:
            raise ValueError("At least two groups are required.")
        units = [np.flatnonzero(codes == i) for i in range(count)]

    def difference(indices):
        result = float(statistic(b[indices]) - statistic(a[indices]))
        if not np.isfinite(result):
            raise ValueError("The statistic is undefined for a bootstrap draw.")
        return result

    estimate = difference(np.arange(len(a)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = rng.integers(0, count, size=count)
        indices = sampled if units is None else np.concatenate([units[j] for j in sampled])
        draws[i] = difference(indices)
    tail = (1 - confidence_level) / 2
    low, high = np.quantile(draws, [tail, 1 - tail])
    return {"difference": estimate, "low": float(low), "high": float(high),
            "confidence_level": confidence_level, "units": count, "draws": draws}
