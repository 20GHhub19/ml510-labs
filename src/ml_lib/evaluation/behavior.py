"""Compare predictions for the same cases under a declared equality tolerance."""
from numbers import Real
import numpy as np
import pandas as pd


def compare_predictions(reference: pd.Series, candidate: pd.Series, *, atol: float, rtol: float) -> pd.DataFrame:
    """Align candidate by case ID and return differences and equality violations.

    The allowed difference is atol + rtol * abs(reference). This checks a
    declared relation, not accuracy. Inspect rows where ``violated`` is True.
    """
    for name, values in [('reference', reference), ('candidate', candidate)]:
        if not isinstance(values, pd.Series) or values.empty:
            raise ValueError(f'{name} must be a nonempty pandas Series indexed by case ID.')
        if not values.index.is_unique or values.index.to_frame(index=False).isna().any().any():
            raise ValueError(f'{name} requires unique, observed case IDs.')
    if len(reference) != len(candidate) or not reference.index.isin(candidate.index).all():
        raise ValueError('Predictions must contain identical case IDs; no cases may be dropped.')
    if any(not isinstance(t, Real) or not np.isfinite(t) or t < 0 for t in [atol, rtol]):
        raise ValueError('Tolerances must be finite nonnegative scalars.')
    a = reference.to_numpy(dtype=float)
    b = candidate.reindex(reference.index).to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Predictions must be finite.')
    difference = b - a
    tolerance = atol + rtol * np.abs(a)
    return pd.DataFrame({'reference': a, 'candidate': b, 'difference': difference,
                         'tolerance': tolerance, 'violated': np.abs(difference) > tolerance},
                        index=reference.index)
