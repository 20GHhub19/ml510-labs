"""Keep one binary outcome and its inputs attached to each case."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ml_lib.problems.contracts import PredictionContract


def binary_labels(values, *, both_classes=False):
    y = np.asarray(values)
    if y.ndim != 1 or len(y) == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected nonempty binary labels encoded as 0 and 1.")
    if both_classes and len(np.unique(y)) != 2:
        raise ValueError("Model fitting requires both classes.")
    return y.astype(np.int32)


def positive_probabilities(values, *, n=None):
    p = np.asarray(values, dtype=float)
    if p.ndim != 1 or len(p) == 0 or (n is not None and len(p) != n):
        raise ValueError("Expected one positive-class probability per case.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return p


@dataclass
class ClassificationProblem:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    contract: PredictionContract

    def __post_init__(self):
        if self.contract.task != "classification":
            raise ValueError("A classification contract is required.")
        if not self.X.index.equals(self.y.index) or not self.X.index.equals(self.metadata.index):
            raise ValueError("Inputs, labels and metadata must have identical ordered case IDs.")
        if not self.X.index.is_unique or self.X.index.hasnans:
            raise ValueError("Case IDs must be unique and observed.")
        binary_labels(self.y)
