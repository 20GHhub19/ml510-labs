"""Keep tabular inputs, targets, and selected feature columns together."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ml_lib.features.schema import FeatureSchema
from ml_lib.problems.contracts import PredictionContract


@dataclass
class RegressionProblem:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    contract: PredictionContract
    coverage: dict

    def __post_init__(self):
        if not self.X.index.equals(self.y.index) or not self.X.index.equals(self.metadata.index):
            raise ValueError("X, y and metadata must have identical ordered indexes.")
        if not self.X.index.is_unique:
            raise ValueError("One row must represent one prediction opportunity.")
        if len(self.y) == 0 or not np.isfinite(self.y.to_numpy(dtype=float)).all():
            raise ValueError("Targets must be observed, finite and nonempty.")
