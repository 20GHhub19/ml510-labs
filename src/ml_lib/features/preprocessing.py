"""Explicit preprocessing choices, fitted inside the training pipeline."""
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ml_lib.features.schema import FeatureSchema


def check_tabular_inputs(X, schema, config, *, training=False):
    """Check selected inputs before fitting or applying preparation."""
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("Expected a nonempty input DataFrame.")
    schema.validate(X)
    selected = X[schema.columns]
    if config.missing_values == "error" and selected.isna().any().any():
        raise ValueError("Missing selected inputs; investigate or explicitly choose imputation.")
    if schema.numeric:
        numeric = selected[list(schema.numeric)].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric).any():
            raise ValueError("Numeric inputs may be missing, but not infinite.")
    if training:
        empty_columns = selected.columns[selected.isna().all()].tolist()
        if empty_columns:
            raise ValueError(f"Training inputs have no observed values in: {empty_columns}.")


@dataclass(frozen=True)
class PreprocessingConfig:
    scale_numeric: bool
    missing_values: Literal["error", "impute"]

    def __post_init__(self):
        if type(self.scale_numeric) is not bool or self.missing_values not in {"error", "impute"}:
            raise ValueError("Choose scale_numeric=True/False and missing_values='error'/'impute'.")


def make_preprocessor(schema: FeatureSchema, config: PreprocessingConfig) -> ColumnTransformer:
    transformations = []
    if schema.numeric:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="median")))
        if config.scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformations.append(("numeric", Pipeline(steps) if steps else "passthrough", list(schema.numeric)))
    if schema.categorical:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
        transformations.append(("category", Pipeline(steps), list(schema.categorical)))
    return ColumnTransformer(transformations, remainder="drop", verbose_feature_names_out=True)
