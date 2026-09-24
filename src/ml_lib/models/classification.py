"""Small binary classifiers with preparation fitted on training cases only."""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig, make_preprocessor, check_tabular_inputs
from ml_lib.problems.classification import binary_labels


@dataclass(frozen=True)
class ClassifierSpec:
    family: str
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        controls = {
            "prior": (), "logistic": ("C",),
            "tree": ("max_depth", "min_samples_leaf"),
            "forest": ("n_estimators", "max_depth", "min_samples_leaf", "max_features"),
            "boosting": ("learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "class_weight"),
        }
        if self.family not in controls:
            raise ValueError(f"Unknown classifier family; choose {list(controls)}.")
        expected, supplied = set(controls[self.family]), set(self.parameters)
        if expected != supplied:
            raise ValueError(f"{self.family} expects controls {sorted(expected)}; "
                             f"missing {sorted(expected - supplied)}, unsupported {sorted(supplied - expected)}.")
        if self.family == "boosting" and self.parameters["class_weight"] not in (None, "balanced"):
            raise ValueError("Boosting class_weight must be None or 'balanced'.")


def check_inputs(X, schema, preprocessing, y=None, *, training=False):
    check_tabular_inputs(X, schema, preprocessing, training=training)
    if y is not None:
        labels = binary_labels(y, both_classes=True)
        if len(labels) != len(X) or (isinstance(y, pd.Series) and not X.index.equals(y.index)):
            raise ValueError("Training inputs and labels must be aligned.")
        return labels


class TabularClassifier:
    def __init__(self, spec, schema, *, preprocessing, seed=42):
        self.spec = ClassifierSpec(spec.family, deepcopy(spec.parameters))
        self.schema, self.preprocessing, self.seed = schema, preprocessing, seed
        p = dict(self.spec.parameters)
        factories = {
            "prior": lambda: DummyClassifier(strategy="prior"),
            "logistic": lambda: LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, **p),
            "tree": lambda: DecisionTreeClassifier(random_state=seed, **p),
            "forest": lambda: RandomForestClassifier(random_state=seed, n_jobs=1, **p),
            "boosting": lambda: HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p),
        }
        self.pipeline = Pipeline([("prepare", make_preprocessor(schema, preprocessing)),
                                  ("model", factories[spec.family]())])
        self.is_fitted = False

    def fit(self, X, y):
        labels = check_inputs(X, self.schema, self.preprocessing, y, training=True)
        self.pipeline.fit(X, labels)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Fit or load the classifier first.")
        check_inputs(X, self.schema, self.preprocessing)
        return np.asarray(self.pipeline.predict_proba(X), dtype=float)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def describe(self):
        return {"backend": "scikit-learn", "spec": asdict(self.spec), "schema": asdict(self.schema),
                "preprocessing": asdict(self.preprocessing), "seed": self.seed, "classes": [0, 1],
                "policies": {"class_weight": self.spec.parameters.get("class_weight"), "positive_class": 1,
                             "early_stopping": False if self.spec.family == "boosting" else None,
                             "n_jobs": 1 if self.spec.family == "forest" else None,
                             "logistic_penalty": "l2" if self.spec.family == "logistic" else None}}

    def coefficients(self):
        if not self.is_fitted or self.spec.family != "logistic":
            raise ValueError("Coefficients require a fitted logistic model.")
        return pd.DataFrame({"feature": self.pipeline.named_steps["prepare"].get_feature_names_out(),
                             "coefficient": self.pipeline.named_steps["model"].coef_[0]})

    def save(self, directory):
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "kind.json").write_text(json.dumps({"kind": "classical"}), encoding="utf-8")
        with (directory / "classifier.pkl").open("wb") as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)


def build_classifier(spec, schema, *, preprocessing, seed=42):
    return TabularClassifier(spec, schema, preprocessing=preprocessing, seed=seed)


def load_classifier(directory, *, trusted=False):
    """Load your own classifier bundle, including its fitted preparation."""
    if not trusted:
        raise ValueError("Loading fitted preparation uses pickle; require trusted=True for your own artifacts.")
    directory = Path(directory)
    kind = json.loads((directory / "kind.json").read_text(encoding="utf-8"))["kind"]
    if kind == "classical":
        with (directory / "classifier.pkl").open("rb") as stream:
            model = pickle.load(stream)
        if not isinstance(model, TabularClassifier):
            raise TypeError("Not a tabular classifier artifact.")
        return model
    if kind == "neural":
        from ml_lib.models.neural_classification import NeuralClassifier
        return NeuralClassifier.load(directory, trusted=True)
    if kind == "calibrated":
        from ml_lib.models.calibration import CalibratedClassifier
        return CalibratedClassifier.load(directory, trusted=True)
    raise ValueError(f"Unknown classifier artifact kind: {kind}")
