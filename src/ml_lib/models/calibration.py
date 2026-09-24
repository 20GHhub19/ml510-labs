"""Fit a sigmoid mapping on held-out probability scores."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from ml_lib.problems.classification import binary_labels, positive_probabilities


class SigmoidCalibrator:
    epsilon = 1e-7

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, probability, y):
        labels = binary_labels(y, both_classes=True)
        p = positive_probabilities(probability, n=len(labels))
        scores = logit(np.clip(p, self.epsilon, 1 - self.epsilon))[:, None]
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(scores, labels)
        self.slope = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        return self

    def predict(self, probability):
        if self.slope is None:
            raise RuntimeError("Fit the calibration mapping first.")
        p = positive_probabilities(probability)
        return expit(self.slope * logit(np.clip(p, self.epsilon, 1 - self.epsilon)) + self.intercept)

    def describe(self):
        return {"method": "logistic recalibration of clipped log-odds", "epsilon": self.epsilon,
                "slope": self.slope, "intercept": self.intercept}

    def save(self, path):
        if self.slope is None:
            raise RuntimeError("Cannot save an unfitted calibration mapping.")
        Path(path).write_text(json.dumps(self.describe(), allow_nan=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        info = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls()
        if not all(np.isfinite(info[key]) for key in ["slope", "intercept", "epsilon"]) or not 0 < info["epsilon"] < 0.5:
            raise ValueError("Invalid calibration mapping.")
        model.slope, model.intercept, model.epsilon = info["slope"], info["intercept"], info["epsilon"]
        return model


class CalibratedClassifier:
    def __init__(self, model, calibrator):
        self.model, self.calibrator = model, calibrator
        self.schema, self.preprocessing = model.schema, model.preprocessing

    def predict_proba(self, X):
        p = self.calibrator.predict(self.model.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])

    def describe(self):
        return {"base": self.model.describe(), "calibration": self.calibrator.describe()}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "base")
        self.calibrator.save(directory / "calibration.json")
        (directory / "kind.json").write_text(json.dumps({"kind": "calibrated"}), encoding="utf-8")

    @classmethod
    def load(cls, directory, *, trusted=False):
        from ml_lib.models.classification import load_classifier
        directory = Path(directory)
        return cls(load_classifier(directory / "base", trusted=trusted),
                   SigmoidCalibrator.load(directory / "calibration.json"))
