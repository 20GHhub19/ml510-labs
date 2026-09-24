"""Separate probability evidence from decisions made with those probabilities."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
from ml_lib.problems.classification import binary_labels, positive_probabilities
from ml_lib.problems.contracts import PredictionContract
from ml_lib.evaluation.policies import select_threshold


def classification_metrics(y_true, probability):
    y = binary_labels(y_true)
    p = positive_probabilities(probability, n=len(y))
    both = len(np.unique(y)) == 2
    return {
        "n": len(y), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def decision_metrics(y_true, reviewed):
    y = binary_labels(y_true)
    decision = np.asarray(reviewed)
    if decision.shape != y.shape or not np.isin(decision, [0, 1]).all():
        raise ValueError("Expected one binary review decision per case.")
    d = decision.astype(bool)
    tp, fp = int(((y == 1) & d).sum()), int(((y == 0) & d).sum())
    fn, tn = int(((y == 1) & ~d).sum()), int(((y == 0) & ~d).sum())
    def ratio(a, b):
        return a / b if b else None
    return {"n": len(y), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "reviews": tp + fp, "review_rate": (tp + fp) / len(y),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "false_positive_rate": ratio(fp, fp + tn), "accuracy": (tp + tn) / len(y)}


@dataclass
class ClassificationReport:
    model_name: str
    contract: PredictionContract
    split: str
    dataset_fingerprint: str
    predictions: pd.DataFrame
    max_false_positive_rate: float | None = None

    def __post_init__(self):
        if self.max_false_positive_rate is not None:
            if self.split != "validation":
                raise ValueError("Select an FPR-constrained threshold on validation only; freeze it before test.")
            if not np.isfinite(self.max_false_positive_rate) or not 0 <= self.max_false_positive_rate <= 1:
                raise ValueError("The false-positive-rate limit must be in [0, 1].")

    @property
    def metrics(self):
        result = classification_metrics(self.predictions.actual, self.predictions.probability)
        if self.max_false_positive_rate is not None:
            chosen = select_threshold(self.predictions.actual, self.predictions.probability,
                                      max_false_positive_rate=self.max_false_positive_rate)
            decisions = decision_metrics(self.predictions.actual,
                                         self.predictions.probability >= chosen["threshold"])
            result.update(recall_at_fpr=chosen["validation_recall"],
                          max_false_positive_rate=self.max_false_positive_rate,
                          threshold=chosen["threshold"],
                          achieved_fpr=decisions["false_positive_rate"],
                          precision=decisions["precision"], reviews=decisions["reviews"])
        return result

    def by(self, column):
        rows = []
        for key, group in self.predictions.groupby(column, observed=True, dropna=False):
            rows.append({column: key, **classification_metrics(group.actual, group.probability)})
        return pd.DataFrame(rows)


def evaluate_classification(y_true, probability, *, case_ids, contract, model_name,
                            split, dataset_fingerprint, metadata=None, max_false_positive_rate=None):
    if contract.task != "classification" or split not in {"train", "calibration", "validation", "test"}:
        raise ValueError("Expected a classification contract and a named evaluation split.")
    y = binary_labels(y_true)
    p = positive_probabilities(probability, n=len(y))
    ids = pd.Index(case_ids)
    if len(ids) != len(y) or not ids.is_unique or ids.hasnans:
        raise ValueError("Case IDs must be observed, unique and aligned with predictions.")
    if isinstance(y_true, pd.Series) and not y_true.index.equals(ids):
        raise ValueError("Observed labels must have the exact ordered case IDs.")
    frame = pd.DataFrame({"case_id": ids, "actual": y, "probability": p})
    if metadata is not None:
        if not metadata.index.equals(ids):
            raise ValueError("Slice metadata must have the exact ordered case IDs.")
        for column in metadata:
            if column in frame:
                raise ValueError(f"Reserved prediction column: {column}")
            frame[column] = metadata[column].to_numpy()
    return ClassificationReport(model_name, contract, split, dataset_fingerprint, frame, max_false_positive_rate)


def classification_comparison(reports):
    if not reports:
        raise ValueError("No reports to compare.")
    first = reports[0]
    names = []
    for report in reports:
        if (report.contract.fingerprint, report.split, report.dataset_fingerprint) != (
                first.contract.fingerprint, first.split, first.dataset_fingerprint):
            raise ValueError("Comparisons require the same contract, split and dataset.")
        if report.max_false_positive_rate != first.max_false_positive_rate:
            raise ValueError("Comparisons require the same false-positive-rate limit.")
        same_cases = pd.Index(report.predictions.case_id).equals(pd.Index(first.predictions.case_id))
        same_labels = np.array_equal(report.predictions.actual, first.predictions.actual)
        if not same_cases or not same_labels:
            raise ValueError("Comparisons require identical ordered cases and labels.")
        names.append(report.model_name)
    if len(set(names)) != len(names):
        raise ValueError("Model names must be unique within a comparison.")
    return pd.DataFrame([{"model": r.model_name, **r.metrics} for r in reports]).set_index("model")


def reliability_table(y_true, probability, *, bins=10):
    y = binary_labels(y_true)
    p = positive_probabilities(probability, n=len(y))
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer.")
    frame = pd.DataFrame({"actual": y, "probability": p,
                          "bin": np.minimum((p * bins).astype(int), bins - 1)})
    result = frame.groupby("bin").agg(n=("actual", "size"), positives=("actual", "sum"),
                                     mean_probability=("probability", "mean"), observed_rate=("actual", "mean"))
    return result.reset_index()
