"""Apply explicit review rules; labels are used only to select a validation cutoff."""
import numpy as np
import pandas as pd
from ml_lib.problems.classification import binary_labels, positive_probabilities


def select_threshold(y_true, probability, *, max_false_positive_rate):
    y = binary_labels(y_true, both_classes=True)
    p = positive_probabilities(probability, n=len(y))
    if not np.isfinite(max_false_positive_rate) or not 0 <= max_false_positive_rate <= 1:
        raise ValueError("The false-positive-rate limit must be in [0, 1].")
    # Score groups move together. A cutoff just above one permits reviewing nobody.
    order = np.argsort(-p, kind="stable")
    scores, labels = p[order], y[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    tp = np.r_[0, np.cumsum(labels)[ends]]
    fp = np.r_[0, np.cumsum(1 - labels)[ends]]
    thresholds = np.r_[np.nextafter(1.0, np.inf), scores[ends]]
    recall, fpr = tp / y.sum(), fp / (len(y) - y.sum())
    valid = np.flatnonzero(fpr <= max_false_positive_rate)
    # Thresholds are descending: argmax retains the highest cutoff on recall ties.
    best = valid[np.argmax(recall[valid])]
    return {"threshold": float(thresholds[best]), "validation_recall": float(recall[best]),
            "validation_false_positive_rate": float(fpr[best]),
            "max_false_positive_rate": float(max_false_positive_rate)}


def review_policy(probability, *, case_ids, threshold, capacity_fraction=None):
    p = positive_probabilities(probability)
    ids = pd.Index(case_ids)
    if len(ids) != len(p) or not ids.is_unique or ids.hasnans:
        raise ValueError("Review cases require unique, observed and aligned IDs.")
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1.0, np.inf):
        raise ValueError("Invalid probability threshold.")
    eligible = p >= threshold
    reviewed = eligible.copy()
    capacity = len(p)
    if capacity_fraction is not None:
        if not np.isfinite(capacity_fraction) or not 0 <= capacity_fraction <= 1:
            raise ValueError("capacity_fraction must be in [0, 1].")
        capacity = int(np.floor(capacity_fraction * len(p)))
        ranking = pd.DataFrame({"position": np.flatnonzero(eligible),
                                "score": p[eligible], "case_id": ids[eligible]})
        ranking = ranking.sort_values(["score", "case_id"], ascending=[False, True], kind="stable")
        reviewed[:] = False
        reviewed[ranking.position.iloc[:capacity].to_numpy()] = True
    return pd.DataFrame({"case_id": ids, "eligible": eligible, "reviewed": reviewed,
                         "over_capacity": eligible & ~reviewed}), {
        "eligible": int(eligible.sum()), "reviews": int(reviewed.sum()),
        "over_capacity": int((eligible & ~reviewed).sum()), "capacity": capacity,
    }
