"""Validation-only model selection with an explicit metric direction."""
import math


def select_model(validation_table, metric: str = "mae") -> dict:
    """Select from a metric-column table; ties keep the first candidate.

    Accepts a pandas DataFrame (or a mapping of metric to model-score mappings).
    Every candidate must have a finite score for the selected metric.
    """
    directions = {"mae": "minimize", "rmse": "minimize", "wape": "minimize", "r2": "maximize",
                  "average_precision": "maximize", "roc_auc": "maximize", "recall_at_fpr": "maximize",
                  "log_loss": "minimize", "brier": "minimize"}
    if metric not in directions:
        raise ValueError(f"Unsupported selection metric {metric!r}. Choose {list(directions)}.")
    if metric not in validation_table:
        raise ValueError(f"Validation scores do not include {metric!r}.")
    constraint = None
    if metric == "recall_at_fpr":
        if "max_false_positive_rate" not in validation_table:
            raise ValueError("recall_at_fpr requires a shared max_false_positive_rate column.")
        column = validation_table["max_false_positive_rate"]
        limits = list(column.values()) if isinstance(column, dict) else list(column)
        invalid = any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in limits)
        if not limits or invalid or len(set(limits)) != 1:
            raise ValueError("recall_at_fpr comparisons require the same finite false-positive-rate limit.")
        constraint = float(limits[0])
    candidates = list(validation_table[metric].items())
    if not candidates:
        raise ValueError("No validation candidates to select.")
    scores = []
    for name, value in candidates:
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Undefined {metric} validation score for {name!r}.") from None
        if not math.isfinite(value):
            raise ValueError(f"Undefined {metric} validation score for {name!r}.")
        scores.append((name, value))
    choose = min if directions[metric] == "minimize" else max
    name, score = choose(scores, key=lambda candidate: candidate[1])
    result = {"selected_model": name, "metric": metric, "direction": directions[metric],
              "validation_score": score, "rule": f"{directions[metric]} validation {metric}"}
    if constraint is not None:
        result["max_false_positive_rate"] = constraint
    return result
