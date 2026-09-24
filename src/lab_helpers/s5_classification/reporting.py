"""Save S5 model descriptions and histories after visible notebook evaluation."""
import pandas as pd


def record_model_evidence(run, experiment, name, model, fit_seconds, validation_report):
    """Record fitted choices; fitting, prediction and selection stay in the notebook."""
    experiment["models"][name] = {
        "description": model.describe(), "fit_seconds": fit_seconds,
        "validation_operating_point": validation_report.metrics,
    }
    if hasattr(model, "history"):
        history = pd.DataFrame(model.history)
        history.insert(0, "epoch", range(1, len(history) + 1))
        run.save_table("history-" + name, history)
    run.record("experiment", experiment)
