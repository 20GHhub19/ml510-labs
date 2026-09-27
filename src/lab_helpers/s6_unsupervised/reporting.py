"""Save evaluated S6 results without hiding fitting or scoring decisions."""
from pathlib import Path
import pandas as pd
from ml_lib.models.discovery import AnomalyModel
from ml_lib.models.autoencoder import Autoencoder


def save_evidence(run, name, split, result, records):
    metrics, scores, episodes, events = result
    records[(name, split)] = {'model': name, 'split': split, **metrics}
    run.save_table(f'{split}-{name}-scores', scores)
    run.save_table(f'{split}-{name}-episodes', episodes)
    run.save_table(f'{split}-{name}-incidents', events)
    table = pd.DataFrame(records.values())
    run.save_table('summary', table)
    return table


def load_models(source, experiment):
    models = {}
    for name, info in experiment['models'].items():
        cls = Autoencoder if info['kind'] == 'autoencoder' else AnomalyModel
        models[name] = cls.load(Path(source)/'model'/name, trusted=True)
    return models


def model_inputs(model, data):
    """Use minute arrays for reconstruction and summaries for other detectors."""
    if isinstance(model, Autoencoder) or model.family == 'pca':
        return data.values, data.feature_names
    table = data.summaries()
    return table.to_numpy(), tuple(table.columns)


def save_validation(run, name, model, calibration, calibration_scores, cutoff,
                    result, fit_seconds, quantile, records, experiment):
    """Record an already evaluated candidate, its cutoff, and its fitted model."""
    from ml_lib.evaluation.discovery import score_table

    result[0]['fit_seconds'] = fit_seconds
    result[0]['parameters'] = model.describe().get('parameters') if isinstance(model, Autoencoder) else None
    save_evidence(run, name, 'validation', result, records)
    run.save_table('calibration-' + name, score_table(calibration, calibration_scores))
    model.save(run.directory / 'model' / name)
    experiment['models'][name] = {
        'kind': 'autoencoder' if isinstance(model, Autoencoder) else 'classical',
        'threshold': cutoff, 'fit_seconds': fit_seconds, 'description': model.describe(),
        'threshold_quantile': quantile,
        'achieved_reference_alert_fraction': float((calibration_scores > cutoff).mean()),
    }
    if isinstance(model, Autoencoder):
        history = pd.DataFrame(model.history)
        experiment['models'][name]['training'] = {
            'epochs_run': len(history), 'epoch_budget': model.config.epochs,
            'best_monitor_epoch': int(history.val_loss.idxmin()) + 1,
            'best_monitor_loss': float(history.val_loss.min()),
            'early_stopping': len(history) < model.config.epochs,
            'stop_reason': 'patience' if len(history) < model.config.epochs else 'epoch_limit',
        }
    run.record('experiment', experiment)


def evidence_ties(comparison):
    """List candidates sharing the leader's two selection criteria."""
    from ml_lib.evaluation.discovery import select_detector

    leader = select_detector(comparison)
    row = comparison.loc[comparison.model == leader].iloc[0]
    same = ((comparison.detected_incidents == row.detected_incidents)
            & (comparison.outside_report_episodes_per_day == row.outside_report_episodes_per_day))
    return comparison.loc[same, 'model'].tolist()
