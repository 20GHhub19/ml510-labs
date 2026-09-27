"""Compare structure and alert evidence without assuming unreported means healthy."""
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, adjusted_rand_score


def cluster_diagnostics(X, labels):
    a, labels = np.asarray(X, float), np.asarray(labels)
    if a.ndim != 2 or len(a) != len(labels) or not len(a) or not np.isfinite(a).all():
        raise ValueError('Provide matching finite features and assignments.')
    assigned = labels != -1
    k = len(np.unique(labels[assigned]))
    score = None
    if 1 < k < int(assigned.sum()):
        score = float(silhouette_score(a[assigned], labels[assigned], sample_size=min(2000, int(assigned.sum())), random_state=42))
    return {'samples': len(a), 'clusters': k, 'noise_fraction': float((~assigned).mean()), 'silhouette': score}


def assignment_agreement(left, right):
    if not left.index.is_unique or not left.index.equals(right.index) or left.isna().any() or right.isna().any():
        raise ValueError('Assignments require matching ordered IDs and nonmissing labels.')
    return float(adjusted_rand_score(left, right))


def reference_threshold(scores, quantile=0.99):
    values = np.asarray(scores, float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or not 0 < quantile < 1:
        raise ValueError('Use finite scores and a quantile strictly between zero and one.')
    return float(np.quantile(values, quantile, method='higher'))


def score_table(dataset, scores):
    scores = np.asarray(scores, float)
    if scores.shape != (len(dataset.starts),) or not np.isfinite(scores).all():
        raise ValueError('One finite score is required per window.')
    return pd.DataFrame({'window_id': dataset.ids, 'start': dataset.starts, 'end': dataset.ends, 'score': scores})


def require_matching_windows(tables):
    if not tables:
        raise ValueError('No scores to compare.')
    keys = ['window_id', 'start', 'end']
    reference = tables[0][keys]
    for table in tables:
        if not table.window_id.is_unique or not table[keys].equals(reference):
            raise ValueError('Comparisons require identical ordered windows.')


def evaluate_alerts(table, threshold, incidents):
    """Compare alerts with half-open reports; scores are available at window end.

    Durations measure flagged observation time, not maintenance workload. Report
    intervals are merged for duration accounting, but evaluated individually.
    """
    if not len(table) or not np.isfinite(threshold) or not np.isfinite(table.score).all():
        raise ValueError('Use nonempty finite scores and a finite threshold.')
    rows = table.copy()
    rows['start'], rows['end'] = pd.to_datetime(rows.start), pd.to_datetime(rows.end)
    if rows[['start', 'end']].isna().any().any():
        raise ValueError('Scoring windows require nonmissing start and end times.')
    if not rows.window_id.is_unique or not rows.start.is_monotonic_increasing or (rows.end <= rows.start).any():
        raise ValueError('Scores require ordered unique windows with positive durations.')
    if len(rows) > 1 and (rows.start.iloc[1:].to_numpy() < rows.end.iloc[:-1].to_numpy()).any():
        raise ValueError('Scoring windows must not overlap.')
    rows['alert'] = rows.score > threshold
    annotated = np.zeros(len(rows), bool)
    evidence = []
    intervals = []
    for event in incidents.itertuples():
        start, end = pd.Timestamp(event.start), pd.Timestamp(event.end)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError('Invalid incident interval.')
        intervals.append((start, end))
        overlap = (rows.start < end) & (rows.end > start)
        available = (rows.end > start) & (rows.end < end) & overlap
        annotated |= overlap.to_numpy()
        seconds = sum(max(0., (min(b, end) - max(a, start)).total_seconds()) for a,b in zip(rows.start[overlap], rows.end[overlap]))
        fraction = seconds / (end - start).total_seconds()
        covered = fraction >= 0.5 and bool(available.any())
        alerts = rows.loc[available & rows.alert, 'end']
        evidence.append({'incident': event.incident, 'start': start, 'end': end,
                         'coverage_fraction': fraction, 'adequately_covered': covered,
                         'detected': bool(len(alerts)),
                         'first_alert': alerts.iloc[0] if len(alerts) else None,
                         'delay_minutes': (alerts.iloc[0] - start).total_seconds()/60 if len(alerts) else None})
    # Merge reports only for duration accounting. Keep individual incident evidence.
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    inside_seconds = np.zeros(len(rows))
    for start, end in merged:
        overlap = (rows.start < end) & (rows.end > start)
        inside_seconds[overlap] += [
            (min(b, end) - max(a, start)).total_seconds()
            for a, b in zip(rows.start[overlap], rows.end[overlap])
        ]
    durations = (rows.end - rows.start).dt.total_seconds().to_numpy()
    episodes = []
    current = None
    for i, row in enumerate(rows.itertuples()):
        if row.alert:
            if current is None or current['end'] != row.start:
                current = {'start': row.start, 'end': row.end, 'first_available': row.end,
                           'windows': 0, 'overlaps_report': False,
                           'duration_hours': 0., 'inside_report_hours': 0.,
                           'outside_report_hours': 0.}
                episodes.append(current)
            current['end'] = row.end
            current['windows'] += 1
            current['overlaps_report'] |= bool(annotated[i])
            current['duration_hours'] += durations[i] / 3600
            current['inside_report_hours'] += inside_seconds[i] / 3600
            current['outside_report_hours'] += (durations[i] - inside_seconds[i]) / 3600
        else:
            current = None
    events = pd.DataFrame(evidence)
    episode_table = pd.DataFrame(episodes, columns=[
        'start', 'end', 'first_available', 'windows', 'overlaps_report',
        'duration_hours', 'inside_report_hours', 'outside_report_hours'])
    days = durations.sum()/86400
    outside = sum(not e['overlaps_report'] for e in episodes)
    metrics = {'samples': len(rows), 'threshold': float(threshold), 'observed_days': float(days),
               'covered_incidents': int(events.adequately_covered.sum()) if len(events) else 0,
               'detected_incidents': int((events.adequately_covered & events.detected).sum()) if len(events) else 0,
               'alert_episodes': len(episodes), 'outside_report_episodes': outside,
               'outside_report_episodes_per_day': outside/days,
               'alerted_hours': float(durations[rows.alert].sum()/3600),
               'inside_report_alerted_hours': float(inside_seconds[rows.alert].sum()/3600),
               'outside_report_alerted_hours': float((durations - inside_seconds)[rows.alert].sum()/3600),
               'alert_fraction': float(rows.alert.mean())}
    return metrics, rows, episode_table, events


def select_detector(table):
    required = ['detected_incidents', 'outside_report_episodes_per_day']
    if table.empty or not np.isfinite(table[required].to_numpy(float)).all():
        raise ValueError('Selection needs finite validation evidence.')
    if table.covered_incidents.nunique() != 1 or table.covered_incidents.iloc[0] == 0:
        raise ValueError('Selection requires matching, nonzero incident coverage.')
    # Stable sorting preserves candidate order when both evidence columns tie.
    return table.sort_values(required, ascending=[False, True], kind='stable').iloc[0]['model']
