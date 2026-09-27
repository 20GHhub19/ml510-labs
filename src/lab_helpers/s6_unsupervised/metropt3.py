"""Keep sensor meanings, reference dates, and failure reports specific to S6."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ml_lib.data.loading import sha256_file
from ml_lib.features.windows import make_windows, partition_windows
from ml_lib.problems.discovery import DiscoveryContract

SHA256 = 'db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24'
ANALOG = ('TP2','TP3','H1','DV_pressure','Reservoirs','Oil_temperature','Motor_current')
DIGITAL = ('COMP','DV_eletric','Towers','MPG','LPS','Pressure_switch','Oil_level','Caudal_impulses')
SENSORS = {
    'TP2': ('bar', 'compressor pressure'), 'TP3': ('bar', 'pneumatic panel pressure'),
    'H1': ('bar', 'cyclonic filter discharge pressure'), 'DV_pressure': ('bar', 'dryer discharge pressure'),
    'Reservoirs': ('bar', 'reservoir pressure'), 'Oil_temperature': ('degrees C', 'compressor oil temperature'),
    'Motor_current': ('A', 'motor current; changes with operating state')}
SOURCE = 'https://archive.ics.uci.edu/dataset/791/metropt+3+dataset'


def load_metropt3(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Download the official MetroPT-3 CSV; see docs/s6_unsupervised.md.')
    if sha256_file(path) != SHA256:
        raise ValueError('Unexpected MetroPT-3 fingerprint. Verify the official 2020 release before continuing.')
    frame = pd.read_csv(path)
    if set(frame) != {'Unnamed: 0','timestamp',*ANALOG,*DIGITAL}:
        raise ValueError('Unexpected MetroPT-3 fields.')
    frame = frame.drop(columns='Unnamed: 0').set_index(pd.to_datetime(frame.timestamp)).drop(columns='timestamp')
    if frame.index.hasnans or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError('Raw timestamps must be unique and sorted.')
    if not np.isfinite(frame.to_numpy(float)).all():
        raise ValueError('Unexpected missing or infinite raw sensor values.')
    return frame


def contract():
    return DiscoveryContract('MetroPT-3 maintenance investigation', 'one completed compressor window',
        'ten observed minutes; score available at window end', 'analog sensors observed within the window',
        'operating profiles and unusualness scores', 'prioritize human maintenance inspection',
        'unreported is not confirmed healthy; scores do not establish cause or prevented downtime')


def incidents():
    # Original endpoints have minute precision. Treat that final minute as included.
    # The source repeats incident #1 and attaches a contradictory April maintenance
    # note to the May event. Use stable local IDs; do not infer maintenance dates.
    rows = [('reported_1','2020-04-18 00:00','2020-04-18 23:59'),
            ('reported_2','2020-05-29 23:30','2020-05-30 06:00'),
            ('reported_3','2020-06-05 10:00','2020-06-07 14:30'),
            ('reported_4','2020-07-15 14:30','2020-07-15 19:00')]
    table = pd.DataFrame(rows, columns=['incident','start','reported_end'])
    table['start'] = pd.to_datetime(table.start)
    table['end'] = pd.to_datetime(table.reported_end) + pd.Timedelta(minutes=1)
    table['source'] = SOURCE
    return table


def prepare_minutes(raw, features=ANALOG):
    if not features or not set(features).issubset(ANALOG):
        raise ValueError('Choose analog sensors for the starting representation.')
    counts = raw.resample('min').size()
    minute = raw[list(features)].resample('min').mean()
    valid = counts >= 5
    gaps = raw.index.to_series().diff().dt.total_seconds()
    for i in np.flatnonzero(gaps.to_numpy() > 20):
        valid.loc[raw.index[i-1].floor('min'):raw.index[i].floor('min')] = False
    minute.loc[~valid] = np.nan
    audit = pd.DataFrame({'observations': counts, 'usable': valid})
    return minute, audit


def prepare_data(raw, config, features=ANALOG, steps=10):
    minute, minute_audit = prepare_minutes(raw, features)
    windows = make_windows(minute, frequency='1min', steps=steps)
    masks = partition_windows(windows, config['split'])
    digital = raw[list(DIGITAL)].resample(f'{steps}min').mean().reindex(windows.starts)
    digital.index = pd.Index(windows.ids, name='window_id')
    return windows, masks, minute_audit, digital


def data_audit(raw):
    delta = raw.index.to_series().diff().dt.total_seconds()
    return {'rows': len(raw), 'start': str(raw.index.min()), 'end': str(raw.index.max()),
            'sha256': SHA256, 'source': SOURCE, 'gaps_over_20_seconds': int((delta>20).sum()),
            'spacing_counts': {str(k):int(v) for k,v in delta.value_counts().head(15).items()},
            'notes': ['Sampling is observed, not assumed from inconsistent source descriptions.',
                      'File extends into September 1; final observations remain in test.',
                      'Failure reports are incomplete evidence, not exhaustive labels.',
                      'Repeated report number and contradictory maintenance note preserved as uncertainty.']}


def signature(config, windows):
    return {'dataset_sha256': SHA256, 'split': config['split'], 'features': list(windows.feature_names),
            'steps': windows.values.shape[1], 'frequency': '1min', 'minimum_observations': 5,
            'maximum_gap_seconds': 20, 'contract': contract().to_dict()}


def validate_handoff(source_run, root, expected_signature, expected_phase):
    if not str(source_run).strip():
        raise ValueError('Set source_run to the exact completed run printed by the previous notebook.')
    path = Path(source_run)
    if not path.is_absolute():
        path = Path(root)/path
    if not all((path/name).is_file() for name in ['experiment.json', 'manifest.json']):
        raise FileNotFoundError(
            f'No complete run metadata at {path}. Copy source_run from the end '
            f'of notebook {expected_phase:02d}, not its model or training subfolder.')
    experiment = json.loads((path/'experiment.json').read_text(encoding='utf-8'))
    manifest = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest['dataset']['sha256'] != SHA256 or experiment.get('handoff') != expected_signature:
        raise ValueError('Data, windows, features, or partitions differ from the source run.')
    if experiment.get('phase') != expected_phase:
        raise ValueError(f'Expected the run from S6 notebook {expected_phase:02d}; '
                         f"this run records phase {experiment.get('phase')}.")
    if not experiment.get('completed'):
        raise ValueError('The preceding run must be completed; a checkpoint is not a handoff.')
    return path, experiment
