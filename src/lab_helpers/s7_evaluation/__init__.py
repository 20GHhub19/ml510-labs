"""Load the S7 evidence package and the small earlier-lab teaching populations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_lib.data.loading import sha256_file
from ml_lib.features.schema import FeatureSchema
from ml_lib.experiment.artifacts import RunArtifacts
from lab_helpers.s4_regression.bike_sharing import load_bike_hourly, make_tabular_problem
from lab_helpers.s5_classification.bank_fraud import load_baf, prepare_baf, CATEGORICAL, EXCLUDED


def project_root():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / 'pyproject.toml').is_file():
            return path
    raise FileNotFoundError('Open the notebook inside the ml510-labs repository.')


def load_evidence(path):
    """Verify the evidence files used by the five S7 notebooks."""
    path = Path(path).resolve()
    manifest = path / 'manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('Copy s7_evidence from Moodle into data/processed/. See docs/s7_evaluation.md.')
    info = json.loads(manifest.read_text(encoding='utf-8'))
    if info.get('version') != 1:
        raise ValueError('Expected the S7 version 1 evidence package.')
    required = {
        's4/manifest.json', 's4/predictions.csv.gz',
    }
    missing = required.difference(info.get('files', {}))
    if missing:
        raise ValueError(f'Incomplete S7 evidence package: {sorted(missing)}. Restore it from Moodle.')
    for name in sorted(required):
        fingerprint = info['files'][name]
        file = (path / name).resolve()
        if not file.is_relative_to(path) or not file.is_file() or sha256_file(file) != fingerprint:
            raise ValueError(f'Evidence file missing or changed: {name}. Restore the Moodle package.')
    return info


def start_run(root, evidence, label):
    info = load_evidence(evidence)
    run = RunArtifacts(root, label, config={'evidence': info, 'seed': 42},
                       dataset_path=Path(evidence) / 'manifest.json')
    return run


def bike_development(root):
    """Use only the original S4 training period; later periods remain outside fitting."""
    path = Path(root) / 'data/raw/bike_sharing/hour.csv'
    fingerprint = sha256_file(path)
    source = Path(root) / 'data/processed/s7_evidence/s4/manifest.json'
    expected = json.loads(source.read_text(encoding='utf-8'))['dataset']['sha256']
    if fingerprint != expected:
        raise ValueError('Bike Sharing differs from the supplied evaluation evidence.')
    problem = make_tabular_problem(load_bike_hourly(path))
    keep = problem.X.index < pd.Timestamp('2012-04-01')
    return problem.X.loc[keep].copy(), problem.y.loc[keep].copy(), fingerprint


def small_fraud(root):
    """Fixed, naturally imbalanced subsets; IDs retain their original row positions."""
    path = Path(root) / 'data/raw/bank_account_fraud/Base.csv'
    frame = prepare_baf(load_baf(path))
    fingerprint = sha256_file(path)
    source = Path(root) / 'data/processed/s7_evidence/s5/manifest.json'
    expected = json.loads(source.read_text(encoding='utf-8'))['dataset']['sha256']
    if fingerprint != expected:
        raise ValueError('BAF differs from the supplied evaluation evidence.')
    schema = FeatureSchema(numeric=tuple(c for c in frame if c not in EXCLUDED and c not in CATEGORICAL),
                           categorical=CATEGORICAL)
    samples = {}
    for role, months, size in [('train', [0, 1, 2], 20000), ('monitor', [3], 5000), ('evaluation', [5], 10000)]:
        pool = frame.loc[frame.month.isin(months)]
        selected, _ = train_test_split(pool.index.to_numpy(), train_size=size, random_state=42,
                                       stratify=pool.fraud_bool)
        samples[role] = pool.loc[np.sort(selected)].copy()
    return samples, schema, fingerprint


def paired_rows(frame, reference, candidate, *, keys, value):
    """Require the same complete population, not an unnoticed inner-join subset."""
    a = frame.loc[frame.model == reference].set_index(keys).sort_index()
    b = frame.loc[frame.model == candidate].set_index(keys).sort_index()
    if a.empty or not a.index.is_unique or not b.index.is_unique or not a.index.equals(b.index):
        raise ValueError('Comparison requires identical unique case keys for both methods.')
    if not np.array_equal(a.actual.to_numpy(), b.actual.to_numpy()):
        raise ValueError('Observed labels disagree between methods.')
    result = a.drop(columns=['model']).copy()
    result['reference'] = a[value]
    result['candidate'] = b[value]
    return result.reset_index()


def finish(run, experiment, summary):
    run.record('experiment', {**experiment, 'completed': True})
    return run.write_overview('S7 evaluation evidence', summary,
                              inspect_first=['tables/summary.csv'])
