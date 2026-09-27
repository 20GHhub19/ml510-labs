"""Complete, label-free windows with explicit observation times."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class WindowDataset:
    values: np.ndarray
    starts: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    coverage: dict

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float32)
        self.starts, self.ends = pd.DatetimeIndex(self.starts), pd.DatetimeIndex(self.ends)
        self.feature_names = tuple(self.feature_names)
        if self.values.ndim != 3 or min(self.values.shape[1:]) < 1:
            raise ValueError('Expected samples × steps × features.')
        if self.values.shape != (len(self.starts), self.values.shape[1], len(self.feature_names)) or len(self.ends) != len(self.starts):
            raise ValueError('Window dimensions and names disagree.')
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('Feature names must be unique and nonempty.')
        if not np.isfinite(self.values).all() or self.starts.hasnans or self.ends.hasnans:
            raise ValueError('Windows and timestamps must be finite.')
        if not self.starts.is_unique or not self.starts.is_monotonic_increasing or (self.ends <= self.starts).any():
            raise ValueError('Window times must be ordered and unique, with positive duration.')
        if len(self.starts) > 1 and (self.starts[1:] < self.ends[:-1]).any():
            raise ValueError('These observation windows must not overlap.')

    @property
    def ids(self):
        return self.starts.strftime('%Y%m%dT%H%M%S')

    def subset(self, mask):
        return WindowDataset(self.values[mask], self.starts[mask], self.ends[mask], self.feature_names,
                             {**self.coverage, 'subset_windows': int(np.asarray(self.values[mask]).shape[0])})

    def summaries(self):
        columns = [name + '_mean' for name in self.feature_names] + [name + '_std' for name in self.feature_names]
        values = np.column_stack([self.values.mean(axis=1), self.values.std(axis=1)])
        return pd.DataFrame(values, columns=columns, index=pd.Index(self.ids, name='window_id'))


def make_windows(frame, *, frequency, steps, start=None, end=None):
    """Build clock-aligned, disjoint windows; exclude incomplete observations."""
    if type(steps) is not int or steps < 1 or not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError('Provide nonempty data and a positive integer window length.')
    index = frame.index
    delta = pd.Timedelta(frequency)
    if delta <= pd.Timedelta(0) or not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise ValueError('Use valid timestamps and a positive fixed frequency.')
    if not index.is_unique or not index.is_monotonic_increasing or not (index == index.floor(delta)).all():
        raise ValueError('Timestamps must be sorted, unique, and aligned with frequency.')
    if start is None:
        start = index.min().floor(delta * steps)
    if end is None:
        end = (index.max() + delta).ceil(delta * steps)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if pd.isna(start) or pd.isna(end) or end <= start or start != start.floor(delta * steps) or end != end.floor(delta * steps):
        raise ValueError('Use ordered clock-aligned window boundaries.')
    grid = pd.date_range(start, end, freq=delta, inclusive='left')
    a = frame.reindex(grid).to_numpy(dtype=np.float32)
    a = a.reshape(-1, steps, len(frame.columns))
    valid = np.isfinite(a).all(axis=(1, 2))
    starts = grid[::steps][valid]
    return WindowDataset(a[valid], starts, starts + delta * steps, tuple(frame.columns),
                         {'candidate_windows': len(a), 'retained_windows': int(valid.sum()),
                          'dropped_windows': int((~valid).sum()), 'frequency': str(delta), 'steps': steps})


def partition_windows(dataset, boundaries):
    """Reject straddling windows rather than share observations between roles."""
    masks = {}
    used = np.zeros(len(dataset.starts), dtype=int)
    for name, (start, end) in boundaries.items():
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if pd.isna(start) or pd.isna(end) or start >= end:
            raise ValueError('Invalid partition boundaries.')
        mask = (dataset.starts >= start) & (dataset.ends <= end)
        if not mask.any():
            raise ValueError(f'No complete windows in {name}.')
        masks[name] = np.asarray(mask)
        used += mask
    if (used != 1).any():
        raise ValueError('Every retained window must belong to exactly one partition.')
    return masks
