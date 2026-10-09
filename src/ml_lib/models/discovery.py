"""Small clustering and anomaly adapters; scores increase with unusualness."""
from copy import deepcopy
from pathlib import Path
import pickle
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def checked(values, names, expected=None):
    a = np.asarray(values, dtype=float)
    names = tuple(names)
    if a.ndim not in (2, 3) or not a.size or not np.isfinite(a).all():
        raise ValueError('Expected nonempty finite vectors or windows.')
    if len(names) != a.shape[-1] or len(set(names)) != len(names):
        raise ValueError('Feature names do not match inputs.')
    if expected is not None and (names, a.shape[1:]) != expected:
        raise ValueError('Fitted feature signature differs from inputs.')
    return a


class StoredModel:
    def save(self, directory):
        if not getattr(self, 'is_fitted', False):
            raise RuntimeError('Fit before saving.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / 'model.pkl').open('wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, directory, *, trusted=False):
        if not trusted:
            raise ValueError('Load only your own trusted model artifacts.')
        with (Path(directory) / 'model.pkl').open('rb') as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise ValueError('Unexpected saved model type.')
        return model


class ClusterModel(StoredModel):
    controls = {'kmeans': {'n_clusters', 'n_init'}, 'ward': {'n_clusters'}, 'dbscan': {'eps', 'min_samples'}}

    def __init__(self, family, parameters, *, seed=42):
        if family not in self.controls or set(parameters) != self.controls[family]:
            raise ValueError(f'Expected controls: {self.controls.get(family, self.controls)}')
        self.family, self.parameters, self.seed = family, deepcopy(parameters), seed
        self.is_fitted = False

    def fit(self, X, *, feature_names):
        a = checked(X, feature_names)
        if a.ndim != 2:
            raise ValueError('Clustering expects feature vectors.')
        self.signature = (tuple(feature_names), a.shape[1:])
        self.scaler = StandardScaler().fit(a)
        z = self.scaler.transform(a)
        if self.family == 'kmeans':
            self.estimator = KMeans(**self.parameters, random_state=self.seed)
        elif self.family == 'ward':
            self.estimator = AgglomerativeClustering(**self.parameters, linkage='ward')
        else:
            self.estimator = DBSCAN(**self.parameters)
        self.labels_ = self.estimator.fit_predict(z)
        self.is_fitted = True
        return self

    def predict(self, X, *, feature_names):
        if not self.is_fitted:
            raise RuntimeError('Fit before prediction.')
        if self.family != 'kmeans':
            raise ValueError('This exploratory clustering method has no out-of-sample prediction.')
        a = checked(X, feature_names, self.signature)
        return self.estimator.predict(self.scaler.transform(a))

    def describe(self):
        return {'family': self.family, 'parameters': deepcopy(self.parameters), 'seed': self.seed,
                'preparation': 'training-fitted standard scaling'}


class AnomalyModel(StoredModel):
    controls = {'deviation': set(), 'centroid': {'n_clusters', 'n_init'}, 'pca': {'n_components'},
                'isolation': {'n_estimators', 'max_samples'}, 'lof': {'n_neighbors'}}

    def __init__(self, family, parameters, *, seed=42):
        if family not in self.controls or set(parameters) != self.controls[family]:
            raise ValueError(f'Expected controls: {self.controls.get(family, self.controls)}')
        self.family, self.parameters, self.seed = family, deepcopy(parameters), seed
        self.is_fitted = False

    def fit(self, X, *, feature_names):
        a = checked(X, feature_names)
        if self.family != 'pca' and a.ndim != 2:
            raise ValueError('Summary detectors expect feature vectors.')
        self.signature = (tuple(feature_names), a.shape[1:])
        self.scaler = StandardScaler().fit(a.reshape(-1, a.shape[-1]))
        z = self._prepare(a)
        if self.family == 'pca':
            self.estimator = PCA(**self.parameters, svd_solver='full')
        elif self.family == 'centroid':
            self.estimator = KMeans(**self.parameters, random_state=self.seed)
        elif self.family == 'isolation':
            self.estimator = IsolationForest(**self.parameters, random_state=self.seed, contamination='auto', n_jobs=1)
        elif self.family == 'lof':
            if self.parameters['n_neighbors'] >= len(a):
                raise ValueError('LOF needs more training rows than neighbours.')
            self.estimator = LocalOutlierFactor(**self.parameters, novelty=True, contamination='auto')
        else:
            self.estimator = None
        if self.estimator is not None:
            self.estimator.fit(z)
        self.is_fitted = True
        return self

    def _prepare(self, a):
        return self.scaler.transform(a.reshape(-1, a.shape[-1])).reshape(len(a), -1)

    def score_samples(self, X, *, feature_names):
        if not self.is_fitted:
            raise RuntimeError('Fit before scoring.')
        a = checked(X, feature_names, self.signature)
        z = self._prepare(a)
        if self.family == 'deviation':
            return np.abs(z).max(axis=1)
        if self.family == 'centroid':
            return self.estimator.transform(z).min(axis=1)
        if self.family == 'pca':
            return ((z - self.estimator.inverse_transform(self.estimator.transform(z))) ** 2).mean(axis=1)
        return -self.estimator.score_samples(z)

    def encode(self, X, *, feature_names):
        if not self.is_fitted or self.family != 'pca':
            raise ValueError('Encoding requires fitted PCA.')
        return self.estimator.transform(self._prepare(checked(X, feature_names, self.signature)))

    def reconstruct(self, X, *, feature_names):
        if not self.is_fitted or self.family != 'pca':
            raise ValueError('Reconstruction requires fitted PCA.')
        a = checked(X, feature_names, self.signature)
        reconstructed = self.estimator.inverse_transform(self.encode(a, feature_names=feature_names))
        return self.scaler.inverse_transform(reconstructed.reshape(-1, a.shape[-1])).reshape(a.shape)

    def contributions(self, X, *, feature_names):
        a = checked(X, feature_names, self.signature)
        error = (a - self.reconstruct(a, feature_names=feature_names)) / self.scaler.scale_
        return (error ** 2).mean(axis=1) if a.ndim == 3 else error ** 2

    def describe(self):
        return {'family': self.family, 'parameters': deepcopy(self.parameters), 'seed': self.seed,
                'score_direction': 'higher is more unusual', 'preparation': 'training-fitted per-feature standard scaling',
                'feature_names': list(self.signature[0]) if self.is_fitted else None}
