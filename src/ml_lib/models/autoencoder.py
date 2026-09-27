"""Reconstruct observed windows; no future target or fault label is fitted."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from ml_lib.models.discovery import checked
from ml_lib.experiment.tensorflow import configure_tensorflow as _tensorflow
from ml_lib.experiment.artifacts import write_json


@dataclass(frozen=True)
class AutoencoderConfig:
    architecture: str
    hidden_units: int
    latent_units: int
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    seed: int = 42
    cpu_only: bool = True
    deterministic: bool = True

    def __post_init__(self):
        if self.architecture not in {'dense', 'gru'}:
            raise ValueError('Choose dense or gru reconstruction.')
        if any(type(v) is not int or v < 1 for v in [self.hidden_units, self.latent_units, self.batch_size, self.epochs]):
            raise ValueError('Dimensions and budgets must be positive integers.')
        if type(self.patience) is not int or self.patience < 0 or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Use nonnegative integer patience and positive finite learning rate.')


class Autoencoder:
    def __init__(self, config):
        self.config = config
        self.is_fitted = False
        self.history = {}

    def _scaled(self, a):
        return self.scaler.transform(a.reshape(-1, a.shape[-1])).reshape(a.shape).astype('float32')

    def fit(self, X, *, feature_names, validation, training_directory=None):
        a = checked(X, feature_names)
        if a.ndim != 3:
            raise ValueError('Autoencoders expect samples x steps x features.')
        self.signature = (tuple(feature_names), a.shape[1:])
        v = checked(validation, feature_names, self.signature)
        if self.config.latent_units >= np.prod(a.shape[1:]):
            raise ValueError('The bottleneck must be smaller than the input window.')
        self.is_fitted, self.history = False, {}
        self.scaler = StandardScaler().fit(a.reshape(-1, a.shape[-1]))
        tf = _tensorflow(self.config)
        tf.keras.backend.clear_session()
        inputs = tf.keras.Input(shape=a.shape[1:])
        if self.config.architecture == 'dense':
            h = tf.keras.layers.Flatten()(inputs)
            h = tf.keras.layers.Dense(self.config.hidden_units, activation='relu')(h)
        else:
            h = tf.keras.layers.GRU(self.config.hidden_units)(inputs)
        latent = tf.keras.layers.Dense(self.config.latent_units, name='latent')(h)
        if self.config.architecture == 'dense':
            h = tf.keras.layers.Dense(self.config.hidden_units, activation='relu')(latent)
            h = tf.keras.layers.Dense(int(np.prod(a.shape[1:])))(h)
            output = tf.keras.layers.Reshape(a.shape[1:])(h)
        else:
            h = tf.keras.layers.RepeatVector(a.shape[1])(latent)
            h = tf.keras.layers.GRU(self.config.hidden_units, return_sequences=True)(h)
            output = tf.keras.layers.Dense(a.shape[-1])(h)
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss='mse')
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.config.patience, restore_best_weights=True)]
        directory = Path(training_directory) if training_directory is not None else None
        logger = None
        state = {'completed': False, 'state': 'training', 'epochs_finished': 0}
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            for name in ['best.keras', 'history.csv']:
                (directory/name).unlink(missing_ok=True)
            write_json(directory/'configuration.json', self.describe())
            with (directory/'preparation.pkl').open('wb') as file:
                pickle.dump((self.scaler, self.signature), file)
            write_json(directory/'training.json', state)
            logger = tf.keras.callbacks.CSVLogger(str(directory/'history.csv'))
            callbacks += [logger, tf.keras.callbacks.ModelCheckpoint(str(directory/'best.keras'), monitor='val_loss', save_best_only=True)]
        def progress(epoch, logs):
            state.update(epochs_finished=epoch+1, validation_loss=float(logs['val_loss']))
            if directory is not None:
                write_json(directory/'training.json', state)
            print(f"epoch {epoch+1}: loss={logs['loss']:.5f}, monitor={logs['val_loss']:.5f}", flush=True)
        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=progress))
        train, monitor = self._scaled(a), self._scaled(v)
        try:
            result = self.model.fit(train, train, validation_data=(monitor, monitor), epochs=self.config.epochs,
                                    batch_size=self.config.batch_size, shuffle=False, verbose=0, callbacks=callbacks)
            self.history = {k: [float(x) for x in values] for k, values in result.history.items()}
            if not all(np.isfinite(values).all() for values in self.history.values()):
                raise ValueError('Nonfinite reconstruction loss.')
            self.is_fitted = True
            state.update(completed=True, state='completed')
        except BaseException:
            state.update(completed=False, state='interrupted')
            raise
        finally:
            if logger is not None:
                logger.on_train_end()
            if directory is not None:
                write_json(directory/'training.json', state)
        return self

    def _predict(self, X, names):
        if not self.is_fitted:
            raise RuntimeError('Fit before reconstructing.')
        a = checked(X, names, self.signature)
        z = self._scaled(a)
        r = self.model.predict(z, batch_size=self.config.batch_size, verbose=0)
        if not np.isfinite(r).all():
            raise ValueError('Nonfinite reconstruction.')
        return a, z, r

    def score_samples(self, X, *, feature_names):
        _, z, r = self._predict(X, feature_names)
        return ((z-r)**2).mean(axis=(1,2))

    def contributions(self, X, *, feature_names):
        _, z, r = self._predict(X, feature_names)
        return ((z-r)**2).mean(axis=1)

    def reconstruct(self, X, *, feature_names):
        a, _, r = self._predict(X, feature_names)
        return self.scaler.inverse_transform(r.reshape(-1, a.shape[-1])).reshape(a.shape)

    def encode(self, X, *, feature_names):
        if not self.is_fitted:
            raise RuntimeError('Fit before encoding.')
        a = checked(X, feature_names, self.signature)
        import tensorflow as tf
        encoder = tf.keras.Model(self.model.input, self.model.get_layer('latent').output)
        return encoder.predict(self._scaled(a), verbose=0)

    def describe(self):
        return {'family': 'autoencoder', 'configuration': asdict(self.config),
                'policy': 'Adam; standardized-input MSE; best monitor-loss weights',
                'parameters': self.model.count_params() if hasattr(self, 'model') else None}

    def save(self, directory):
        if not self.is_fitted:
            raise RuntimeError('Fit before saving.')
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory/'model.keras')
        write_json(directory/'configuration.json', asdict(self.config))
        with (directory/'preparation.pkl').open('wb') as file:
            pickle.dump((self.scaler, self.signature, self.history), file)

    @classmethod
    def load(cls, directory, *, trusted=False):
        if not trusted:
            raise ValueError('Load only your own trusted artifacts.')
        directory = Path(directory)
        config = AutoencoderConfig(**json.loads((directory/'configuration.json').read_text()))
        tf = _tensorflow(config)
        instance = cls(config)
        instance.model = tf.keras.models.load_model(directory/'model.keras', safe_mode=True)
        with (directory/'preparation.pkl').open('rb') as file:
            instance.scaler, instance.signature, instance.history = pickle.load(file)
        instance.is_fitted = True
        return instance
