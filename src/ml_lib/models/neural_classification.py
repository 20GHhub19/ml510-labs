"""Dense and feature-token classifiers trained with binary cross-entropy."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import os
import pickle
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from ml_lib.features.schema import FeatureSchema
from ml_lib.features.preprocessing import PreprocessingConfig, make_preprocessor
from ml_lib.models.classification import check_inputs


def _training_controls(config):
    if any(type(v) is not int or v < 1 for v in [config.epochs, config.batch_size]):
        raise ValueError("Training budgets must be positive integers.")
    if type(config.patience) is not int or config.patience < 0:
        raise ValueError("Patience must be a nonnegative integer.")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    if not np.isfinite(config.dropout) or not 0 <= config.dropout < 1:
        raise ValueError("Dropout must be finite and in [0, 1).")


@dataclass(frozen=True)
class MLPConfig:
    hidden_units: tuple[int, ...]
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    seed: int = 42
    cpu_only: bool = True
    deterministic: bool = True
    architecture: str = field(default="mlp", init=False)

    def __post_init__(self):
        _training_controls(self)
        if not self.hidden_units or any(type(v) is not int or v < 1 for v in self.hidden_units):
            raise ValueError("Hidden units must be positive integer dimensions.")
        object.__setattr__(self, "hidden_units", tuple(self.hidden_units))


@dataclass(frozen=True)
class TabularTransformerConfig:
    model_dim: int
    num_heads: int
    num_blocks: int
    feedforward_dim: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    seed: int = 42
    cpu_only: bool = True
    deterministic: bool = True
    architecture: str = field(default="tabular_transformer", init=False)

    def __post_init__(self):
        _training_controls(self)
        values = [self.model_dim, self.num_heads, self.num_blocks, self.feedforward_dim]
        if any(type(v) is not int or v < 1 for v in values) or self.model_dim % self.num_heads:
            raise ValueError("Use positive integer dimensions and model_dim divisible by num_heads.")


def _tensorflow(config):
    if config.cpu_only:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if config.deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    import tensorflow as tf
    if config.cpu_only:
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError("Restart the kernel before changing TensorFlow device settings.") from exc
    tf.keras.utils.set_random_seed(config.seed)
    if config.deterministic:
        tf.config.experimental.enable_op_determinism()
    return tf


def _token_preprocessor(schema, config):
    transformers = []
    if schema.numeric:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="median")))
        if config.scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps) if steps else "passthrough", list(schema.numeric)))
    if schema.categorical:
        steps = []
        if config.missing_values == "impute":
            steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        steps.append(("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
        transformers.append(("category", Pipeline(steps), list(schema.categorical)))
    return ColumnTransformer(transformers, remainder="drop")


class NeuralClassifier:
    def __init__(self, config, schema, *, preprocessing):
        self.config, self.schema, self.preprocessing = config, schema, preprocessing
        self.model = None
        self.prepare = None
        self.history = {}

    def _inputs(self, X):
        check_inputs(X, self.schema, self.preprocessing)
        matrix = np.asarray(self.prepare.transform(X), dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Prepared inputs must be finite; inspect missing columns and preprocessing.")
        if isinstance(self.config, MLPConfig):
            return {"features": matrix}
        inputs = {}
        n = len(self.schema.numeric)
        if n:
            inputs["numeric"] = matrix[:, :n]
        if self.schema.categorical:
            inputs["categorical"] = matrix[:, n:].astype(np.int32) + 1
        return inputs

    def fit(self, X, y, *, validation, training_directory=None):
        """Fit on training cases; optionally retain progress and best-weight recovery files.

        Recovery files are not a completed run or an automatic training resume.
        Use save() for the final model bundle consumed by later notebooks.
        """
        labels = check_inputs(X, self.schema, self.preprocessing, y, training=True)
        X_val, y_val = validation
        val_labels = check_inputs(X_val, self.schema, self.preprocessing, y_val)
        if X.index.intersection(X_val.index).size:
            raise ValueError("Training and validation case IDs must be disjoint.")
        self.history = {}
        tf = _tensorflow(self.config)
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(self.config.seed)
        self.prepare = (make_preprocessor(self.schema, self.preprocessing) if isinstance(self.config, MLPConfig)
                        else _token_preprocessor(self.schema, self.preprocessing))
        self.prepare.fit(X)
        train_inputs, val_inputs = self._inputs(X), self._inputs(X_val)
        inputs = {name: tf.keras.Input(shape=array.shape[1:], dtype=array.dtype, name=name)
                  for name, array in train_inputs.items()}
        if isinstance(self.config, MLPConfig):
            representation = inputs["features"]
            for i, units in enumerate(self.config.hidden_units):
                representation = tf.keras.layers.Dense(units, activation="relu", name=f"hidden_{i}")(representation)
                representation = tf.keras.layers.Dropout(self.config.dropout)(representation)
        else:
            from ml_lib.models.tabular_layers import encode_fields
            sizes = []
            if self.schema.categorical:
                encoder = self.prepare.named_transformers_["category"].named_steps["encode"]
                sizes = [len(values) + 1 for values in encoder.categories_]
            representation = encode_fields(inputs, numeric_count=len(self.schema.numeric), category_sizes=sizes, config=self.config)
        probability = tf.keras.layers.Dense(1, activation="sigmoid", name="positive_probability")(representation)
        self.model = tf.keras.Model(inputs, probability)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.config.learning_rate), loss="binary_crossentropy")

        def batches(features, targets, shuffle):
            data = tf.data.Dataset.from_tensor_slices((features, targets.astype(np.float32)[:, None]))
            if shuffle:
                data = data.shuffle(len(targets), seed=self.config.seed, reshuffle_each_iteration=True)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            options.threading.max_intra_op_parallelism = 1
            return data.batch(self.config.batch_size).with_options(options).prefetch(1)

        stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=self.config.patience, restore_best_weights=True)
        callbacks = [stop]
        csv_logger = None
        status = {"completed": False, "state": "training", "epochs_finished": 0}
        directory = Path(training_directory) if training_directory is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            # Repeated fits replace only their own recovery files, never another run.
            for filename in ("best.keras", "history.csv"):
                (directory / filename).unlink(missing_ok=True)
            with (directory / "preparation.pkl").open("wb") as stream:
                pickle.dump(self.prepare, stream)
            configuration = {"config": asdict(self.config), "schema": asdict(self.schema),
                             "preprocessing": asdict(self.preprocessing)}
            (directory / "configuration.json").write_text(json.dumps(configuration, indent=2), encoding="utf-8")

            def write_status():
                (directory / "training.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

            def epoch_finished(epoch, logs):
                status["epochs_finished"] = epoch + 1
                status["validation_loss"] = float(logs["val_loss"])
                write_status()
                print(f"Epoch {epoch + 1}/{self.config.epochs}: loss={logs['loss']:.5f}, val_loss={logs['val_loss']:.5f}", flush=True)

            write_status()
            csv_logger = tf.keras.callbacks.CSVLogger(str(directory / "history.csv"))
            callbacks.extend([
                tf.keras.callbacks.ModelCheckpoint(str(directory / "best.keras"), monitor="val_loss",
                                                   mode="min", save_best_only=True),
                csv_logger,
                tf.keras.callbacks.LambdaCallback(on_epoch_end=epoch_finished),
            ])
        try:
            result = self.model.fit(batches(train_inputs, labels, True), validation_data=batches(val_inputs, val_labels, False),
                                    epochs=self.config.epochs, callbacks=callbacks, verbose=0)
        except BaseException:
            if directory is not None:
                status["state"] = "interrupted"
                write_status()
            raise
        finally:
            if csv_logger is not None:
                csv_logger.on_train_end()
        self.history = {key: [float(x) for x in values] for key, values in result.history.items()}
        if directory is not None:
            status.update(completed=True, state="completed")
            write_status()
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Fit or load the classifier first.")
        arrays = self._inputs(X)
        # A batched dataset lets TensorFlow trace prediction once, rather than once per Python call.
        tf = _tensorflow(self.config)
        data = tf.data.Dataset.from_tensor_slices(arrays).batch(self.config.batch_size)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        data = data.with_options(options).prefetch(1)
        p = self.model.predict(data, verbose=0)[:, 0].astype(float)
        return np.column_stack([1 - p, p])

    def describe(self):
        return {"backend": "TensorFlow 2 via tf.keras", "config": asdict(self.config),
                "schema": asdict(self.schema), "preprocessing": asdict(self.preprocessing), "classes": [0, 1],
                "parameters": int(self.model.count_params()) if self.model is not None else None,
                "policies": {"loss": "binary_crossentropy", "optimizer": "Adam", "monitor": "val_loss",
                             "restore_best_weights": True, "class_weight": None, "target_scaling": False,
                             "onednn_override": os.environ.get("TF_ENABLE_ONEDNN_OPTS")},
                "representation": "one-hot inputs" if isinstance(self.config, MLPConfig) else "numeric projections and categorical field embeddings; no temporal positions"}

    def save(self, directory):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted model.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory / "model.keras")
        with (directory / "preparation.pkl").open("wb") as stream:
            pickle.dump(self.prepare, stream)
        metadata = {"kind": "neural", "config": asdict(self.config), "schema": asdict(self.schema),
                    "preprocessing": asdict(self.preprocessing), "history": self.history}
        (directory / "kind.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, directory, *, trusted=False):
        if not trusted:
            raise ValueError("Loading fitted preparation requires trusted=True.")
        directory = Path(directory)
        info = json.loads((directory / "kind.json").read_text(encoding="utf-8"))
        config = dict(info["config"])
        architecture = config.pop("architecture")
        config_type = MLPConfig if architecture == "mlp" else TabularTransformerConfig
        if architecture not in {"mlp", "tabular_transformer"}:
            raise ValueError("Unknown neural architecture.")
        schema = FeatureSchema(**{key: tuple(value) for key, value in info["schema"].items()})
        instance = cls(config_type(**config), schema, preprocessing=PreprocessingConfig(**info["preprocessing"]))
        tf = _tensorflow(instance.config)
        from ml_lib.models import tabular_layers
        instance.model = tf.keras.models.load_model(directory / "model.keras", safe_mode=True)
        with (directory / "preparation.pkl").open("rb") as stream:
            instance.prepare = pickle.load(stream)
        instance.history = info["history"]
        return instance


def build_neural_classifier(config, schema, *, preprocessing):
    return NeuralClassifier(config, schema, preprocessing=preprocessing)
