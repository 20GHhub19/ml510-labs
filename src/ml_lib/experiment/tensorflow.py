"""Configure the existing TensorFlow backend before constructing a neural model."""
import os


def configure_tensorflow(config):
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
