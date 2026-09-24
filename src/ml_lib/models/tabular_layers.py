"""Serializable feature tokens for tabular attention; fields have no time order."""
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="ml_lib")
class FeatureTokens(tf.keras.layers.Layer):
    def __init__(self, numeric_count, category_sizes, model_dim, **kwargs):
        super().__init__(**kwargs)
        self.numeric_count = numeric_count
        self.category_sizes = list(category_sizes)
        self.model_dim = model_dim
        self.embeddings = [tf.keras.layers.Embedding(size, model_dim, name=f"field_{i}")
                           for i, size in enumerate(category_sizes)]

    def build(self, input_shape):
        self.numeric_weights = self.add_weight(name="numeric_weights", shape=(self.numeric_count, self.model_dim), initializer="glorot_uniform")
        self.numeric_bias = self.add_weight(name="numeric_bias", shape=(self.numeric_count, self.model_dim), initializer="zeros")
        self.class_token = self.add_weight(name="class_token", shape=(1, 1, self.model_dim), initializer="glorot_uniform")
        for embedding in self.embeddings:
            embedding.build((None,))
        super().build(input_shape)

    def call(self, inputs):
        first = next(iter(inputs.values()))
        tokens = [tf.tile(self.class_token, [tf.shape(first)[0], 1, 1])]
        if self.numeric_count:
            numeric = tf.cast(inputs["numeric"], self.compute_dtype)
            tokens.append(numeric[:, :, None] * self.numeric_weights[None, :, :] + self.numeric_bias[None, :, :])
        for i, embedding in enumerate(self.embeddings):
            tokens.append(embedding(inputs["categorical"][:, i])[:, None, :])
        return tf.concat(tokens, axis=1)

    def get_config(self):
        return {**super().get_config(), "numeric_count": self.numeric_count,
                "category_sizes": self.category_sizes, "model_dim": self.model_dim}


@tf.keras.utils.register_keras_serializable(package="ml_lib")
class FirstToken(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, 0, :]


def encode_fields(inputs, *, numeric_count, category_sizes, config):
    x = FeatureTokens(numeric_count, category_sizes, config.model_dim, name="feature_tokens")(inputs)
    for i in range(config.num_blocks):
        normalized = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"attention_norm_{i}")(x)
        attention = tf.keras.layers.MultiHeadAttention(num_heads=config.num_heads,
                    key_dim=config.model_dim // config.num_heads, dropout=config.dropout,
                    name=f"attention_{i}")(normalized, normalized)
        x = tf.keras.layers.Add()([x, tf.keras.layers.Dropout(config.dropout)(attention)])
        normalized = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"ff_norm_{i}")(x)
        hidden = tf.keras.layers.Dense(config.feedforward_dim, activation="relu")(normalized)
        hidden = tf.keras.layers.Dropout(config.dropout)(hidden)
        hidden = tf.keras.layers.Dense(config.model_dim)(hidden)
        x = tf.keras.layers.Add()([x, tf.keras.layers.Dropout(config.dropout)(hidden)])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return FirstToken(name="case_representation")(x)
