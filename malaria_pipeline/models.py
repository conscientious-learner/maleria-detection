"""Keras augmentation, model construction, and compilation."""
from __future__ import annotations

import tensorflow as tf

from .config import ExperimentConfig


def build_augmentation(config: ExperimentConfig) -> tf.keras.Sequential:
    """Create modest cell-preserving spatial and photometric augmentation."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=config.seed),
            tf.keras.layers.RandomRotation(config.rotation_factor, seed=config.seed + 1),
            tf.keras.layers.RandomZoom(config.zoom_factor, seed=config.seed + 2),
            tf.keras.layers.RandomTranslation(
                config.translation_factor, config.translation_factor, seed=config.seed + 3
            ),
            tf.keras.layers.RandomBrightness(
                config.brightness_factor, value_range=(0.0, 1.0), seed=config.seed + 4
            ),
            tf.keras.layers.RandomContrast(config.contrast_factor, seed=config.seed + 5),
        ],
        name="safe_data_augmentation",
    )


def _compile_model(
    model: tf.keras.Model, config: ExperimentConfig, learning_rate: float
) -> tf.keras.Model:
    """Compile a binary model with AdamW and smoothed binary cross-entropy."""
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        beta_1=config.adam_beta_1,
        beta_2=config.adam_beta_2,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def conv_block(
    inputs: tf.Tensor,
    filters: int,
    config: ExperimentConfig,
    regularized: bool = False,
) -> tf.Tensor:
    """Add one convolution/pooling block, optionally with BN and spatial dropout."""
    regularizer = tf.keras.regularizers.l2(config.weight_decay)
    if regularized:
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", use_bias=False, kernel_regularizer=regularizer
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        return tf.keras.layers.SpatialDropout2D(config.regularized_spatial_dropout)(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", activation="relu", kernel_regularizer=regularizer
    )(inputs)
    return tf.keras.layers.MaxPooling2D()(x)


def classification_head(
    inputs: tf.Tensor,
    config: ExperimentConfig,
    dropout_rate: float,
    regularized: bool = False,
) -> tf.Tensor:
    """Build a GAP-based binary head, with an optional normalized dense layer."""
    regularizer = tf.keras.regularizers.l2(config.weight_decay)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(inputs)
    x = tf.keras.layers.Dense(
        config.dense_units,
        use_bias=not regularized,
        kernel_regularizer=regularizer,
    )(x)
    if regularized:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)


def build_baseline(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    config: ExperimentConfig | None = None,
    augmented: bool = False,
) -> tf.keras.Model:
    """Build the baseline or augmented CNN with global average pooling."""
    config = config or ExperimentConfig(image_size=input_shape[0])
    inputs = tf.keras.Input(shape=input_shape)
    x = build_augmentation(config)(inputs) if augmented else inputs
    for filters in config.conv_filters:
        x = conv_block(x, filters, config)
    outputs = classification_head(x, config, config.baseline_dropout)
    model = tf.keras.Model(
        inputs, outputs, name="augmented_malaria_cnn" if augmented else "baseline_malaria_cnn"
    )
    return _compile_model(model, config, config.learning_rate)


def build_regularized(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    config: ExperimentConfig | None = None,
) -> tf.keras.Model:
    """Build the from-scratch model with BN, SpatialDropout2D, L2, and GAP."""
    config = config or ExperimentConfig(image_size=input_shape[0])
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for filters in config.conv_filters:
        x = conv_block(x, filters, config, regularized=True)
    outputs = classification_head(x, config, config.regularized_dropout, regularized=True)
    model = tf.keras.Model(inputs, outputs, name="regularized_malaria_cnn")
    return _compile_model(model, config, config.learning_rate)


def build_transfer(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    config: ExperimentConfig | None = None,
) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning with its required [-1, 1] input scale."""
    config = config or ExperimentConfig(image_size=input_shape[0])
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(2.0, offset=-1.0, name="backbone_preprocessing")(inputs)
    backbone = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=input_shape, name="backbone"
    )
    backbone.trainable = False
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(config.transfer_dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_malaria")
    return _compile_model(model, config, config.learning_rate)
