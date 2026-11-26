"""Reproducible optimizer, callbacks, and transfer-learning phases."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ExperimentConfig


LOGGER = logging.getLogger("malaria.train")


def configure_reproducibility(seed: int, fast: bool = False) -> None:
    """Seed all RNGs and optionally disable deterministic-op execution."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if not fast:
        tf.config.experimental.enable_op_determinism()
    LOGGER.info("Deterministic TensorFlow ops: %s", not fast)
    if tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        LOGGER.info("GPU detected; mixed precision enabled.")
        return
    tf.keras.mixed_precision.set_global_policy("float32")
    LOGGER.info("No GPU detected; using float32.")


def compute_class_weights(
    labels: np.ndarray, config: ExperimentConfig | None = None
) -> dict[int, float] | None:
    """Return balanced weights only when training class frequencies differ materially."""
    config = config or ExperimentConfig()
    counts = np.bincount(labels.astype(np.int32), minlength=2)
    if counts.min() == 0:
        raise ValueError("Both classes must be represented in the training split.")
    if counts.max() / counts.min() < config.class_imbalance_ratio:
        LOGGER.info("Training classes are balanced; class weighting is disabled.")
        return None

    total = int(counts.sum())
    weights = {index: total / (2.0 * int(count)) for index, count in enumerate(counts)}
    LOGGER.info("Applying class weights: %s", weights)
    return weights


def _callbacks(
    checkpoint: Path,
    csv_path: Path,
    config: ExperimentConfig,
    append_csv: bool = False,
) -> list[tf.keras.callbacks.Callback]:
    """Create the required early-stop, checkpoint, LR, and CSV callbacks."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.minimum_learning_rate,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(csv_path), append=append_csv),
    ]


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    checkpoint: Path,
    csv_path: Path,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
    epochs: int | None = None,
    append_csv: bool = False,
) -> tf.keras.callbacks.History:
    """Fit one phase and persist its best validation-loss checkpoint."""
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs if epochs is not None else config.epochs,
        callbacks=_callbacks(checkpoint, csv_path, config, append_csv),
        class_weight=class_weights,
        verbose=2,
    )


def _unfreeze_top_layers(model: tf.keras.Model, config: ExperimentConfig) -> None:
    """Unfreeze only the backbone tail while keeping all BatchNorm layers frozen."""
    backbone = model.get_layer("backbone")
    backbone.trainable = True
    start = max(0, len(backbone.layers) - config.fine_tune_layers)
    for layer in backbone.layers[:start]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def _merge_histories(
    first: tf.keras.callbacks.History, second: tf.keras.callbacks.History
) -> dict[str, list[float]]:
    """Concatenate training logs from the two transfer phases."""
    return {
        key: first.history[key] + second.history.get(key, [])
        for key in first.history
    }


def train_transfer_two_phase(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    checkpoint: Path,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
) -> dict[str, list[float]]:
    """Fit a frozen MobileNetV2, then fine-tune its configured top layers."""
    phase_one_path, phase_two_path, csv_path = _transfer_artifact_paths(checkpoint)
    first_epochs = max(1, config.epochs // 2)
    second_epochs = max(1, config.epochs - first_epochs)
    first = train_model(
        model, train_dataset, validation_dataset, phase_one_path, csv_path,
        config, class_weights, first_epochs,
    )
    second = _fine_tune_transfer(
        model, train_dataset, validation_dataset, phase_two_path, csv_path,
        config, class_weights, second_epochs,
    )
    _save_best_transfer_checkpoint(first, second, phase_one_path, phase_two_path, checkpoint)
    return _merge_histories(first, second)


def _transfer_artifact_paths(checkpoint: Path) -> tuple[Path, Path, Path]:
    """Return temporary phase checkpoints and the stable CSV log path."""
    phase_one = checkpoint.with_name(f".{checkpoint.stem}_phase1.keras")
    phase_two = checkpoint.with_name(f".{checkpoint.stem}_phase2.keras")
    csv_path = checkpoint.with_name(f"{checkpoint.stem.removesuffix('_malaria_cnn')}_training.csv")
    return phase_one, phase_two, csv_path


def _fine_tune_transfer(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    checkpoint: Path,
    csv_path: Path,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
    epochs: int,
) -> tf.keras.callbacks.History:
    """Unfreeze the configured backbone tail and fit at the fine-tuning rate."""
    _unfreeze_top_layers(model, config)
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=config.fine_tune_learning_rate,
            weight_decay=config.weight_decay,
            beta_1=config.adam_beta_1,
            beta_2=config.adam_beta_2,
        ),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return train_model(
        model, train_dataset, validation_dataset, checkpoint, csv_path,
        config, class_weights, epochs, append_csv=True,
    )


def _save_best_transfer_checkpoint(
    first: tf.keras.callbacks.History,
    second: tf.keras.callbacks.History,
    first_path: Path,
    second_path: Path,
    checkpoint: Path,
) -> None:
    """Promote the phase checkpoint with the lower best validation loss."""
    first_loss = min(first.history["val_loss"])
    second_loss = min(second.history["val_loss"])
    chosen = first_path if first_loss <= second_loss else second_path
    tf.keras.models.load_model(chosen).save(checkpoint)
    _remove_phase_checkpoints(first_path, second_path)


def _remove_phase_checkpoints(*paths: Path) -> None:
    """Delete only the two temporary phase checkpoints after promotion."""
    for path in paths:
        if path.exists():
            path.unlink()


def plot_history(history: dict[str, list[float]], output_path: Path) -> None:
    """Save train/validation loss and accuracy curves."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, key, title in (
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ):
        if key in history:
            axis.plot(history[key], label="Train")
        if f"val_{key}" in history:
            axis.plot(history[f"val_{key}"], label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
