"""Model registry, candidate fitting, and validation-based ensemble ranking."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import DEFAULT_ENSEMBLE_MEMBERS, ExperimentConfig
from .data import make_dataset
from .models import build_baseline, build_regularized, build_transfer
from .train import compute_class_weights, plot_history, train_model, train_transfer_two_phase


LOGGER = logging.getLogger("malaria.experiments")
ModelBuilder = Callable[[ExperimentConfig], tf.keras.Model]


@dataclass(frozen=True)
class ExperimentSpec:
    """Builder and stable checkpoint name for one trainable model."""

    builder: ModelBuilder
    checkpoint_name: str


EXPERIMENTS: dict[str, ExperimentSpec] = {
    "baseline": ExperimentSpec(
        lambda config: build_baseline(config.image_shape, config),
        "baseline_malaria_cnn.keras",
    ),
    "augmented": ExperimentSpec(
        lambda config: build_baseline(config.image_shape, config, augmented=True),
        "augmented_malaria_cnn.keras",
    ),
    "regularized": ExperimentSpec(
        lambda config: build_regularized(config.image_shape, config),
        "regularized_malaria_cnn.keras",
    ),
    "transfer": ExperimentSpec(
        lambda config: build_transfer(config.image_shape, config),
        "transfer_malaria_cnn.keras",
    ),
}
ENSEMBLE_CHECKPOINT = Path("ensemble_malaria_cnn.keras")


def prepare_ensemble_inputs(
    config: ExperimentConfig,
    train_data: tuple[list[str], np.ndarray],
    validation_data: tuple[list[str], np.ndarray],
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float] | None]:
    """Create reusable member datasets and class weights for an ensemble."""
    train_set = make_dataset(
        train_data[0], train_data[1], config, training=True, split_name="train"
    )
    validation_set = make_dataset(
        validation_data[0], validation_data[1], config, split_name="validation"
    )
    weights = compute_class_weights(train_data[1], config)
    return train_set, validation_set, weights


def experiment_spec(name: str) -> ExperimentSpec:
    """Resolve a model name through the single experiment registry."""
    try:
        return EXPERIMENTS[name]
    except KeyError as error:
        raise ValueError(f"Unknown experiment: {name}") from error


def load_or_train(
    name: str,
    spec: ExperimentSpec,
    train_set: tf.data.Dataset,
    val_set: tf.data.Dataset,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
    reuse_checkpoints: bool,
) -> tuple[tf.keras.Model, dict[str, list[float]]]:
    """Load an available checkpoint when requested, otherwise train it."""
    checkpoint = Path(spec.checkpoint_name)
    if reuse_checkpoints and checkpoint.is_file():
        LOGGER.info("Reusing checkpoint %s.", checkpoint)
        model = tf.keras.models.load_model(checkpoint, compile=False)
        validate_input_shape(model, config)
        return model, read_training_history(Path(f"{name}_training.csv"))

    model = spec.builder(config)
    model.summary(print_fn=LOGGER.info)
    history = train_candidate(name, model, train_set, val_set, checkpoint, config, class_weights)
    return tf.keras.models.load_model(checkpoint, compile=False), history


def train_candidate(
    name: str,
    model: tf.keras.Model,
    train_set: tf.data.Dataset,
    val_set: tf.data.Dataset,
    checkpoint: Path,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
) -> dict[str, list[float]]:
    """Train one registered candidate and return its per-epoch history."""
    if name == "transfer":
        return train_transfer_two_phase(
            model, train_set, val_set, checkpoint, config, class_weights
        )
    result = train_model(
        model, train_set, val_set, checkpoint, Path(f"{name}_training.csv"),
        config, class_weights,
    )
    return result.history


def read_training_history(csv_path: Path) -> dict[str, list[float]]:
    """Restore scalar history from a prior CSV when a checkpoint is reused."""
    if not csv_path.is_file():
        return {}
    with csv_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return {}
    return {
        key: [float(row[key]) for row in rows if row.get(key)]
        for key in rows[0]
        if key != "epoch"
    }


def validate_input_shape(model: tf.keras.Model, config: ExperimentConfig) -> None:
    """Reject checkpoint reuse when model resolution differs from current inputs."""
    if tuple(model.input_shape[1:]) != config.image_shape:
        raise ValueError(
            f"Checkpoint input shape {model.input_shape[1:]} does not match "
            f"configured shape {config.image_shape}."
        )


def plot_history_if_available(history: dict[str, list[float]], experiment: str) -> None:
    """Save existing training curves without fabricating a reuse history."""
    if history:
        plot_history(history, Path(f"{experiment}_training_curves.png"))


def rank_candidates(
    names: Sequence[str],
    train_set: tf.data.Dataset,
    val_set: tf.data.Dataset,
    val_labels: np.ndarray,
    config: ExperimentConfig,
    class_weights: dict[int, float] | None,
    reuse_checkpoints: bool,
) -> list[tuple[float, tf.keras.Model, str, np.ndarray]]:
    """Train or load requested members and sort them by validation ROC-AUC."""
    ranked = []
    for name in names:
        spec = experiment_spec(name)
        model, history = load_or_train(
            name, spec, train_set, val_set, config, class_weights, reuse_checkpoints
        )
        plot_history_if_available(history, name)
        probabilities = predict_probabilities(model, val_set)
        auc = float(roc_auc_score(val_labels, probabilities))
        LOGGER.info("%s validation ROC-AUC: %.5f", name, auc)
        ranked.append((auc, model, name, probabilities))
    return sorted(ranked, key=lambda row: row[0], reverse=True)


def build_ensemble(
    members: Sequence[tuple[float, tf.keras.Model, str, np.ndarray]],
    config: ExperimentConfig,
) -> tf.keras.Model:
    """Create an averaged-probability Keras model from ranked members."""
    input_tensor = tf.keras.Input(shape=config.image_shape, name="cell_image")
    outputs = [model(input_tensor, training=False) for _, model, _, _ in members]
    return tf.keras.Model(
        input_tensor,
        tf.keras.layers.Average(name="mean_probability")(outputs),
        name="malaria_probability_ensemble",
    )


def predict_probabilities(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Return one-dimensional positive-class probabilities."""
    return model.predict(dataset, verbose=0).reshape(-1).astype(np.float64)
