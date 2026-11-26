"""Shared validation thresholding, test evaluation, promotion, and metrics output."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .data import make_dataset
from .evaluate import choose_threshold, evaluate_test_set


LOGGER = logging.getLogger("malaria.experiments")
Member = tuple[float, tf.keras.Model, str, np.ndarray]


def _finalize_experiment(
    experiment: str,
    model: tf.keras.Model,
    checkpoint: Path,
    config: ExperimentConfig,
    validation_data: tuple[list[str], np.ndarray],
    test_data: tuple[list[str], np.ndarray],
    history: dict[str, list[float]],
    promote_if_better: bool,
    members: Sequence[Member] = (),
    gradcam_model: tf.keras.Model | None = None,
) -> None:
    """Select threshold on validation, evaluate test once, and save run metadata."""
    threshold, threshold_metrics, candidate_auc = _validation_operating_point(
        model, validation_data, config
    )
    promote = _should_promote(candidate_auc, validation_data, config, promote_if_better)
    if promote:
        model.save("final_malaria_model.keras")

    metrics = _evaluate_final(experiment, model, test_data, threshold, config, gradcam_model)
    metrics["validation_threshold_metrics"] = threshold_metrics
    if experiment != "ensemble":
        _add_history_metrics(metrics, history)
    metrics["validation_threshold_target_recall"] = config.target_recall
    metrics["image_size"] = config.image_size
    if members:
        metrics["members"] = _member_metadata(members)
    _save_metrics(metrics, experiment, config, checkpoint, promote)


def _validation_operating_point(
    model: tf.keras.Model,
    validation_data: tuple[list[str], np.ndarray],
    config: ExperimentConfig,
) -> tuple[float, dict[str, float], float]:
    """Choose a recall-constrained threshold and calculate validation ROC-AUC."""
    paths, labels = validation_data
    dataset = make_dataset(paths, labels, config, split_name="validation")
    probabilities = _predict(model, dataset)
    threshold, threshold_metrics = choose_threshold(labels, probabilities, config.target_recall)
    return threshold, threshold_metrics, float(roc_auc_score(labels, probabilities))


def _evaluate_final(
    experiment: str,
    model: tf.keras.Model,
    test_data: tuple[list[str], np.ndarray],
    threshold: float,
    config: ExperimentConfig,
    gradcam_model: tf.keras.Model | None,
) -> dict[str, Any]:
    """Build the held-out input pipeline immediately before one test evaluation."""
    test_set = make_dataset(test_data[0], test_data[1], config, split_name="test")
    output_dir = Path("artifacts") / experiment
    metrics, _ = evaluate_test_set(
        model,
        test_data[0],
        test_set,
        test_data[1],
        threshold,
        config,
        experiment,
        output_dir,
        gradcam_model=gradcam_model,
    )
    return metrics


def _add_history_metrics(metrics: dict[str, Any], history: dict[str, list[float]]) -> None:
    """Add the existing best-epoch training/validation loss fields."""
    validation_losses = history.get("val_loss", [])
    training_losses = history.get("loss", [])
    if not validation_losses or len(training_losses) != len(validation_losses):
        metrics["best_validation_loss"] = None
        metrics["training_loss_at_best_validation"] = None
        metrics["train_validation_loss_gap"] = None
        return
    best_epoch = int(np.argmin(validation_losses))
    metrics["best_validation_loss"] = float(validation_losses[best_epoch])
    metrics["training_loss_at_best_validation"] = float(training_losses[best_epoch])
    metrics["train_validation_loss_gap"] = float(
        validation_losses[best_epoch] - training_losses[best_epoch]
    )


def _member_metadata(members: Sequence[Member]) -> list[dict[str, Any]]:
    """Serialize ensemble member names and validation scores."""
    return [
        {"experiment": name, "validation_roc_auc": auc}
        for auc, _, name, _ in members
    ]


def _predict(model: tf.keras.Model, dataset: tf.data.Dataset) -> np.ndarray:
    """Return one-dimensional positive-class probabilities."""
    return model.predict(dataset, verbose=0).reshape(-1).astype(np.float64)


def _current_validation_auc(
    validation_data: tuple[list[str], np.ndarray], config: ExperimentConfig
) -> float | None:
    """Measure an existing promoted model on current validation data only."""
    checkpoint = Path("final_malaria_model.keras")
    if not checkpoint.is_file():
        return None
    model = tf.keras.models.load_model(checkpoint, compile=False)
    current_config = replace(config, image_size=int(model.input_shape[1]))
    dataset = make_dataset(
        validation_data[0],
        validation_data[1],
        current_config,
        split_name="promotion validation",
    )
    return float(roc_auc_score(validation_data[1], _predict(model, dataset)))


def _should_promote(
    candidate_auc: float,
    validation_data: tuple[list[str], np.ndarray],
    config: ExperimentConfig,
    promote_if_better: bool,
) -> bool:
    """Apply default promotion or require a strict validation-AUC improvement."""
    if not promote_if_better:
        return True
    current_auc = _current_validation_auc(validation_data, config)
    return current_auc is None or candidate_auc > current_auc


def _save_metrics(
    metrics: dict[str, Any],
    experiment: str,
    config: ExperimentConfig,
    checkpoint: Path,
    promoted: bool,
) -> None:
    """Serialize once and write the experiment and promoted metadata payloads."""
    payload = {
        "experiment": experiment,
        "model": str(checkpoint),
        "config": asdict(config),
        **metrics,
    }
    serialized = json.dumps(_json_safe(payload), indent=2)
    Path(f"{experiment}_metrics.json").write_text(serialized, encoding="utf-8")
    if promoted:
        Path("final_malaria_model_metrics.json").write_text(serialized, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    """Convert NumPy scalars, paths, and tuples to JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
