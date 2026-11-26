"""Threshold selection, test metrics, confidence intervals, and plots."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import ExperimentConfig
from .evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_calibration_plot,
    save_figure,
    save_misclassified_gallery,
    save_evaluation_plots,
)


LOGGER = logging.getLogger("malaria.evaluate")
_CI_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")


def choose_threshold(
    labels: np.ndarray, probabilities: np.ndarray, target_recall: float
) -> tuple[float, dict[str, float]]:
    """Choose maximum precision at or above the requested validation recall."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    points = list(zip(thresholds, precision[:-1], recall[:-1]))
    if not np.any(thresholds == 0.0):
        points.append((0.0, float(np.mean(labels)), 1.0))
    if not np.any(thresholds == 1.0):
        points.append((1.0, 0.0, 0.0))
    eligible = [
        (float(prec), float(threshold), float(rec))
        for threshold, prec, rec in points
        if rec >= target_recall
    ]
    if not eligible:
        threshold = 0.0
        return threshold, _threshold_metrics(labels, probabilities, threshold)
    best_precision, best_threshold, best_recall = max(eligible, key=lambda row: (row[0], row[1]))
    return best_threshold, {"precision": best_precision, "recall": best_recall}


def _threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    """Calculate validation precision and recall at a specified threshold."""
    predictions = (probabilities >= threshold).astype(np.int32)
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def calculate_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute binary metrics using Parasitized as the positive class."""
    predictions = (probabilities >= threshold).astype(np.int32)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if len(np.unique(labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["pr_auc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
    return metrics


def _bootstrap_sample(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | None]:
    """Calculate a bootstrap row with direct counts and ranking-score metrics."""
    predictions = probabilities >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(positive & predictions))
    fp = int(np.count_nonzero(~positive & predictions))
    fn = int(np.count_nonzero(positive & ~predictions))
    tn = len(labels) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result: dict[str, float | None] = {
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "specificity": tn / max(tn + fp, 1),
        "f1": f1,
        "roc_auc": None,
        "pr_auc": None,
    }
    if np.any(positive) and np.any(~positive):
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["pr_auc"] = float(average_precision_score(labels, probabilities))
    return result


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    samples: int,
    seed: int,
) -> dict[str, list[float] | None]:
    """Calculate the same seeded percentile CIs with fewer sklearn metric calls."""
    rng = np.random.default_rng(seed)
    observed: dict[str, list[float]] = {name: [] for name in _CI_METRICS}
    for _ in range(samples):
        indices = rng.integers(0, len(labels), size=len(labels))
        values = _bootstrap_sample(labels[indices], probabilities[indices], threshold)
        for name, value in values.items():
            if value is not None and np.isfinite(value):
                observed[name].append(float(value))
    return _percentile_intervals(observed)


def _percentile_intervals(
    observed: dict[str, list[float]]
) -> dict[str, list[float] | None]:
    """Format bootstrap samples as 2.5/97.5 percentile endpoints."""
    return {
        name: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        if values else None
        for name, values in observed.items()
    }


def evaluate_test_set(
    model: tf.keras.Model,
    test_paths: Sequence[str],
    test_dataset: tf.data.Dataset,
    test_labels: np.ndarray,
    threshold: float,
    config: ExperimentConfig,
    experiment: str,
    output_dir: Path,
    gradcam_model: tf.keras.Model | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Evaluate once and save the unchanged metric schema plus diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(test_dataset, verbose=0).reshape(-1).astype(np.float64)
    metrics = _test_metrics_and_plots(
        test_paths, test_labels, probabilities, threshold, config, experiment, output_dir
    )
    _save_gradcam_examples(
        gradcam_model if gradcam_model is not None else model,
        test_paths,
        test_labels,
        probabilities,
        threshold,
        config.image_size,
        output_dir / "gradcam",
    )
    LOGGER.info("Test metrics (%s): %s", experiment, metrics)
    return metrics, probabilities


def _test_metrics_and_plots(
    paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    config: ExperimentConfig,
    experiment: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Calculate held-out scores, confidence intervals, and diagnostic plots."""
    metrics = calculate_metrics(labels, probabilities, threshold)
    metrics["threshold"] = float(threshold)
    metrics["confidence_intervals_95"] = bootstrap_confidence_intervals(
        labels, probabilities, threshold, config.bootstrap_samples, config.seed
    )
    save_evaluation_plots(metrics, labels, probabilities, paths, threshold, output_dir, experiment)
    return metrics


def _save_gradcam_examples(
    model: tf.keras.Model,
    test_paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    image_size: int,
    output_dir: Path,
) -> None:
    """Create the three requested Grad-CAM categories without importing eagerly."""
    from .gradcam import save_gradcam_examples

    save_gradcam_examples(
        model,
        test_paths,
        labels,
        probabilities,
        threshold,
        image_size,
        output_dir,
    )
