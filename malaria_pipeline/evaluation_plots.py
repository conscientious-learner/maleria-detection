"""Plots and image galleries used by held-out test evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .config import CALIBRATION_BINS, CLASS_NAMES, GALLERY_LIMIT


LOGGER = logging.getLogger("malaria.evaluate")


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Save a figure at publication resolution and close it to release memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(matrix: Sequence[Sequence[int]], path: Path, experiment: str) -> None:
    """Render a labeled test confusion matrix."""
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted", ylabel="Actual", title=f"{experiment.title()} confusion matrix",
    )
    for (row, column), count in np.ndenumerate(values):
        axis.text(column, row, str(count), ha="center", va="center")
    save_figure(figure, path)


def plot_roc_curve(labels: np.ndarray, probabilities: np.ndarray, auc: float, path: Path) -> None:
    """Render a held-out ROC curve."""
    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {auc:.4f}")
    axis.plot([0, 1], [0, 1], "--", color="gray")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curve")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    save_figure(figure, path)


def save_calibration_plot(labels: np.ndarray, probabilities: np.ndarray, path: Path) -> None:
    """Save a reliability diagram with configured equal-width probability bins."""
    edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)
    means, frequencies = _calibration_points(labels, probabilities, edges)
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    axis.plot(means, frequencies, "o-", label="Model")
    axis.set(
        xlabel="Mean predicted probability",
        ylabel="Observed positive fraction",
        title="Test-set calibration",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    save_figure(figure, path)


def _calibration_points(
    labels: np.ndarray, probabilities: np.ndarray, edges: np.ndarray
) -> tuple[list[float], list[float]]:
    """Calculate mean confidence and observed frequency for populated bins."""
    means, frequencies = [], []
    for index in range(len(edges) - 1):
        upper = (
            probabilities <= edges[index + 1]
            if index == len(edges) - 2
            else probabilities < edges[index + 1]
        )
        mask = (probabilities >= edges[index]) & upper
        if np.any(mask):
            means.append(float(np.mean(probabilities[mask])))
            frequencies.append(float(np.mean(labels[mask])))
    return means, frequencies


def _misclassified_indices(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, limit: int
) -> np.ndarray:
    """Return false-negative indices first, followed by false positives."""
    predictions = (probabilities >= threshold).astype(np.int32)
    false_negatives = np.flatnonzero((labels == 1) & (predictions == 0))
    false_positives = np.flatnonzero((labels == 0) & (predictions == 1))
    return np.concatenate((false_negatives, false_positives))[:limit]


def save_misclassified_gallery(
    paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
    limit: int = GALLERY_LIMIT,
) -> None:
    """Save a gallery ordered with false negatives before false positives."""
    indices = _misclassified_indices(labels, probabilities, threshold, limit)
    if not len(indices):
        LOGGER.info("No test misclassifications; gallery not created.")
        return
    figure, axes = _gallery_axes(len(indices))
    _draw_gallery(axes, indices, paths, labels, probabilities, threshold)
    save_figure(figure, output_path)


def _gallery_axes(count: int) -> tuple[plt.Figure, np.ndarray]:
    """Allocate gallery axes and hide unused panels."""
    columns = min(6, count)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(columns * 2.3, rows * 2.7), squeeze=False
    )
    for axis in axes.flat:
        axis.axis("off")
    return figure, axes


def _draw_gallery(
    axes: np.ndarray,
    indices: np.ndarray,
    paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    """Draw each selected error with class names and its positive probability."""
    predictions = (probabilities >= threshold).astype(np.int32)
    for axis, index in zip(axes.flat, indices):
        with Image.open(paths[index]) as image:
            axis.imshow(image.convert("RGB"))
        actual, predicted = int(labels[index]), int(predictions[index])
        error = "FN" if actual == 1 else "FP"
        axis.set_title(
            f"{error}: {CLASS_NAMES[actual]} → {CLASS_NAMES[predicted]}\n"
            f"p={probabilities[index]:.3f}",
            fontsize=8,
        )


def save_evaluation_plots(
    metrics: dict[str, Any],
    labels: np.ndarray,
    probabilities: np.ndarray,
    paths: Sequence[str],
    threshold: float,
    output_dir: Path,
    experiment: str,
) -> None:
    """Save confusion, ROC, calibration, and misclassification visualizations."""
    plot_confusion_matrix(
        metrics["confusion_matrix"], output_dir / "confusion_matrix.png", experiment
    )
    if metrics["roc_auc"] is not None:
        plot_roc_curve(labels, probabilities, metrics["roc_auc"], output_dir / "roc_curve.png")
    save_calibration_plot(labels, probabilities, output_dir / "calibration_plot.png")
    save_misclassified_gallery(
        paths, labels, probabilities, threshold, output_dir / "misclassified_gallery.png"
    )
