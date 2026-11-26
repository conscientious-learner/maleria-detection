"""Compatibility entry point for the modular malaria-cell classification pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from malaria_pipeline.cli import main, parse_args
from malaria_pipeline.config import (
    CLASS_NAMES,
    DATASET_DIR,
    ROOT,
    SUPPORTED_EXTENSIONS,
    ExperimentConfig,
)
from malaria_pipeline.data import (
    AUTOTUNE,
    _decode,
    _resize_and_scale,
    collect_image_paths,
    find_near_duplicates,
    make_dataset,
    split_paths,
)
from malaria_pipeline.duplicates import hamming_distance, perceptual_hash
from malaria_pipeline.evaluate import (
    bootstrap_confidence_intervals,
    calculate_metrics,
    choose_threshold,
    evaluate_test_set,
)
from malaria_pipeline.evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    save_calibration_plot,
    save_figure,
    save_misclassified_gallery,
)
from malaria_pipeline.experiments import (
    ENSEMBLE_CHECKPOINT,
    EXPERIMENTS,
    run_ensemble,
    run_single_experiment,
)
from malaria_pipeline.experiment_finalize import _json_safe
from malaria_pipeline.gradcam import grad_cam, save_gradcam_examples
from malaria_pipeline.models import (
    _compile_model,
    build_augmentation,
    build_baseline,
    build_regularized,
    build_transfer,
)
from malaria_pipeline.train import (
    _callbacks,
    compute_class_weights,
    configure_reproducibility,
    plot_history,
    train_model,
    train_transfer_two_phase,
)


LOGGER = logging.getLogger("malaria")


def _make_model(experiment: str, config: ExperimentConfig) -> tf.keras.Model:
    """Keep the original private model factory available to importing scripts."""
    try:
        return EXPERIMENTS[experiment].builder(config)
    except KeyError as error:
        raise ValueError(f"Unsupported single-model experiment: {experiment}") from error


def _model_checkpoint(experiment: str) -> Path:
    """Return a registered experiment's stable checkpoint path."""
    try:
        return Path(EXPERIMENTS[experiment].checkpoint_name)
    except KeyError as error:
        raise ValueError(f"Unsupported single-model experiment: {experiment}") from error


def _predict_probabilities(
    model: tf.keras.Model, dataset: tf.data.Dataset
) -> np.ndarray:
    """Return the model's positive-class probabilities as a flat float array."""
    return model.predict(dataset, verbose=0).reshape(-1).astype(np.float64)


if __name__ == "__main__":
    main()
