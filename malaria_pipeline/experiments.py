"""Single-model and ranked-ensemble experiment orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .config import DEFAULT_ENSEMBLE_MEMBERS, ExperimentConfig
from .data import make_dataset
from .experiment_candidates import (
    ENSEMBLE_CHECKPOINT,
    EXPERIMENTS,
    build_ensemble,
    experiment_spec,
    load_or_train,
    plot_history_if_available,
    prepare_ensemble_inputs,
    rank_candidates,
)
from .experiment_finalize import _finalize_experiment
from .train import compute_class_weights


def run_single_experiment(
    experiment: str,
    config: ExperimentConfig,
    train_data: tuple[list[str], np.ndarray],
    validation_data: tuple[list[str], np.ndarray],
    test_data: tuple[list[str], np.ndarray],
    *,
    reuse_checkpoints: bool = False,
    promote_if_better: bool = False,
) -> None:
    """Train or reuse one model, then evaluate it through the shared finalizer."""
    spec = experiment_spec(experiment)
    train_set, validation_set, class_weights = _prepare_single_inputs(
        config, train_data, validation_data
    )
    model, history = load_or_train(
        experiment, spec, train_set, validation_set, config, class_weights, reuse_checkpoints
    )
    plot_history_if_available(history, experiment)
    _finalize_experiment(
        experiment,
        model,
        Path(spec.checkpoint_name),
        config,
        validation_data,
        test_data,
        history,
        promote_if_better,
    )


def run_ensemble(
    config: ExperimentConfig,
    train_data: tuple[list[str], np.ndarray],
    validation_data: tuple[list[str], np.ndarray],
    test_data: tuple[list[str], np.ndarray],
    *,
    ensemble_members: Sequence[str] = DEFAULT_ENSEMBLE_MEMBERS,
    reuse_checkpoints: bool = False,
    promote_if_better: bool = False,
) -> None:
    """Rank requested candidates on validation ROC-AUC and average the best three."""
    if len(set(ensemble_members)) < 2:
        raise ValueError("An ensemble needs at least two distinct candidate models.")
    train_set, validation_set, weights = prepare_ensemble_inputs(
        config, train_data, validation_data
    )
    ranked = rank_candidates(
        ensemble_members,
        train_set,
        validation_set,
        validation_data[1],
        config,
        weights,
        reuse_checkpoints,
    )
    _save_ensemble_result(
        ranked[:3], config, validation_data, test_data, promote_if_better
    )


def _prepare_single_inputs(
    config: ExperimentConfig,
    train_data: tuple[list[str], np.ndarray],
    validation_data: tuple[list[str], np.ndarray],
) -> tuple[tf.data.Dataset, tf.data.Dataset, dict[int, float] | None]:
    """Create training/validation datasets and the applicable class weights."""
    train_set = make_dataset(
        train_data[0], train_data[1], config, training=True, split_name="train"
    )
    validation_set = make_dataset(
        validation_data[0], validation_data[1], config, split_name="validation"
    )
    return train_set, validation_set, compute_class_weights(train_data[1], config)


def _save_ensemble_result(
    selected: Sequence[tuple[float, tf.keras.Model, str, np.ndarray]],
    config: ExperimentConfig,
    validation_data: tuple[list[str], np.ndarray],
    test_data: tuple[list[str], np.ndarray],
    promote_if_better: bool,
) -> None:
    """Persist the averaged model, curves, metrics, and test diagnostics."""
    ensemble = build_ensemble(selected, config)
    ensemble.save(ENSEMBLE_CHECKPOINT)
    _finalize_experiment(
        "ensemble",
        ensemble,
        ENSEMBLE_CHECKPOINT,
        config,
        validation_data,
        test_data,
        {},
        promote_if_better,
        selected,
        gradcam_model=selected[0][1],
    )
