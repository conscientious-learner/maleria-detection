"""Command-line parsing and top-level experiment dispatch."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .config import (
    DATASET_DIR,
    DEFAULT_ENSEMBLE_MEMBERS,
    EXPERIMENT_NAMES,
    ExperimentConfig,
)
from .data import collect_image_paths, find_near_duplicates, split_paths
from .experiments import ENSEMBLE_CHECKPOINT, EXPERIMENTS, run_ensemble, run_single_experiment
from .gradcam import grad_cam
from .train import configure_reproducibility


LOGGER = logging.getLogger("malaria")
_ALL_EXPERIMENTS = (*EXPERIMENT_NAMES, "ensemble")


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, ExperimentConfig]:
    """Parse CLI options and build config using dataclass defaults."""
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser, defaults)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args, _config_from_args(args)


def _add_arguments(parser: argparse.ArgumentParser, defaults: ExperimentConfig) -> None:
    """Register stable and optional experiment flags."""
    parser.add_argument("--experiment", choices=_ALL_EXPERIMENTS, default="baseline")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.learning_rate)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--target-recall", type=float, default=defaults.target_recall)
    parser.add_argument("--limit-per-class", type=int)
    parser.add_argument("--gradcam-image", type=Path)
    parser.add_argument("--fast", action="store_true", help="Disable deterministic TensorFlow ops")
    parser.add_argument("--reuse-checkpoints", action="store_true")
    parser.add_argument("--promote-if-better", action="store_true")
    parser.add_argument(
        "--ensemble-members",
        nargs="+",
        choices=EXPERIMENT_NAMES,
        default=DEFAULT_ENSEMBLE_MEMBERS,
    )


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject invalid ranges before scanning or training data."""
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.image_size < 32:
        parser.error("--epochs, --batch-size, --lr must be positive; --image-size must be >= 32")
    if not 0.0 <= args.target_recall <= 1.0:
        parser.error("--target-recall must be between 0 and 1 inclusive")
    if args.limit_per_class is not None and args.limit_per_class < 4:
        parser.error("--limit-per-class must be at least 4 for stratified 70/15/15 splits")
    if args.experiment == "ensemble" and len(set(args.ensemble_members)) < 2:
        parser.error("--ensemble-members must contain at least two distinct models")
    if args.gradcam_image is not None and not args.gradcam_image.is_file():
        parser.error(f"Grad-CAM image not found: {args.gradcam_image}")


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Copy tunable CLI fields over the central experiment defaults."""
    return ExperimentConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        target_recall=args.target_recall,
    )


def _write_duplicate_report(
    pairs: list[dict[str, object]], max_distance: int, output_path: Path
) -> None:
    """Write the stable duplicate-report schema for reproducibility."""
    payload = {
        "hash": "64-bit DCT pHash",
        "max_hamming_distance": max_distance,
        "duplicate_pair_count": len(pairs),
        "pairs": pairs,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if pairs:
        LOGGER.warning(
            "Found %d pHash near-duplicate pairs; connected groups stay in one split. "
            "Details: %s",
            len(pairs), output_path,
        )
        return
    LOGGER.info("No near-duplicate images found at the configured pHash distance.")


def _run_experiment(
    args: argparse.Namespace,
    config: ExperimentConfig,
    train_data: tuple[list[str], np.ndarray],
    validation_data: tuple[list[str], np.ndarray],
    test_data: tuple[list[str], np.ndarray],
) -> None:
    """Dispatch an experiment with its optional checkpoint and promotion policies."""
    common = {
        "reuse_checkpoints": args.reuse_checkpoints,
        "promote_if_better": args.promote_if_better,
    }
    if args.experiment == "ensemble":
        run_ensemble(
            config,
            train_data,
            validation_data,
            test_data,
            ensemble_members=args.ensemble_members,
            **common,
        )
        return
    run_single_experiment(
        args.experiment, config, train_data, validation_data, test_data, **common
    )


def _save_requested_gradcam(image_path: Path, experiment: str, image_size: int) -> None:
    """Save the optional user-selected Grad-CAM without changing test metrics."""
    checkpoint = ENSEMBLE_CHECKPOINT if experiment == "ensemble" else Path(
        EXPERIMENTS[experiment].checkpoint_name
    )
    model = tf.keras.models.load_model(checkpoint, compile=False)
    output_path = Path("artifacts") / experiment / "gradcam" / "user_supplied.png"
    grad_cam(model, image_path, image_size, output_path)
    LOGGER.info("Saved requested Grad-CAM overlay: %s", output_path)


def main(argv: Sequence[str] | None = None) -> None:
    """Run duplicate audit, construct the fixed split, then fit/evaluate once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args, config = parse_args(argv)
    configure_reproducibility(config.seed, fast=args.fast)
    paths, labels = collect_image_paths(args.dataset, args.limit_per_class)
    duplicates, groups = find_near_duplicates(paths, config.duplicate_hash_distance)
    _write_duplicate_report(
        duplicates, config.duplicate_hash_distance, Path("duplicate_report.json")
    )
    splits = split_paths(paths, labels, groups=groups, seed=config.seed)
    _run_experiment(args, config, *splits)
    if args.gradcam_image is not None:
        _save_requested_gradcam(args.gradcam_image, args.experiment, config.image_size)
