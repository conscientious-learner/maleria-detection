"""Reproducible stratified and duplicate-group-aware data splitting."""
from __future__ import annotations

import logging

import numpy as np

from .config import GROUP_SPLIT_ATTEMPTS


LOGGER = logging.getLogger("malaria.data")
SplitData = tuple[list[str], np.ndarray]


def _best_group_holdout(
    indices: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a deterministic grouped holdout close to size and class targets."""
    from sklearn.model_selection import GroupShuffleSplit

    splitter = GroupShuffleSplit(
        n_splits=GROUP_SPLIT_ATTEMPTS, test_size=fraction, random_state=seed
    )
    positive_rate = float(np.mean(labels[indices]))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for keep_local, held_local in splitter.split(indices, labels[indices], groups[indices]):
        keep, held = indices[keep_local], indices[held_local]
        if not len(keep) or not len(held):
            continue
        size_error = abs(len(held) / len(indices) - fraction)
        class_error = abs(float(np.mean(labels[held])) - positive_rate)
        score = size_error + class_error
        if best is None or score < best[0]:
            best = (score, keep, held)
    if best is None:
        raise ValueError("Unable to create non-empty group-aware dataset splits.")
    return best[1], best[2]


def _stratified_split(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the existing two-stage 70/15/15 stratified split."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train, remaining, _, remaining_labels = train_test_split(
        indices, labels, test_size=0.30, random_state=seed, stratify=labels
    )
    validation, test = train_test_split(
        remaining, test_size=0.50, random_state=seed, stratify=remaining_labels
    )
    return train, validation, test


def _split_indices(
    labels: np.ndarray, groups: np.ndarray | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the original stratified split or duplicate-group split."""
    if groups is None or len(np.unique(groups)) == len(labels):
        return _stratified_split(labels, seed)
    indices = np.arange(len(labels))
    train_val, test = _best_group_holdout(indices, labels, groups, 0.15, seed)
    train, validation = _best_group_holdout(
        train_val, labels, groups, 0.15 / 0.85, seed + 1
    )
    _assert_no_group_leakage(groups, train, validation, test)
    return train, validation, test


def _assert_no_group_leakage(
    groups: np.ndarray, train: np.ndarray, validation: np.ndarray, test: np.ndarray
) -> None:
    """Fail loudly if a duplicate-connected component crosses split boundaries."""
    train_groups, validation_groups, test_groups = (
        set(groups[indices]) for indices in (train, validation, test)
    )
    overlaps = (
        train_groups & validation_groups,
        train_groups & test_groups,
        validation_groups & test_groups,
    )
    if any(overlaps):
        raise RuntimeError("Duplicate group leakage detected after split.")
    LOGGER.info("Duplicate groups are isolated to a single data split.")


def _log_split(name: str, labels: np.ndarray) -> None:
    """Log the class distribution for one split."""
    counts = np.bincount(labels, minlength=2)
    LOGGER.info(
        "%s: %d total | Uninfected=%d | Parasitized=%d",
        name, len(labels), counts[0], counts[1],
    )


def split_paths(
    paths: list[str],
    labels: np.ndarray,
    groups: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[SplitData, SplitData, SplitData]:
    """Create reproducible train, validation, and test splits."""
    train_indices, val_indices, test_indices = _split_indices(labels, groups, seed)
    result = []
    for name, indices in (
        ("Train", train_indices), ("Validation", val_indices), ("Test", test_indices)
    ):
        split_labels = labels[indices].astype(np.int32)
        _log_split(name, split_labels)
        result.append(([paths[index] for index in indices], split_labels))
    if groups is not None and len(np.unique(groups)) != len(labels):
        LOGGER.info("Grouped split sizes may vary slightly to keep duplicates together.")
    return result[0], result[1], result[2]
