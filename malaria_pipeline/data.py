"""Image loading, group-aware splits, and cached TensorFlow input datasets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from .config import (
    AUTOTUNE_NAME,
    CACHE_FILE_NAME,
    DATASET_DIR,
    IMAGE_CHANNELS,
    SHUFFLE_BUFFER,
    SUPPORTED_EXTENSIONS,
    ExperimentConfig,
)
from .duplicates import cached_image_records, find_near_duplicates as _find_near_duplicates
from .splitting import split_paths


LOGGER = logging.getLogger("malaria.data")
AUTOTUNE = tf.data.AUTOTUNE
_HASHES_BY_PATH: dict[str, int] = {}


def collect_image_paths(
    dataset_dir: Path = DATASET_DIR,
    limit_per_class: int | None = None,
    cache_path: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load verified class-folder images and reuse unchanged cached pHashes."""
    if limit_per_class is not None and limit_per_class < 1:
        raise ValueError("limit_per_class must be positive.")
    candidates = _collect_candidates(dataset_dir, limit_per_class)
    cache_path = cache_path or Path.cwd() / CACHE_FILE_NAME
    records = cached_image_records([path for path, _ in candidates], cache_path)
    paths, labels = _select_verified(candidates, records, limit_per_class)
    if not paths:
        raise ValueError(f"No valid images found under {dataset_dir}")

    _HASHES_BY_PATH.clear()
    _HASHES_BY_PATH.update({path: int(records[path]["phash"]) for path in paths})
    LOGGER.info("Loaded %d verified cell images.", len(paths))
    return paths, np.asarray(labels, dtype=np.int32)


def _collect_candidates(
    dataset_dir: Path, limit_per_class: int | None
) -> list[tuple[str, int]]:
    """Return sorted supported image candidates from the two class folders."""
    candidates: list[tuple[str, int]] = []
    for folder, label in (("Uninfected", 0), ("Parasitized", 1)):
        class_dir = dataset_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if limit_per_class is not None:
            files = files[:limit_per_class]
        candidates.extend((str(path), label) for path in files)
    return candidates


def _select_verified(
    candidates: Sequence[tuple[str, int]],
    records: dict[str, dict[str, object]],
    limit_per_class: int | None,
) -> tuple[list[str], list[int]]:
    """Preserve sorted class order while filtering unreadable files."""
    paths: list[str] = []
    labels: list[int] = []
    counts = [0, 0]
    for source, label in candidates:
        record = records.get(str(Path(source).resolve()))
        if record is None or not record["valid"]:
            LOGGER.warning("Skipping unreadable or inaccessible image: %s", source)
            continue
        if limit_per_class is not None and counts[label] >= limit_per_class:
            continue
        paths.append(str(Path(source).resolve()))
        labels.append(label)
        counts[label] += 1
    return paths, labels


def find_near_duplicates(
    paths: Sequence[str], max_distance: int
) -> tuple[list[dict[str, object]], np.ndarray]:
    """Return pHash matches using hashes populated during image collection."""
    return _find_near_duplicates(paths, max_distance, _HASHES_BY_PATH or None)


def _decode(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode a file to RGB uint8 pixels."""
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=IMAGE_CHANNELS, expand_animations=False
    )
    image.set_shape([None, None, IMAGE_CHANNELS])
    return image, label


def _decode_resize_uint8(
    path: tf.Tensor, label: tf.Tensor, image_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode and resize once before caching compact uint8 tensors."""
    image, label = _decode(path, label)
    resized = tf.image.resize(image, [image_size, image_size], method="bilinear")
    return tf.cast(tf.round(resized), tf.uint8), label


def _scale_cached(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Convert cached uint8 pixels to model float input."""
    return tf.cast(image, tf.float32) / 255.0, label


def _resize_and_scale(
    image: tf.Tensor, label: tf.Tensor, image_size: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Compatibility helper for direct image resize and normalization."""
    image = tf.image.resize(image, [image_size, image_size], method="bilinear")
    return tf.cast(image, tf.float32) / 255.0, label


def _log_cache_memory(path_count: int, config: ExperimentConfig, split_name: str) -> None:
    """Report resized uint8 cache usage and the equivalent float32 allocation."""
    bytes_per_image = config.image_size * config.image_size * IMAGE_CHANNELS
    uint8_mib = path_count * bytes_per_image / (1024**2)
    LOGGER.info(
        "%s cache estimate: %.1f MiB uint8 (%.1f MiB float32).",
        split_name, uint8_mib, uint8_mib * 4,
    )


def make_dataset(
    paths: Sequence[str],
    labels: np.ndarray,
    config: ExperimentConfig,
    training: bool = False,
    split_name: str = "dataset",
) -> tf.data.Dataset:
    """Resize once to uint8, cache, scale, batch, and prefetch the image dataset."""
    _log_cache_memory(len(paths), config, split_name)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    dataset = dataset.map(
        lambda path, label: _decode_resize_uint8(path, label, config.image_size),
        num_parallel_calls=AUTOTUNE,
    )
    if config.cache:
        dataset = dataset.cache()
    if training:
        dataset = dataset.shuffle(
            min(len(paths), SHUFFLE_BUFFER),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(_scale_cached, num_parallel_calls=AUTOTUNE)
    return dataset.batch(config.batch_size).prefetch(AUTOTUNE)
