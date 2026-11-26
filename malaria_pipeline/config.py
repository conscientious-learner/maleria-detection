"""Configuration and shared constants for malaria classification."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "archive"
CLASS_NAMES = ("Uninfected", "Parasitized")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
IMAGE_CHANNELS = 3
AUTOTUNE_NAME = "AUTOTUNE"
CACHE_FILE_NAME = ".malaria_image_cache.json"
SHUFFLE_BUFFER = 10_000
GALLERY_LIMIT = 48
GRADCAM_SAMPLES_PER_CATEGORY = 2
HASH_PROGRESS_INTERVAL = 5_000
GROUP_SPLIT_ATTEMPTS = 128
CALIBRATION_BINS = 10
FINE_TUNING_PHASES = 2
DEFAULT_ENSEMBLE_MEMBERS = ("baseline", "augmented", "regularized")
EXPERIMENT_NAMES = ("baseline", "augmented", "regularized", "transfer")


@dataclass(frozen=True)
class ExperimentConfig:
    """Model, data, and optimization parameters for one reproducible run."""

    image_size: int = 128
    batch_size: int = 32
    epochs: int = 25
    learning_rate: float = 1e-3
    seed: int = 42
    label_smoothing: float = 0.05
    weight_decay: float = 1e-4
    adam_beta_1: float = 0.9
    adam_beta_2: float = 0.999
    target_recall: float = 0.98
    bootstrap_samples: int = 1000
    duplicate_hash_distance: int = 2
    class_imbalance_ratio: float = 1.05
    cache: bool = True
    conv_filters: tuple[int, ...] = (32, 64, 128)
    dense_units: int = 128
    baseline_dropout: float = 0.35
    regularized_spatial_dropout: float = 0.18
    regularized_dropout: float = 0.40
    transfer_dropout: float = 0.30
    rotation_factor: float = 0.04
    zoom_factor: float = 0.06
    translation_factor: float = 0.04
    brightness_factor: float = 0.08
    contrast_factor: float = 0.10
    early_stopping_patience: int = 5
    reduce_lr_factor: float = 0.2
    reduce_lr_patience: int = 2
    minimum_learning_rate: float = 1e-7
    fine_tune_layers: int = 30
    fine_tune_learning_rate: float = 1e-5

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Return configured RGB image dimensions."""
        return (self.image_size, self.image_size, IMAGE_CHANNELS)
