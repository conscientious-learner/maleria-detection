"""Predict whether a blood-cell image is parasitized or uninfected."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger("malaria.predict")
DEFAULT_MODEL = Path("final_malaria_model.keras")
DEFAULT_METADATA = Path("final_malaria_model_metrics.json")
CLASS_NAMES = ("Uninfected", "Parasitized")


def _read_metadata(model_path: Path, explicit_path: Path | None) -> dict[str, Any]:
    """Load threshold and preprocessing metadata associated with a saved model."""
    candidates = [explicit_path] if explicit_path else []
    if model_path.resolve() == DEFAULT_MODEL.resolve():
        candidates.append(DEFAULT_METADATA)
    elif model_path.stem.endswith("_malaria_cnn"):
        experiment = model_path.stem.removesuffix("_malaria_cnn")
        candidates.append(Path(f"{experiment}_metrics.json"))
    for path in candidates:
        if path is not None and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "No model metadata found. Train the model first or pass --metadata with an "
        "experiment metrics JSON containing its validation-selected threshold."
    )


def preprocess_image(image_path: Path, image_size: int) -> np.ndarray:
    """Resize RGB pixels and scale to [0, 1], matching the training input pipeline."""
    image = tf.keras.utils.load_img(
        image_path, target_size=(image_size, image_size), color_mode="rgb"
    )
    array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
    return array[None, ...]


def _prompt_for_image_path() -> Path | None:
    """Ask for an image path when the command is run without a positional image."""
    try:
        value = input("Enter an image path (leave blank to cancel): ").strip()
    except EOFError:
        LOGGER.info("No interactive input received; no prediction was made.")
        return None
    if not value:
        LOGGER.info("No image path entered; no prediction was made.")
        return None
    if value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return Path(value)


def main() -> None:
    """Load a Keras model and classify one cell image using the validation threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image_path", type=Path, nargs="?", help="Path to a cell image; prompted if omitted"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved .keras model path")
    parser.add_argument(
        "--metadata", type=Path, help="Metrics JSON containing image size and selected threshold"
    )
    parser.add_argument(
        "--threshold", type=float, help="Override the validation-selected threshold"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.model.is_file():
        parser.error(f"Model file does not exist: {args.model}. Run model.py first.")
    image_path = args.image_path or _prompt_for_image_path()
    if image_path is None:
        return
    if not image_path.is_file():
        parser.error(f"Image file does not exist: {image_path}")

    metadata = _read_metadata(args.model, args.metadata)
    image_size = int(metadata.get("image_size", metadata.get("config", {}).get("image_size", 128)))
    threshold = float(metadata["threshold"] if args.threshold is None else args.threshold)
    if not 0.0 <= threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1 inclusive")

    model = tf.keras.models.load_model(args.model, compile=False)
    probabilities = model.predict(preprocess_image(image_path, image_size), verbose=0)
    probability = float(np.clip(probabilities.reshape(-1)[0], 0.0, 1.0))
    label = int(probability >= threshold)
    confidence = probability if label else 1.0 - probability

    LOGGER.info("Image: %s", image_path.resolve())
    LOGGER.info("Prediction: %s", CLASS_NAMES[label])
    LOGGER.info("Parasitized probability: %.2f%%", probability * 100.0)
    LOGGER.info("Confidence: %.2f%% (threshold %.6f)", confidence * 100.0, threshold)


if __name__ == "__main__":
    main()
