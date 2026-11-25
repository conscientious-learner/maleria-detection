"""Predict whether one blood-cell image is parasitized or uninfected."""
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (128, 128)  # Must match model.py
MODEL_PATH = Path("final_malaria_model.keras")
CLASS_NAMES = ("Uninfected", "Parasitized")


def preprocess_image(image_path: Path) -> np.ndarray:
    """Read RGB, resize, and normalize exactly as the training input pipeline."""
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE, color_mode="rgb")
    array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
    return array[None, ...]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_path", type=Path, help="Path to a cell image")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Saved .keras model path")
    args = parser.parse_args()
    if not args.image_path.is_file():
        parser.error(f"Image file does not exist: {args.image_path}")
    if not args.model.is_file():
        parser.error(f"Model file does not exist: {args.model}. Train model.py first.")

    model = tf.keras.models.load_model(args.model)
    probability_parasitized = float(model.predict(preprocess_image(args.image_path), verbose=0)[0][0])
    label = int(probability_parasitized >= 0.5)
    confidence = probability_parasitized if label else 1.0 - probability_parasitized
    print(f"Prediction: {CLASS_NAMES[label]}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
