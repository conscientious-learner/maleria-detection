"""Grad-CAM graph construction, heatmap computation, and visualization."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import GRADCAM_SAMPLES_PER_CATEGORY


LOGGER = logging.getLogger("malaria.gradcam")


def find_last_conv_layer(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model | None, tf.keras.layers.Conv2D]:
    """Find the final convolution, including inside the named transfer backbone."""
    try:
        backbone = model.get_layer("backbone")
    except ValueError:
        backbone = None
    container = backbone if backbone is not None else model
    convolutions = [
        layer for layer in container.layers if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not convolutions:
        raise ValueError("Grad-CAM requires a Conv2D layer in the selected model.")
    return backbone, convolutions[-1]


def build_gradcam_model(model: tf.keras.Model) -> tf.keras.Model:
    """Build a differentiable model for plain CNN or nested MobileNetV2 outputs."""
    backbone, convolution = find_last_conv_layer(model)
    if backbone is None:
        return tf.keras.Model(model.inputs, [convolution.output, model.output])
    return _build_backbone_gradcam_model(model, backbone, convolution)


def _build_backbone_gradcam_model(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
    convolution: tf.keras.layers.Conv2D,
) -> tf.keras.Model:
    """Rebuild the transfer-model tail so predictions depend on captured features."""
    features = tf.keras.Model(backbone.input, convolution.output)
    tail = tf.keras.Model(convolution.output, backbone.output)
    prepared_input = model.get_layer("backbone_preprocessing")(model.inputs[0])
    feature_maps = features(prepared_input, training=False)
    scores = tail(feature_maps, training=False)
    backbone_index = model.layers.index(backbone)
    for layer in model.layers[backbone_index + 1:]:
        scores = layer(scores, training=False)
    return tf.keras.Model(model.inputs, [feature_maps, scores])


def compute_heatmap(gradcam_model: tf.keras.Model, batch: tf.Tensor) -> np.ndarray:
    """Calculate a normalized positive-class Grad-CAM map."""
    with tf.GradientTape() as tape:
        feature_maps, score = gradcam_model(batch, training=False)
        positive_score = score[:, 0]
    gradients = tape.gradient(positive_score, feature_maps)
    if gradients is None:
        raise ValueError("Grad-CAM gradients are disconnected from the classifier output.")
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * feature_maps, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4
) -> np.ndarray:
    """Blend a normalized heatmap with a float RGB image."""
    color_map = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip(image * (1.0 - alpha) + color_map * alpha, 0.0, 1.0)


def save_overlay(image: np.ndarray, overlay: np.ndarray, output_path: Path) -> None:
    """Save paired input and activation views."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def grad_cam(
    model: tf.keras.Model,
    image_path: str | Path,
    image_size: int,
    output_path: Path,
    alpha: float = 0.4,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap and save its overlay."""
    image = tf.keras.utils.load_img(
        image_path, target_size=(image_size, image_size), color_mode="rgb"
    )
    array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
    batch = tf.convert_to_tensor(array[None, ...])
    gradient_model = build_gradcam_model(model)
    heatmap = compute_heatmap(gradient_model, batch)
    heatmap = tf.image.resize(heatmap[..., None], (image_size, image_size)).numpy().squeeze()
    save_overlay(array, overlay_heatmap(array, heatmap, alpha), output_path)
    return heatmap


def _example_indices(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """Group test positions into true-positive, false-negative, and false-positive sets."""
    predictions = (probabilities >= threshold).astype(np.int32)
    return {
        "true_positive": np.flatnonzero((labels == 1) & (predictions == 1)),
        "false_negative": np.flatnonzero((labels == 1) & (predictions == 0)),
        "false_positive": np.flatnonzero((labels == 0) & (predictions == 1)),
    }


def save_gradcam_examples(
    model: tf.keras.Model,
    paths: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    image_size: int,
    output_dir: Path,
) -> None:
    """Save a small Grad-CAM sample for each available test prediction group."""
    for category, indices in _example_indices(labels, probabilities, threshold).items():
        _save_category_examples(model, category, indices, paths, image_size, output_dir)


def _save_category_examples(
    model: tf.keras.Model,
    category: str,
    indices: np.ndarray,
    paths: Sequence[str],
    image_size: int,
    output_dir: Path,
) -> None:
    """Save configured examples, logging isolated Grad-CAM failures."""
    for number, index in enumerate(indices[:GRADCAM_SAMPLES_PER_CATEGORY], start=1):
        destination = output_dir / f"{category}_{number}.png"
        try:
            grad_cam(model, paths[index], image_size, destination)
        except (ValueError, tf.errors.InvalidArgumentError) as error:
            LOGGER.warning("Grad-CAM failed for %s: %s", paths[index], error)
