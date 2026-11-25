"""End-to-end malaria cell image classification pipeline.

Run a baseline (default): ``python model.py``
Run an experiment: ``python model.py --experiment transfer``
The held-out test split is used only in ``evaluate_test_set`` after training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

# Configuration
DATASET_DIR = Path(r"E:\COLLEGE STUFFS\SEM3\AI for engineers\EST PROJECT\archive\cell_images")
IMAGE_SIZE = (128, 128)  # (height, width); shared with predict.py
SEED = 42
BATCH_SIZE = 32
EPOCHS = 25
CLASS_NAMES = ["Uninfected", "Parasitized"]  # numeric labels 0 and 1
AUTOTUNE = tf.data.AUTOTUNE

tf.keras.utils.set_random_seed(SEED)


def collect_image_paths(dataset_dir: Path = DATASET_DIR) -> Tuple[List[str], np.ndarray]:
    """Return verified image files and labels; unreadable/non-image files are skipped."""
    paths: List[str] = []
    labels: List[int] = []
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    for folder, label in (("Uninfected", 0), ("Parasitized", 1)):
        class_dir = dataset_dir / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            try:
                # verify() checks the file without retaining a decoded image in memory.
                from PIL import Image
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                continue
            paths.append(str(path))
            labels.append(label)
    if not paths:
        raise ValueError(f"No valid images found under {dataset_dir}")
    return paths, np.asarray(labels, dtype=np.int32)


def split_paths(paths: List[str], labels: np.ndarray):
    """Stratified 70/15/15 split, with test references held aside until evaluation."""
    train_paths, remaining_paths, train_y, remaining_y = train_test_split(
        paths, labels, test_size=0.30, random_state=SEED, stratify=labels)
    val_paths, test_paths, val_y, test_y = train_test_split(
        remaining_paths, remaining_y, test_size=0.50, random_state=SEED, stratify=remaining_y)
    print_distribution("Train", train_y)
    print_distribution("Validation", val_y)
    print_distribution("Test", test_y)
    return (train_paths, train_y), (val_paths, val_y), (test_paths, test_y)


def print_distribution(name: str, labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=2)
    print(f"{name:10s}: {len(labels):6d} total | Uninfected={counts[0]} | Parasitized={counts[1]}")


def decode_and_normalize(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.float32)


def make_dataset(paths, labels, training=False, augmentation=None):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), 10000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_normalize, num_parallel_calls=AUTOTUNE)
    if augmentation is not None:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def build_baseline(input_shape=(*IMAGE_SIZE, 3)) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ], name="baseline_malaria_cnn")
    return compile_model(model)


def compile_model(model):
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                           tf.keras.metrics.Recall(name="recall")])
    return model


def build_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
        tf.keras.layers.RandomRotation(0.08, seed=SEED),
        tf.keras.layers.RandomZoom(0.08, seed=SEED),
    ], name="data_augmentation")


def build_regularized(input_shape=(*IMAGE_SIZE, 3)):
    layers = [tf.keras.layers.Input(shape=input_shape)]
    for filters in (32, 64, 128):
        layers += [tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
                   tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu"),
                   tf.keras.layers.MaxPooling2D(), tf.keras.layers.Dropout(0.20)]
    layers += [tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, use_bias=False),
               tf.keras.layers.BatchNormalization(), tf.keras.layers.Activation("relu"),
               tf.keras.layers.Dropout(0.40), tf.keras.layers.Dense(1, activation="sigmoid")]
    return compile_model(tf.keras.Sequential(layers, name="regularized_malaria_cnn"))


def build_transfer(input_shape=(*IMAGE_SIZE, 3)):
    """MobileNetV2 expects RGB values scaled to [-1, 1]."""
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenetv2_preprocessing")(inputs)
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=input_shape)
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_malaria")
    return compile_model(model)


def train_model(model, train_ds, val_ds, checkpoint_path):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True,
                                           save_weights_only=False),
    ]
    history = model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE,
                        epochs=EPOCHS, callbacks=callbacks)
    return history


def plot_history(history, output_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, title in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=title)
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.show()


def evaluate_test_set(model_path, test_ds, test_labels):
    """Load the best checkpoint and evaluate once on the held-out test data."""
    model = tf.keras.models.load_model(model_path)
    probabilities = model.predict(test_ds, verbose=1).ravel()
    predictions = (probabilities >= 0.5).astype(np.int32)
    print("\nTest-set classification report (Parasitized recall is the key safety metric):")
    report = classification_report(test_labels, predictions, target_names=CLASS_NAMES, digits=4)
    print(report)
    print(f"Parasitized recall: {(test_labels[predictions == 1] == 1).sum() / max((test_labels == 1).sum(), 1):.4f}")
    print(f"F1-score: {f1_score(test_labels, predictions):.4f}")
    auc = roc_auc_score(test_labels, probabilities)
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(test_labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    fpr, tpr, _ = roc_curve(test_labels, probabilities)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    return {"accuracy": float(np.mean(predictions == test_labels)),
            "precision": float((predictions[test_labels == 1] == 1).sum() / max((predictions == 1).sum(), 1)),
            "recall": float((predictions[test_labels == 1] == 1).sum() / max((test_labels == 1).sum(), 1)),
            "f1": float(f1_score(test_labels, predictions)), "roc_auc": float(auc)}


def grad_cam(model, image_path, layer_name=None, alpha=0.4):
    """Display Grad-CAM overlay for one image; pass RGB-normalized input through model."""
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = tf.keras.utils.img_to_array(image).astype(np.float32) / 255.0
    batch = tf.convert_to_tensor(array[None, ...])
    # Transfer learning keeps convolutions inside the nested MobileNetV2 model.
    def convolution_layers(container):
        found = []
        for layer in container.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                found.append(layer)
            elif isinstance(layer, tf.keras.Model):
                found.extend(convolution_layers(layer))
        return found

    if layer_name:
        conv_layer = model.get_layer(layer_name)
    else:
        candidates = convolution_layers(model)
        if not candidates:
            raise ValueError("Grad-CAM requires at least one Conv2D layer in the selected model.")
        conv_layer = candidates[-1]
    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, score = grad_model(batch, training=False)
        score = score[:, 0]
    grads = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
    heatmap = tf.image.resize(heatmap[..., None], IMAGE_SIZE).numpy().squeeze()
    import cv2
    color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip(array * 255 * (1 - alpha) + color * alpha, 0, 255))
    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1); plt.imshow(array); plt.axis("off"); plt.title("Input")
    plt.subplot(1, 2, 2); plt.imshow(overlay); plt.axis("off"); plt.title("Grad-CAM")
    plt.tight_layout(); plt.show()
    return heatmap, overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["baseline", "augmented", "regularized", "transfer"], default="baseline")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--gradcam-image", type=Path, help="Optional image to visualize after test evaluation")
    args = parser.parse_args()

    paths, labels = collect_image_paths(args.dataset)
    (train_p, train_y), (val_p, val_y), (test_p, test_y) = split_paths(paths, labels)
    # Test dataset is constructed now but is not iterated or passed to training/selection.
    train_ds = make_dataset(train_p, train_y, training=True)
    val_ds = make_dataset(val_p, val_y)
    test_ds = make_dataset(test_p, test_y)

    augmentation = build_augmentation() if args.experiment == "augmented" else None
    if augmentation is not None:
        train_ds = make_dataset(train_p, train_y, training=True, augmentation=augmentation)
    builders = {"baseline": build_baseline, "augmented": build_baseline,
                "regularized": build_regularized, "transfer": build_transfer}
    model = builders[args.experiment]()
    model.summary()
    checkpoint = "baseline_malaria_cnn.keras" if args.experiment == "baseline" else f"{args.experiment}_malaria_cnn.keras"
    history = train_model(model, train_ds, val_ds, checkpoint)
    plot_history(history, f"{args.experiment}_training_curves.png")
    # Promote this run's best checkpoint as the selected deployable model.
    best_model = tf.keras.models.load_model(checkpoint)
    best_model.save("final_malaria_model.keras")
    evaluate_test_set(checkpoint, test_ds, test_y)
    if args.gradcam_image:
        grad_cam(best_model, args.gradcam_image)


if __name__ == "__main__":
    main()
