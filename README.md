# Malaria Cell Image Classification

Reproducible TensorFlow/Keras experiments for classifying thin blood-smear cell
images as **Uninfected (0)** or **Parasitized (1)**. This research/education
pipeline is not a clinically validated diagnostic device.

## Dataset and split

By default, images are read from `archive/Uninfected/` and
`archive/Parasitized/`. The pipeline verifies images, performs a 64-bit DCT
perceptual-hash scan for exact and near-duplicates, and reports matches in
`duplicate_report.json`. Duplicate-connected images are held in a single split
to limit leakage. In the absence of matches, splits are stratified 70/15/15
with seed 42. For duplicate-connected data, proportions may vary slightly so
groups remain intact; class distributions are logged.

The test set is withheld from training and threshold selection. It is evaluated
once per run after the model and decision threshold have been chosen using
validation data.

## Experiments

| Experiment | Description |
|---|---|
| `baseline` | CNN with global average pooling and modest L2 regularization |
| `augmented` | Baseline CNN with training-only flips, rotation, zoom, translation, brightness, and contrast |
| `regularized` | From-scratch CNN with BatchNorm, SpatialDropout2D, L2, and global average pooling |
| `transfer` | MobileNetV2 with frozen-backbone training followed by top-layer fine-tuning; BatchNorm remains frozen |
| `ensemble` | Trains baseline, augmented, and regularized candidates, ranks them by validation ROC-AUC, then averages probabilities |

All models use AdamW, label smoothing, early stopping, best-validation-loss
checkpoints, ReduceLROnPlateau, and CSV training logs. Images are resized to
`uint8`, cached, and scaled after the cache. Mixed precision is enabled only
when TensorFlow finds a GPU.

## Installation and usage

Use Python 3.10+ and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train one experiment (defaults: 25 epochs, batch size 32, learning rate 1e-3,
image size 128, seed 42):

```bash
python model.py --experiment baseline
python model.py --experiment regularized --epochs 30 --batch-size 32 --lr 0.001
python model.py --experiment augmented --image-size 128 --seed 42
python model.py --experiment transfer
python model.py --experiment ensemble
```

Provide a different dataset root with `--dataset "path/to/archive"`. The
baseline checkpoint remains `baseline_malaria_cnn.keras`; other checkpoints use
`<experiment>_malaria_cnn.keras`, and the selected model is saved as
`final_malaria_model.keras`.

Optional controls include `--target-recall 0.98`,
`--limit-per-class 32` for a small smoke run, and `--fast` to disable
deterministic TensorFlow ops. `--ensemble-members baseline augmented regularized transfer`
lets the ensemble rank those candidates by validation ROC-AUC and average the best three.
`--reuse-checkpoints` skips fitting when a
candidate checkpoint already exists. By default, each completed experiment
replaces the promoted final model; use `--promote-if-better` to require a strict
validation ROC-AUC improvement instead.

Each run saves `<experiment>_metrics.json`, the final-model metadata
`final_malaria_model_metrics.json`, CSV logs, training curves, a calibration
plot, a confusion matrix, a false-negative-first misclassification gallery,
and Grad-CAM overlays for true positives, false negatives, and false positives
when those examples exist. Outputs are written relative to the current working
directory, with diagnostic figures under `artifacts/<experiment>/`.

## Inference

The predictor reads the validation-selected threshold and input size from
`final_malaria_model_metrics.json`. Transfer-model preprocessing is embedded in
the Keras model, so inference always supplies resized RGB pixels in `[0, 1]`.

```bash
python predict.py "path/to/cell_image.png"
python predict.py "path/to/cell_image.png" --model regularized_malaria_cnn.keras \
  --metadata regularized_metrics.json
```

You can also run `python predict.py` without an image argument; it will prompt
for an image path. Press Enter without entering a path to cancel.

`--threshold` can explicitly override the validation-selected threshold.

## Model comparison

After running any experiments, regenerate the table below from their metrics:

```bash
python compare_results.py
```

<!-- MODEL_COMPARISON_START -->
| Model | Accuracy | Precision | Recall (Parasitized) | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline CNN | — | — | — | — | — | — | — |
| Regularized CNN | — | — | — | — | — | — | — |
| Augmented CNN | — | — | — | — | — | — | — |
| MobileNetV2 transfer | — | — | — | — | — | — | — |
| Ensemble | — | — | — | — | — | — | — |
<!-- MODEL_COMPARISON_END -->

Metrics are provided with bootstrap 95% confidence intervals to communicate
sampling uncertainty. Inspect validation-selected operating thresholds,
confusion matrices, calibration, and error galleries alongside aggregate
scores. Performance targets are goals, not guarantees; no medical claims should
be inferred from these experiments.
