# Malaria Cell Image Classification

## Project Overview

An experimental TensorFlow/Keras CNN pipeline that classifies red blood cell images as **Uninfected (0)** or **Parasitized (1)**. It includes reproducible data splits, a baseline model, optional improvement experiments, held-out evaluation, Grad-CAM visualization, and single-image inference. This is an educational prototype and is not a clinically validated diagnostic device.

## Dataset Split

Source directory: `archive/cell_images/`, with `Uninfected/` and `Parasitized/` subfolders. The loader verifies supported image files, converts them to RGB at 128 × 128, and scales pixels to `[0, 1]`.

| Split | Fraction | Use |
|---|---:|---|
| Train | 70% | Fit model parameters |
| Validation | 15% | Monitor training and select the best checkpoint |
| Test | 15% | Final evaluation after training |

Splitting is stratified and uses seed 42. Exact per-class counts are printed at startup.

## Model Comparison

Fill this table with metrics printed by each run. Compare experiments on the same held-out split; avoid selecting a model based on repeated test-set tuning.

| Model | Accuracy | Precision | Recall (Parasitized) | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | — | — | — | — | — |
| Augmented CNN | — | — | — | — | — |
| Regularized CNN | — | — | — | — | — |
| MobileNetV2 | — | — | — | — | — |

Recall for the Parasitized class is especially important because a false negative means an infected cell was missed. Interpret metrics alongside the confusion matrix and clinical context.

## Grad-CAM Findings

Use `--gradcam-image` to display a heatmap overlay for an image after model evaluation. Record whether activation aligns with parasite regions and inspect multiple examples, including errors. Grad-CAM is a qualitative diagnostic aid and does not establish clinical validity.

## Usage Instructions

Install the required packages in a Python environment compatible with TensorFlow:

```bash
pip install tensorflow numpy pillow matplotlib seaborn scikit-learn opencv-python
```

Run the baseline (dataset path defaults to the configured local path in `model.py`):

```bash
python model.py
```

Run one modular experiment at a time:

```bash
python model.py --experiment augmented
python model.py --experiment regularized
python model.py --experiment transfer
```

Override the dataset root if needed:

```bash
python model.py --dataset "path/to/cell_images"
```

Each run saves its best validation-loss checkpoint and promotes that checkpoint to `final_malaria_model.keras`. The baseline checkpoint is `baseline_malaria_cnn.keras`; experiment checkpoints are named by experiment. Evaluation plots are saved in the working directory. To predict one image:

```bash
python predict.py "path/to/cell_image.png"
```

For Grad-CAM:

```bash
python model.py --gradcam-image "path/to/cell_image.png"
```

The default prediction threshold is 0.5. Any deployment threshold should be selected using validation data and domain expertise.
