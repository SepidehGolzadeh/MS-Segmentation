# MS-Segmentation
My Bachelor Thesis

# 2D U-Net for Brain Lesion Segmentation
Deep learning pipeline for slice-wise brain lesion segmentation using a 2D U-Net architecture on MRI data.

## Project Overview
This project implements a complete training and evaluation pipeline for brain lesion segmentation using a 2D U-Net model. The workflow includes loading 3D MRI volumes (NIfTI format), slice-wise 2D extraction, intensity normalization and resizing, balanced slice sampling for training stability, BCE + Dice loss optimization, slice-wise evaluation (Dice, FN/FP analysis), automatic probability threshold selection, and HD95 computation using MedPy.
The model is designed for medical image segmentation under severe class imbalance conditions.

## Model Architecture
The segmentation network is a 2D U-Net with an encoder–decoder structure and skip connections. The architecture uses Batch Normalization, ReLU activations, and He initialization. The final layer uses a sigmoid activation for binary segmentation.
Default input size is (224, 224, 1).

## Project Structure
The repository is organized as follows:
* model_architecture.py — U-Net definition
* data_preparation.py — data loading and preprocessing
* training.py — training pipeline
* evaluation.py — evaluation and threshold selection
* models/ — saved trained models
* README.md — project documentation

## Data Processing Pipeline
The preprocessing pipeline converts 3D MRI volumes into 2D slices suitable for CNN training. The steps are:
1. Load 3D MRI volumes (.nii or .nii.gz).
2. Extract axial slices.
3. Resize slices to a fixed spatial resolution (224×224).
4. Apply slice-wise Gaussian normalization using (x − mean) / (std + 1e-6).
5. Clip intensities to the range [-5, 5] and scale to [-1, 1].
6. Add a channel dimension for compatibility with convolutional networks.

## Training Strategy
Due to strong class imbalance (many background-only slices), the training process uses a balanced slice sampling strategy. Slices are divided into positive (contain lesion) and negative (background only). Each batch enforces a fixed positive/negative ratio to stabilize learning.
The validation split is stratified to preserve distribution consistency.

### Loss Function
Training optimizes a combined loss function:
Binary Cross-Entropy + Dice Loss
Binary Cross-Entropy stabilizes pixel-wise classification, while Dice Loss improves region overlap performance and mitigates class imbalance effects.

## Evaluation Protocol
Evaluation is performed slice-wise on the test set.
Metrics include:
* Dice coefficient (computed on positive slices only)
* False Negative (FN) rate
* False Positive (FP) rate
* HD95 (computed only when both prediction and ground truth are non-empty)
To prevent artificially inflated performance due to empty background slices, Dice is calculated only on slices containing lesions in the ground truth.

## Automatic Threshold Selection
The optimal probability threshold is selected by minimizing a weighted combination of false negative and false positive rates:
Score = −(w_fn × FN_rate + w_fp × FP_rate)
The threshold with the highest score is selected, optionally using Dice as a tie-breaker. This allows controlled trade-off between sensitivity and specificity.

## Requirements
* Python 3.9+
* TensorFlow 2.x
* NumPy
* OpenCV
* nibabel
* MedPy
* Matplotlib

## Outputs
The pipeline produces:
* Best trained model (.keras format)
* Training curves (loss and Dice)
* Slice-level evaluation report
* Optimal threshold summary
* Prediction visualizations

## Research Context
This implementation was developed as part of an academic project focused on deep learning–based medical image segmentation and lesion detection.

## Author
Sepideh Golzadeh
