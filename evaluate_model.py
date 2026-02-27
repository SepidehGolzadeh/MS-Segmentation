"""
Evaluation pipeline for 2D U-Net brain lesion segmentation.

This script:
1. Loads trained model
2. Runs inference on test set
3. Performs slice-wise evaluation
4. Selects best threshold based on FN/FP trade-off
5. Reports Dice and HD95 (MedPy)
"""

import data_preparation
print(data_preparation.__file__)

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.models import load_model
from data_preparation import TrainingLoader, TestingLoader
import numpy as np
from medpy import metric


BATCH_SIZE = 8

# ======================================
# Load training and test datasets
# ======================================
training_loader = TrainingLoader()
testing_loader = TestingLoader()

images_train, masks_train = training_loader.load_training_data(img_size=(224, 224))
images_test, masks_test = testing_loader.load_testing_data(img_size=(224, 224))
import numpy as np
import matplotlib.pyplot as plt

def report_slice_distribution(images, masks, set_name="Dataset"):
    """
    images: numpy array (N, H, W, C)
    masks:  numpy array (N, H, W, C)
    """

    N = masks.shape[0]

    mask_sum = masks.reshape(N, -1).sum(axis=1)

    zero_slices = np.sum(mask_sum == 0)
    lesion_slices = np.sum(mask_sum > 0)

    print("======================================")
    print(f"{set_name} Slice Distribution")
    print(f"Total slices     : {N}")
    print(f"Zero slices      : {zero_slices}")
    print(f"Lesion slices    : {lesion_slices}")
    print(f"Lesion ratio (%) : {100 * lesion_slices / N:.2f}%")
    print("======================================")

    return {
        "total": int(N),
        "zero": int(zero_slices),
        "lesion": int(lesion_slices),
        "lesion_ratio": float(lesion_slices / N)
    }

report_slice_distribution(images_train, masks_train, set_name="Train")
report_slice_distribution(images_test, masks_test, set_name="Test")


# ======================================
def evaluate_slicewise(pred_prob, gt_mask, thr, eps=1e-6):
    """
    pred_prob: numpy array, shape (N,H,W,1) or (N,H,W) - probabilities in [0,1]
    gt_mask:  numpy array, shape (N,H,W,1) or (N,H,W) - ground truth mask (0/1 or 0/255)
    thr: threshold to binarize predictions

    Returns a dict of useful slice-level metrics.
    """

    if pred_prob.ndim == 4 and pred_prob.shape[-1] == 1:
        pred_prob_ = pred_prob[..., 0]
    else:
        pred_prob_ = pred_prob

    if gt_mask.ndim == 4 and gt_mask.shape[-1] == 1:
        gt_ = gt_mask[..., 0]
    else:
        gt_ = gt_mask

    # binarize
    pred_bin = (pred_prob_ >= thr).astype(np.uint8)
    gt_bin   = (gt_ > 0).astype(np.uint8)

    N = gt_bin.shape[0]
    gt_sum = gt_bin.reshape(N, -1).sum(axis=1)
    pr_sum = pred_bin.reshape(N, -1).sum(axis=1)

    pos_idx = np.where(gt_sum > 0)[0]
    neg_idx = np.where(gt_sum == 0)[0]

    # --- Dice on positive slices only ---
    dice_pos = []
    fn_pos = 0 

    for i in pos_idx:
        if pr_sum[i] == 0:
            fn_pos += 1
            dice_pos.append(0.0)
        else:
            d = metric.binary.dc(pred_bin[i], gt_bin[i])
            dice_pos.append(float(d))

    mean_dice_pos = float(np.mean(dice_pos)) if len(dice_pos) else 0.0
    median_dice_pos = float(np.median(dice_pos)) if len(dice_pos) else 0.0

    # --- False positives on negative slices ---
    fp_neg = int(np.sum(pr_sum[neg_idx] > 0)) if len(neg_idx) else 0
    fp_rate_neg = (fp_neg / max(1, len(neg_idx)))

    # --- HD95 only when both are non-empty ---
    hd95_list = []
    both_nonempty_idx = np.where((gt_sum > 0) & (pr_sum > 0))[0]
    for i in both_nonempty_idx:
        try:
            hd = metric.binary.hd95(pred_bin[i], gt_bin[i])
            hd95_list.append(float(hd))
        except Exception:
            pass

    mean_hd95 = float(np.mean(hd95_list)) if len(hd95_list) else 0.0
    median_hd95 = float(np.median(hd95_list)) if len(hd95_list) else 0.0

    # --- counts ---
    out = {
        "thr": thr,
        "N_total_slices": int(N),
        "N_pos_slices": int(len(pos_idx)),
        "N_neg_slices": int(len(neg_idx)),

        "dice_pos_mean": mean_dice_pos,
        "dice_pos_median": median_dice_pos,

        "FN_pos_slices": int(fn_pos),
        "FN_pos_rate": float(fn_pos / max(1, len(pos_idx))),

        "FP_neg_slices": int(fp_neg),
        "FP_neg_rate": float(fp_rate_neg),

        "N_both_nonempty": int(len(both_nonempty_idx)),
        "hd95_mean": mean_hd95,
        "hd95_median": median_hd95,
    }

    # print a summary
    print("======================================")
    print(f"Slice-level evaluation @ thr={thr}")
    print(f"Total slices: {out['N_total_slices']} | Pos: {out['N_pos_slices']} | Neg: {out['N_neg_slices']}")
    print(f"Dice (pos slices only): mean={out['dice_pos_mean']:.4f} | median={out['dice_pos_median']:.4f}")
    print(f"FN on pos slices: {out['FN_pos_slices']} ({out['FN_pos_rate']*100:.2f}%)")
    print(f"FP on neg slices: {out['FP_neg_slices']} ({out['FP_neg_rate']*100:.2f}%)")
    print(f"HD95 (both non-empty): mean={out['hd95_mean']:.4f} | median={out['hd95_median']:.4f} | count={out['N_both_nonempty']}")
    print("======================================")

    return out


def select_best_threshold(pred_prob, gt_mask, thresholds, w_fn=2.0, w_fp=1.0, verbose=True):
    """
    Select the optimal probability threshold by minimizing
    a weighted combination of:

        Score = -(w_fn * FN_rate + w_fp * FP_rate)

    Higher score is better.
    Tie-breaker: higher Dice on positive slices.
    """
    results = []

    best = None
    best_score = -1e18

    for thr in thresholds:
        out = evaluate_slicewise(pred_prob, gt_mask, thr=thr)

        fnr = out["FN_pos_rate"]
        fpr = out["FP_neg_rate"]
        dice = out["dice_pos_mean"]

        score = -(w_fn * fnr + w_fp * fpr) 

        results.append({**out, "score": score})

        if (score > best_score) or (np.isclose(score, best_score) and dice > (best["dice_pos_mean"] if best else -1)):
            best_score = score
            best = results[-1]

    if verbose:
        print("\n================ BEST THRESHOLD ================")
        print(f"Best thr: {best['thr']}")
        print(f"Score   : {best['score']:.6f}   (higher is better)")
        print(f"Dice(pos): {best['dice_pos_mean']:.4f}")
        print(f"FN rate : {best['FN_pos_rate']*100:.2f}%")
        print(f"FP rate : {best['FP_neg_rate']*100:.2f}%")
        print(f"HD95 mean (both): {best['hd95_mean']:.4f}  count={best['N_both_nonempty']}")
        print("================================================\n")

    return best, results

# ======================================
# Metric
# ======================================
def calculate_metric_percase(pred, gt, thr):
    pred = (pred >= thr).astype(np.uint8)
    gt   = (gt > 0).astype(np.uint8)

    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 0.0, 0.0
    elif pred.sum() == 0 and gt.sum() > 0:
        return 0.0, 0.0

# ======================================
# Define loss
# ======================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = (1, 2, 3)
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denom = tf.reduce_sum(y_true + y_pred, axis=axes)
    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# ======================================
# Model
# ======================================
SCRIPT_PATH = os.path.dirname(__file__)
model_path = os.path.join(SCRIPT_PATH, "models", "unet_best.keras")
#model_path = "/kaggle/input/datasets/sepidehgolzadeh/the-balanced-model/unet_best.keras"

model_best = tf.keras.models.load_model(
    model_path,
    compile=False
)

model_best.compile(
    optimizer=Adam(1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, BinaryIoU(target_class_ids=[1], threshold=0.5)]
)

model_best.summary()
# ======================================
# Evaluation
# ======================================
def run_predict(model, x, batch_size=8, thr=0.1):
    pred_prob = model.predict(x, batch_size=batch_size, verbose=1).astype(np.float32)
    pred_bin  = (pred_prob >= thr).astype(np.uint8)

    print("pred_prob stats:", pred_prob.min(), pred_prob.max(), pred_prob.mean(), "sum:", float(pred_prob.sum()))
    print("pred_bin  stats:", pred_bin.min(), pred_bin.max(), pred_bin.mean(), "sum:", int(pred_bin.sum()))
    return pred_prob, pred_bin

pred_prob_full, pred_bin_full = run_predict(model_best, images_test, batch_size=8, thr=0.1)


def plot_prediction(images, masks, pred_prob, idx, thr=0.1, alpha=0.35, title=""):
    """
    images:    (N,H,W,1)
    masks:     (N,H,W,1)  ground truth
    pred_prob: (N,H,W,1)  probabilities
    """
    img = images[idx, ..., 0]
    gt  = (masks[idx, ..., 0] > 0).astype(np.uint8)
    prp = pred_prob[idx, ..., 0]
    prb = (prp >= thr).astype(np.uint8)

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))

    ax[0].imshow(img, cmap="gray")
    ax[0].set_title(f"{title} Image (idx={idx})")
    ax[0].axis("off")

    ax[1].imshow(gt, cmap="gray")
    ax[1].set_title("GT mask")
    ax[1].axis("off")

    ax[2].imshow(prp, cmap="gray")
    ax[2].set_title("Pred prob")
    ax[2].axis("off")

    ax[3].imshow(prb, cmap="gray")
    ax[3].set_title(f"Pred bin (thr={thr})")
    ax[3].axis("off")

    # Overlay
    ax[4].imshow(img, cmap="gray")
    ax[4].imshow(gt, alpha=alpha)          # GT overlay
    ax[4].imshow(prb, alpha=alpha)         # Pred overlay
    ax[4].set_title("Overlay: GT + Pred")
    ax[4].axis("off")

    plt.tight_layout()
    plt.show()

def save_prediction_png(images, masks, pred_prob, idx, thr=0.1, alpha=0.35, out_dir="/kaggle/working/preds"):
    os.makedirs(out_dir, exist_ok=True)
    img = images[idx, ..., 0]
    gt  = (masks[idx, ..., 0] > 0).astype(np.uint8)
    prp = pred_prob[idx, ..., 0]
    prb = (prp >= thr).astype(np.uint8)

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title(f"Image idx={idx}"); ax[0].axis("off")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT"); ax[1].axis("off")
    ax[2].imshow(prp, cmap="gray"); ax[2].set_title("Pred prob"); ax[2].axis("off")
    ax[3].imshow(prb, cmap="gray"); ax[3].set_title(f"Pred bin thr={thr}"); ax[3].axis("off")
    ax[4].imshow(img, cmap="gray"); ax[4].imshow(gt, alpha=alpha); ax[4].imshow(prb, alpha=alpha)
    ax[4].set_title("Overlay"); ax[4].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, f"pred_{idx}_thr{thr}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


THR = [0.5, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]

best, all_results = select_best_threshold(
    pred_prob_full, masks_test,
    thresholds=THR,
    w_fn=2.0,   
    w_fp=1.0,
    verbose=True
)

BEST_THRESHOLD = best["thr"]

mask_sum = masks_test.reshape(masks_test.shape[0], -1).sum(axis=1)
pos_ids = np.where(mask_sum > 0)[0]
print("num positive slices:", len(pos_ids), "example:", pos_ids[:10])

plot_prediction(images_test, masks_test, pred_prob_full, idx=int(pos_ids[25]), thr=BEST_THRESHOLD, title="TEST")
save_prediction_png(images_test, masks_test, pred_prob_full, idx=int(pos_ids[25]), thr=BEST_THRESHOLD)

gt_sum = (masks_test > 0).reshape(masks_test.shape[0], -1).sum(axis=1)
pos_ids = np.where(gt_sum > 0)[0]

dice_list = []
hd95_list = []

pred_bin = (pred_prob_full >= BEST_THRESHOLD).astype(np.uint8)
gt_bin   = (masks_test > 0).astype(np.uint8)

for i in pos_ids:
    p = pred_bin[i, ..., 0]
    g = gt_bin[i, ..., 0]

    if p.sum() > 0 and g.sum() > 0:
        dice_list.append(metric.binary.dc(p, g))
        hd95_list.append(metric.binary.hd95(p, g))
    elif p.sum() == 0 and g.sum() > 0:
        dice_list.append(0.0)  # FN
    else:
        dice_list.append(0.0)

dice_list = np.array(dice_list, dtype=np.float32)
hd95_list = np.array(hd95_list, dtype=np.float32) if len(hd95_list) else np.array([])

print("======================================")
print(f"MedPy (POS slices only) @ thr={BEST_THRESHOLD}")
print("Pos slices:", len(pos_ids))
print("Mean Dice:", float(dice_list.mean()) if len(dice_list) else 0.0)
print("Median Dice:", float(np.median(dice_list)) if len(dice_list) else 0.0)
print("Mean HD95 (both non-empty only):", float(hd95_list.mean()) if len(hd95_list) else 0.0)
print("======================================")