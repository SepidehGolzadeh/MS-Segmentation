"""
Training pipeline for 2D U-Net lesion segmentation.

This script:
1) Loads slice-wise 2D training data
2) Creates a balanced batch sampler (positive/negative slices)
3) Splits a validation subset from the training set
4) Trains U-Net with BCE+Dice loss and saves the best checkpoint
5) (Optional) tunes probability threshold on validation set
6) Saves training curves (loss and Dice) as PNG files
"""

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Enable GPU memory growth 
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import BinaryIoU
from tensorflow.keras.callbacks import EarlyStopping

from data_preparation import TrainingLoader
from unet_model_recipe import unet_model

# ========================================================
class BalancedSliceSequence(tf.keras.utils.Sequence):
    def __init__(self, pos_slices, neg_slices, batch_size=8, pos_ratio=0.5,
                 steps_per_epoch=200, shuffle=True, seed=42):
        """
        pos_slices: (x_pos, y_pos)
        neg_slices: (x_neg, y_neg)
        pos_ratio: fraction of positives in each batch (e.g., 0.5)
        steps_per_epoch: how many batches per epoch (controls how many neg samples you see)
        """
        self.x_pos, self.y_pos = pos_slices
        self.x_neg, self.y_neg = neg_slices

        assert len(self.x_pos) == len(self.y_pos)
        assert len(self.x_neg) == len(self.y_neg)

        self.batch_size = int(batch_size)
        self.pos_ratio = float(pos_ratio)
        self.steps_per_epoch = int(steps_per_epoch)
        self.shuffle = bool(shuffle)

        self.rng = np.random.default_rng(seed)

        self.pos_n = len(self.x_pos)
        self.neg_n = len(self.x_neg)

        # indices for sampling
        self.pos_idx = np.arange(self.pos_n)
        self.neg_idx = np.arange(self.neg_n)
        self.on_epoch_end()

    def __len__(self):
        return self.steps_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.pos_idx)
            self.rng.shuffle(self.neg_idx)
        self._pos_ptr = 0
        self._neg_ptr = 0

    def _take_indices(self, idx_array, ptr, k):
        """
        Take k indices from idx_array starting at ptr; wrap-around if needed.
        """
        n = len(idx_array)
        if k <= 0:
            return np.empty((0,), dtype=np.int64), ptr

        if ptr + k <= n:
            out = idx_array[ptr:ptr+k]
            ptr = ptr + k
            return out, ptr

        # wrap
        first = idx_array[ptr:]
        remain = k - len(first)

        # reshuffle for new cycle if shuffle enabled
        if self.shuffle:
            self.rng.shuffle(idx_array)

        second = idx_array[:remain]
        ptr = remain
        out = np.concatenate([first, second], axis=0)
        return out, ptr

    def __getitem__(self, step):
        # how many positives/negatives in this batch
        k_pos = int(round(self.batch_size * self.pos_ratio))
        k_pos = max(1, min(k_pos, self.batch_size - 1)) 
        k_neg = self.batch_size - k_pos

        pos_ids, self._pos_ptr = self._take_indices(self.pos_idx, self._pos_ptr, k_pos)
        neg_ids, self._neg_ptr = self._take_indices(self.neg_idx, self._neg_ptr, k_neg)

        x_batch = np.concatenate([self.x_pos[pos_ids], self.x_neg[neg_ids]], axis=0)
        y_batch = np.concatenate([self.y_pos[pos_ids], self.y_neg[neg_ids]], axis=0)

        # shuffle within batch
        if self.shuffle:
            perm = self.rng.permutation(self.batch_size)
            x_batch = x_batch[perm]
            y_batch = y_batch[perm]

        return x_batch, y_batch

# ========================================================
SCRIPT_PATH = os.path.dirname(__file__)
EPOCHS = 30
BATCH_SIZE = 8
" or 16 "

# Preprocess data
training_loader = TrainingLoader()
images_train, masks_train = training_loader.load_training_data(img_size=(224, 224))

# ========================================================
# spliting into validation and training set
# ========================================================
mask_sum = masks_train.reshape(masks_train.shape[0], -1).sum(axis=1)
pos_idx = np.where(mask_sum > 0)[0]
neg_idx = np.where(mask_sum == 0)[0]

rng = np.random.default_rng(42)
val_pos = rng.choice(pos_idx, size=max(1, int(0.1*len(pos_idx))), replace=False)
val_neg = rng.choice(neg_idx, size=max(1, int(0.1*len(neg_idx))), replace=False)

val_idx = np.concatenate([val_pos, val_neg])
train_idx = np.setdiff1d(np.arange(len(images_train)), val_idx)

x_train, y_train = images_train[train_idx], masks_train[train_idx]
x_val,   y_val   = images_train[val_idx], masks_train[val_idx]

mask_sum_tr = y_train.reshape(y_train.shape[0], -1).sum(axis=1)
pos_idx_tr = np.where(mask_sum_tr > 0)[0]
neg_idx_tr = np.where(mask_sum_tr == 0)[0]

pos_slices = (x_train[pos_idx_tr], y_train[pos_idx_tr])
neg_slices = (x_train[neg_idx_tr], y_train[neg_idx_tr])

pos_ratio = 0.5 # the balance ratio
num_pos = len(pos_slices[0])

steps_per_epoch = int(np.ceil(num_pos / max(1, int(BATCH_SIZE * pos_ratio))))

train_seq = BalancedSliceSequence(
    pos_slices=pos_slices,
    neg_slices=neg_slices,
    batch_size=BATCH_SIZE,
    pos_ratio=pos_ratio,
    steps_per_epoch=steps_per_epoch,
    shuffle=True,
    seed=42
)

# ========================================================
# Finding the best Threshold 
# ========================================================
def dice_np(pred_bin, gt_bin, smooth=1e-6):
    inter = np.sum(pred_bin * gt_bin)
    denom = np.sum(pred_bin) + np.sum(gt_bin)
    return (2*inter + smooth) / (denom + smooth)

class ThresholdTuner(tf.keras.callbacks.Callback):
    """
    Callback to select the best probability threshold on validation data.

    It evaluates Dice across a set of thresholds and chooses the threshold
    that maximizes the mean Dice score.

    Notes:
        - By default, evaluation is restricted to positive GT slices to avoid
          inflating Dice due to empty-background slices.
    """
    def __init__(self, x_val, y_val, thresholds=(0.1,0.2,0.3,0.4,0.5),
                 eval_on="positive_slices", batch_size=8):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.thresholds = list(thresholds)
        self.eval_on = eval_on
        self.batch_size = batch_size
        self.best_thr = None
        self.best_score = -1

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        pred = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        gt = (self.y_val > 0).astype(np.uint8)

        # slice masks to choose which slices to score
        gt_sum = gt.reshape(gt.shape[0], -1).sum(axis=1)

        if self.eval_on == "positive_slices":
            use = np.where(gt_sum > 0)[0]
        elif self.eval_on == "both_nonempty": 
            use = None
        else:
            use = np.arange(len(gt))

        best_thr = None
        best_dice = -1

        for thr in self.thresholds:
            pred_bin = (pred >= thr).astype(np.uint8)

            if self.eval_on == "both_nonempty":
                pred_sum = pred_bin.reshape(pred_bin.shape[0], -1).sum(axis=1)
                use_thr = np.where((gt_sum > 0) & (pred_sum > 0))[0]
            else:
                use_thr = use

            if use_thr is None or len(use_thr) == 0:
                mean_d = 0.0
            else:
                dices = []
                for i in use_thr:
                    dices.append(dice_np(pred_bin[i], gt[i]))
                mean_d = float(np.mean(dices))

            if mean_d > best_dice:
                best_dice = mean_d
                best_thr = thr

        self.best_thr = best_thr
        self.best_score = best_dice

        logs["val_best_thr"] = best_thr
        logs["val_best_dice_at_thr"] = best_dice

        print(f"\n[ThresholdTuner] epoch={epoch+1} best_thr={best_thr} val_dice={best_dice:.4f}")

# ========================================================
# Define loss
# ========================================================
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # compute per-image dice (reduce over H,W,C)
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

# ========================================================
# Create and compile model
# ========================================================
model = unet_model(input_size=(224, 224, 1))

model.compile(
    optimizer = Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, BinaryIoU(target_class_ids=[1], threshold=0.5)]
)

# ========================================================
# Save only the best model (minimum validation loss) for reproducibility
# and to prevent overfitting.
model_checkpoint = ModelCheckpoint(
    "/kaggle/working/models/unet_best.keras",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    train_seq,
    validation_data=(x_val, y_val),
    epochs=EPOCHS,
    callbacks=[model_checkpoint, early_stopping],
    verbose=1
)
# ========================================================

tuner = ThresholdTuner(
    x_val, y_val,
    thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5),
    eval_on="positive_slices",
    batch_size=BATCH_SIZE
)
tuner.set_model(model)        
tuner.on_epoch_end(epoch=0)        

print("Best threshold on VAL:", tuner.best_thr, "VAL dice:", tuner.best_score)

# ========================================================
# Plot
# ========================================================

plot_dir = "/kaggle/working/plots"
os.makedirs(plot_dir, exist_ok=True)

# Loss Curve
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Val Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()


# Dice Curve
plt.figure(figsize=(6,4))

# train dice
if 'dice_coef' in history.history:
    plt.plot(history.history['dice_coef'], label='Train Dice')

# validation dice
if 'val_dice_coef' in history.history:
    plt.plot(history.history['val_dice_coef'], label='Val Dice')

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Training & Validation Dice")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "dice_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

print("Plots saved in:", plot_dir)