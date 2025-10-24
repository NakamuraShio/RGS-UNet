from .config import MIXED_PRECISION

import numpy as np
import tensorflow as tf


# ------------ Focal Loss ------------
def focal_loss(y_true, y_pred, alpha=1.0, gamma=2.0):
    if MIXED_PRECISION:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1. - 1e-7)
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.pow(1 - y_pred, gamma)
    return tf.reduce_mean(weight * cross_entropy)


# ------------ Dice Loss ------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    if MIXED_PRECISION:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice


# ------------ Tversky Loss ------------
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss for binary segmentation.
    y_true, y_pred: same shape, last dim = channels
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # If logits -> apply sigmoid
    y_pred = tf.nn.sigmoid(y_pred)

    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1.0 - tversky_index


# ------------ Combined Focal + Dice + Tversky ------------
def focal_dice_tversky_loss(y_true, y_pred,
                            focal_weight=0.6,
                            dice_weight=0.2,
                            tversky_weight=0.2,
                            alpha=0.8,
                            beta=0.2):
    focal = focal_loss(y_true, y_pred)
    dice  = dice_loss(y_true, y_pred)
    tversky = tversky_loss(y_true, y_pred, alpha=alpha, beta=beta)
    return (focal_weight * focal +
            dice_weight * dice +
            tversky_weight * tversky)


# ---------- F1-score (power_line class) ----------
def f1_score_power_line(y_true, y_pred, threshold=0.5):
    if MIXED_PRECISION:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
    y_pred_bin = tf.cast(y_pred > threshold, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred_bin)
    fp = tf.reduce_sum((1 - y_true) * y_pred_bin)
    fn = tf.reduce_sum(y_true * (1 - y_pred_bin))

    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())


# ---------- mAP 0.5:0.95 metric ----------
class MeanAveragePrecisionIoU(tf.keras.metrics.Metric):
    def __init__(self, thresholds=None, name="mean_ap_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        if thresholds is None:
            thresholds = np.arange(0.5, 1.0, 0.05)
        self.thresholds = list(thresholds)
        self.total_ap = self.add_weight(name="total_ap", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        if MIXED_PRECISION:
            y_pred = tf.cast(y_pred, tf.float32)

        aps = []
        for thresh in self.thresholds:
            y_pred_thresh = tf.cast(y_pred > thresh, tf.float32)

            intersection = tf.reduce_sum(y_true * y_pred_thresh, axis=[1, 2, 3])
            union = tf.reduce_sum(tf.cast((y_true + y_pred_thresh) >= 1.0, tf.float32), axis=[1, 2, 3])
            iou = intersection / (union + 1e-7)
            hits = tf.cast(iou >= thresh, tf.float32)

            aps.append(hits)

        mean_ap = tf.reduce_mean(tf.stack(aps, axis=0))  # (num_thresholds, batch)
        self.total_ap.assign_add(tf.reduce_mean(mean_ap))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_ap / (self.count + 1e-7)

    def reset_states(self):
        self.total_ap.assign(0.0)
        self.count.assign(0.0)
