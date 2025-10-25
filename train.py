from .data_generator import COCOSegmentationGenerator
from .config import DIMENSIONS, OPTIMIZER_STEP, TOTAL_EPOCHS, MIXED_PRECISION, BATCH_SIZE
from .nn_utils import *
from .nn_architecture import build_rgs_unet

import os
import re
import glob
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
if MIXED_PRECISION:
    from tensorflow.keras import mixed_precision


# ---------- Helper to find latest checkpoint ----------
def find_latest_ckpt(path, pattern=r"power_line_seg_epoch_(\d+)\.weights\.h5"):
    tx = []
    for fn in glob.glob(os.path.join(path, "*.weights.h5")):  # Searching for all available model weights
        m = re.search(pattern, os.path.basename(fn))
        if m:
            tx.append((int(m.group(1)), fn))
    if not tx:
        return None, 0
    epoch, fpath = max(tx, key=lambda x: x[0])
    return fpath, epoch


def train_model(dataset_path, save_dir):
    # ---------- mixed precision if GPU is available ----------
    if MIXED_PRECISION:
        if tf.config.list_physical_devices('GPU'):
            mixed_precision.set_global_policy('mixed_float16')
        else:
            mixed_precision.set_global_policy('float32')

    # ---------- Constructing DataGenerator ----------
    train_images = os.path.join(dataset_path, "train")
    train_ann = os.path.join(dataset_path, "train", "_annotations.coco.json")

    valid_images = os.path.join(dataset_path, "valid")
    valid_ann = os.path.join(dataset_path, "valid", "_annotations.coco.json")

    train_gen = COCOSegmentationGenerator(train_images, train_ann, target_class_id=1, batch_size=BATCH_SIZE)
    valid_gen = COCOSegmentationGenerator(valid_images, valid_ann, target_class_id=1, batch_size=BATCH_SIZE,
                                          shuffle=False)

    # ---------- Optimizer with mixed precision support ----------
    optimizer = tf.keras.optimizers.Adam(learning_rate=OPTIMIZER_STEP)
    if MIXED_PRECISION:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    # ---------- Model compilation ----------
    model = build_rgs_unet(input_shape=DIMENSIONS, num_classes=1)
    model.compile(
        optimizer=optimizer,
        loss=focal_dice_tversky_loss,
        metrics=[tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
                 f1_score_power_line,
                 MeanAveragePrecisionIoU()])

    # ---------- Searching latest checkpoint ----------
    latest_path, last_epoch = find_latest_ckpt(save_dir)
    if latest_path and os.path.exists(latest_path):
        print(f"Found latest checkpoint: {os.path.basename(latest_path)} (epoch {last_epoch})")
        model.load_weights(latest_path)
    else:
        print("No checkpoint found; starting from scratch.")
    last_epoch = last_epoch or 0

    # ---------- Add callback to save model weights after each epoch ----------
    checkpoint_cb = ModelCheckpoint(
        filepath=os.path.join(save_dir, "power_line_seg_epoch_{epoch:03d}.weights.h5"),
        save_weights_only=True,
        save_freq='epoch'
    )

    # ---------- Add ReduceLROnPlateau ----------
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',  # metric to monitor
        factor=0.5,  # reduce learning rate by half
        patience=8,  # number of epochs with no improvement before reducing LR
        verbose=1,
        min_lr=1e-7  # lower bound for learning rate
    )

    callbacks = [
        checkpoint_cb,
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        reduce_lr
    ]

    # ---------- Start training ----------
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        initial_epoch=last_epoch,
        epochs=TOTAL_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


def main():
    parser = argparse.ArgumentParser(description="RGS-UNet train CLI")
    parser.add_argument("--dataset", type=str, required=True, help="Path to COCO segmentation format dataset")
    parser.add_argument("--weights", type=str, required=True, help="Directory where the model weights will be saved")
    args = parser.parse_args()

    os.makedirs(args.weights, exist_ok=True)

    history = train_model(args.dataset, args.weights)


if __name__ == "__main__":
    main()
