from .config import DIMENSIONS, WEIGHTS_PATH, MIXED_PRECISION
from .nn_architecture import build_rgs_unet

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from datetime import timedelta
if MIXED_PRECISION:
    from tensorflow.keras import mixed_precision


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_infer_model(weights_path, warmup=False):
    """Model creation and weight loading"""
    # Enable mixed precision only if a compatible GPU is available
    if MIXED_PRECISION:
        if tf.config.list_physical_devices('GPU'):
            mixed_precision.set_global_policy('mixed_float16')
        else:
            mixed_precision.set_global_policy('float32')

    model = build_rgs_unet(input_shape=DIMENSIONS, num_classes=1)
    model.load_weights(weights_path)
    # Warm-up run to initialize the model before timing actual inference
    if warmup:
        dummy_input = np.zeros((1, *DIMENSIONS), dtype=np.float32)
        _ = model.predict(dummy_input, verbose=0)
    return model


def segment_frame(model, img, filename=None, save_dir=None):
    """Image processing"""
    if filename:
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # Resize mask to match model input dimensions
    resized = cv2.resize(img, (DIMENSIONS[1], DIMENSIONS[0]))
    input_tensor = resized.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Inference
    pred = model.predict(input_tensor, verbose=0)[0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Convert segmentation mask to purple overlay
    overlay = img.copy()
    overlay[:, :, 1] = np.where(mask_resized > 0, 120, overlay[:, :, 1])

    # Save output
    if filename:
        filename_processed = os.path.splitext(filename)[0] + "_processed.jpg"
        save_path = os.path.join(save_dir, filename_processed)
        cv2.imwrite(save_path, overlay)

        print(f"[INFO] Segmented image saved: {save_path}")

    return pred


def segment_video(model, video_path, save_dir=None, record_video=True):
    """Video processing"""
    if record_video:
        if save_dir is None:
            save_dir = os.getcwd()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video source {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if record_video:
        output_path = os.path.join(save_dir, video_name + "_processed.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (DIMENSIONS[1], DIMENSIONS[0]))

    processed_frames = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize mask to match model input dimensions
        resized = cv2.resize(frame, (DIMENSIONS[1], DIMENSIONS[0]))
        input_tensor = resized.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # Inference
        pred = model.predict(input_tensor, verbose=0)[0]
        mask = (pred > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Convert segmentation mask to purple overlay
        overlay = frame.copy()
        overlay[:, :, 1] = np.where(mask_resized > 0, 120, overlay[:, :, 1])

        if record_video:
            out.write(cv2.convertScaleAbs(overlay))

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"[INFO] {processed_frames}/{total_frames} frames processed...")

    cap.release()
    if record_video:
        out.release()

    elapsed = time.time() - start_time
    fps_total = processed_frames / elapsed
    print(f"[INFO] Processing complete: {timedelta(seconds=int(elapsed))}, FPS={fps_total:.2f}")
    if record_video:
        print(f"[INFO] Video saved successfully: {output_path}")


def process_path(model, input_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    ext = os.path.splitext(input_path)[1].lower()

    if ext in IMAGE_EXTENSIONS:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Failed to load image {input_path}")
        filename = os.path.basename(input_path)
        segment_frame(model, img, filename, output_dir)
    elif ext in VIDEO_EXTENSIONS:
        segment_video(model, input_path, output_dir)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def run_segmentation(model, input_path, save_path=os.getcwd()):
    if os.path.isdir(input_path):
        # Batch processing
        for fname in os.listdir(input_path):
            process_path(model, os.path.join(input_path, fname), save_path)
    else:
        # Single file processing
        process_path(model, input_path, save_path)


def main():
    parser = argparse.ArgumentParser(description="RGS-UNet inference CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to image/video or folder")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    model = load_infer_model(WEIGHTS_PATH, warmup=False)
    run_segmentation(model, args.input, args.output)


if __name__ == "__main__":
    main()
