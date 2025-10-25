"""
RGS-UNet package initialization.

Contains:
- RGS-UNet model architecture
- Metrics and loss functions
- Data generators
- Inference utilities (image and video processing)
"""

# --- Imports from internal modules ---
from nn_architecture import build_rgs_unet

from .nn_utils import (
    focal_dice_tversky_loss,
    dice_loss,
    focal_loss,
    tversky_loss,
    f1_score_power_line,
    MeanAveragePrecisionIoU
)

from .data_generator import COCOSegmentationGenerator
from .config import (
    DIMENSIONS,
    BATCH_SIZE,
    TOTAL_EPOCHS,
    OPTIMIZER_STEP,
    MIXED_PRECISION
)

from .infer import (
    segment_frame,
    segment_video
)

# --- Define public exports for 'from RGS_UNet import *' ---
__all__ = [
    "build_rgs_unet",
    "focal_dice_tversky_loss",
    "dice_loss",
    "focal_loss",
    "tversky_loss",
    "f1_score_power_line",
    "MeanAveragePrecisionIoU",
    "COCOSegmentationGenerator",
    "DIMENSIONS",
    "BATCH_SIZE",
    "TOTAL_EPOCHS",
    "OPTIMIZER_STEP",
    "MIXED_PRECISION",
    "segment_frame",
    "segment_video"
]
