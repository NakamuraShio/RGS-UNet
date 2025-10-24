"""
RGS-UNet package initialization.

Содержит:
- Архитектуру модели RGS-UNet
- Метрики и функции потерь
- Генераторы данных
- Утилиты для инференса (обработка изображений и видео)
"""

# --- Импорты из внутренних модулей ---
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

# --- Определяем, что экспортировать при from RGS_UNet import * ---
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
