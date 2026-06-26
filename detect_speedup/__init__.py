# detect_speedup — Unified large-image inference for YOLO, RTMDet, and RF-DETR

from .run_large_image_inference import (
    BACKEND_DEFAULTS,
    COMMON_DEFAULTS,
    RFDETR_DEFAULTS,
    RTMDET_DEFAULTS,
    YOLO_DEFAULTS,
    merge_config,
    run_inference,
    run_rfdetr,
    run_rtmdet,
    run_yolo,
)

# Lazy import — visualization requires PIL, which may not be installed
try:
    from .visualize_predictions import run_visualization
except ImportError:
    run_visualization = None  # type: ignore

__all__ = [
    "run_inference",
    "run_yolo",
    "run_rtmdet",
    "run_rfdetr",
    "run_visualization",
    "merge_config",
    "COMMON_DEFAULTS",
    "YOLO_DEFAULTS",
    "RTMDET_DEFAULTS",
    "RFDETR_DEFAULTS",
    "BACKEND_DEFAULTS",
]
