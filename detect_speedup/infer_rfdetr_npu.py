#!/usr/bin/env python
"""RF-DETR inference helpers — GPU / NPU compatible."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# NPU support — optional
try:
    import torch_npu  # noqa: F401
    import torch_npu.contrib.transfer_to_npu  # noqa: F401
except ImportError:
    torch_npu = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def normalize_device(device: str) -> str:
    """Normalize device string. Handles npu / cuda / gpu / cpu."""
    device = device.strip()
    if ":" in device:
        return device
    if device.startswith("npu"):
        return f"npu:{device[3:]}"
    if device.startswith("cuda"):
        return f"cuda:{device[4:]}"
    if device.isdigit():
        # Default to cuda if just a number
        if torch.cuda.is_available():
            return f"cuda:{device}"
        if torch_npu is not None:
            return f"npu:{device}"
    return device


def load_model(
    checkpoint_path: str,
    num_classes: int = 1,
    device: str = "cuda:0",
    resolution: int = 640,
) -> Tuple[Any, List[str]]:
    """Load RF-DETR model from checkpoint and move to device."""
    from rfdetr import RFDETRLarge

    device = normalize_device(device)
    is_npu = device.startswith("npu")

    rfdetr = RFDETRLarge(
        num_classes=num_classes,
        pretrain_weights=checkpoint_path,
        amp=False,
        fused_optimizer=False,
        device="cuda" if is_npu else device,
        resolution=resolution,
    )

    nn_model = rfdetr.model.model
    nn_model = nn_model.to(device)
    nn_model.eval()
    rfdetr.model.device = torch.device(device)

    class_names = (
        rfdetr.class_names
        if hasattr(rfdetr, "class_names")
        else [f"class_{i}" for i in range(num_classes)]
    )
    return rfdetr, class_names


def preprocess_image(
    image: np.ndarray,
    resolution: int = 640,
    input_color: str = "bgr",
    device: str = "cuda:0",
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Preprocess image for RF-DETR inference."""
    from torchvision.transforms import functional as F

    h, w = image.shape[:2]

    if input_color == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    from PIL import Image
    pil_image = Image.fromarray(image)
    tensor = F.to_tensor(pil_image)
    tensor = F.resize(tensor, [resolution, resolution])

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    tensor = tensor.to(device)
    tensor = (tensor - mean) / std

    return tensor.unsqueeze(0), (h, w)


def postprocess_predictions(
    detections: Any,
    orig_size: Tuple[int, int],
    resolution: int = 640,
    conf_threshold: float = 0.3,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Postprocess model predictions to standard dict format."""
    results = []
    h_orig, w_orig = orig_size

    if detections is None or len(detections) == 0:
        return results

    if isinstance(detections, list):
        det = detections[0]
    else:
        det = detections

    boxes = det.get("boxes", None)
    scores = det.get("scores", None)
    labels = det.get("labels", None)

    if boxes is None or scores is None:
        return results

    for i in range(len(boxes)):
        if scores[i] < conf_threshold:
            continue
        box = boxes[i]
        class_id = int(labels[i]) if labels is not None else 0
        class_name = (
            class_names[class_id]
            if class_names and class_id < len(class_names)
            else f"class_{class_id}"
        )
        results.append({
            "bbox_xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "score": float(scores[i]),
            "class_id": class_id,
            "class_name": class_name,
        })
    return results


def nms_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression to detections."""
    if not detections:
        return []

    boxes = np.array([d["bbox_xyxy"] for d in detections])
    scores = np.array([d["score"] for d in detections])

    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return [detections[i] for i in keep]
