from . import runtime_warning_filters  # noqa: F401
import argparse
import contextlib
import datetime as dt
import gc
import json
import math
import multiprocessing as mp
import os
import queue
import struct
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

from .process_title_utils import infer_title, read_title, set_process_title

os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")
warnings.filterwarnings(
    "ignore",
    message=r"Warning: The .* owner does not match the current owner\.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*owner does not match the current owner.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`TorchScript` support for functional optimizers is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*TorchScript.*functional optimizers is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"mmengine\.optim\.optimizer\.zero_optimizer",
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"torch\.distributed\.optim",
)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast\(args\.\.\.\).*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release, it will be required to pass the indexing argument\..*",
    category=UserWarning,
)

import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.transforms import Compose
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import get_test_pipeline_cfg
try:
    from . import rtmdet_custom_datasets  # noqa: F401
except ImportError:
    pass

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]
SUPPORTED_INPUT_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class MultiprocessMMDetInferConfig:
    source: str
    config_file: str
    checkpoint_file: str
    output: Optional[str] = "./outputs/large_image_predictions.json"
    output_shp: Optional[str] = None
    output_bbox_shp: Optional[str] = None
    output_mask_shp: Optional[str] = None
    save_json: bool = True
    device: str = "cpu"
    devices: Optional[List[str]] = None
    worker_count: Optional[int] = None
    reader_workers: int = 1
    queue_size: int = 8
    mp_start: str = "spawn"
    score_thr: float = 0.3
    tile_size: int = 1280
    stride: Optional[int] = None
    batch_size: int = 4
    global_nms: bool = False
    global_iou: float = 0.45
    classes: Optional[List[int]] = None
    fp16: bool = False
    small_image_threshold: int = 12000
    prefer_single_process_for_small: bool = True
    save_mask_overlay: bool = False
    # 建筑物轮廓正则化配置 (dict)，None = 不启用
    regularize: Optional[dict] = None


@dataclass
class ReusableMMDetModel:
    model: Any
    class_names: List[str]
    device: str
    test_pipeline: Optional[Compose] = None


class ProgressReporter:
    def __init__(self, total_tiles, desc_prefix="Tiles"):
        self.total_tiles = total_tiles
        self.start_time = time.monotonic()
        self.last_status_time = 0.0
        self.status_interval = 5.0
        self.read_tiles = 0
        self.infer_tiles = 0
        self.detections = 0
        self.read_bar = None
        self.infer_bar = None
        if tqdm is not None:
            self.read_bar = tqdm(
                total=total_tiles,
                desc=f"{desc_prefix} read ",
                unit="tile",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            self.infer_bar = tqdm(
                total=total_tiles,
                desc=f"{desc_prefix} infer",
                unit="tile",
                position=1,
                leave=True,
                dynamic_ncols=True,
            )

    def write(self, message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message, flush=True)

    def update_read(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.read_tiles = self._update_bar(self.read_bar, self.read_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.read_bar, queue_size, ready_workers, total_workers, self.detections)
        self.maybe_log_status(queue_size, ready_workers, total_workers, self.detections)

    def update_infer(self, tiles, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        self.infer_tiles = self._update_bar(self.infer_bar, self.infer_tiles, tiles)
        if detections is not None:
            self.detections = max(int(detections), 0)
        self._set_postfix(self.infer_bar, queue_size, ready_workers, total_workers, self.detections)
        self.maybe_log_status(queue_size, ready_workers, total_workers, self.detections)

    def _update_bar(self, bar, current_tiles, new_tiles):
        new_tiles = min(max(int(new_tiles), 0), self.total_tiles)
        delta = new_tiles - current_tiles
        if bar is not None and delta > 0:
            bar.update(delta)
        return max(current_tiles, new_tiles)

    def _set_postfix(self, bar, queue_size=None, ready_workers=None, total_workers=None, detections=None):
        if bar is None:
            return
        postfix = {}
        if queue_size is not None:
            postfix["queue"] = queue_size
        if ready_workers is not None and total_workers is not None:
            postfix["workers"] = f"{ready_workers}/{total_workers}"
        if detections is not None:
            postfix["detections"] = int(detections)
        if postfix:
            bar.set_postfix(postfix, refresh=False)

    def maybe_log_status(self, queue_size=None, ready_workers=None, total_workers=None, detections=None, force=False):
        if tqdm is not None:
            return
        now = time.monotonic()
        if not force and (now - self.last_status_time) < self.status_interval:
            return

        elapsed = max(now - self.start_time, 1e-6)
        read_speed = self.read_tiles / elapsed
        infer_speed = self.infer_tiles / elapsed
        read_pct = (self.read_tiles / self.total_tiles * 100.0) if self.total_tiles > 0 else 0.0
        infer_pct = (self.infer_tiles / self.total_tiles * 100.0) if self.total_tiles > 0 else 0.0
        eta_text = self.format_eta(infer_speed)

        parts = [
            f"[progress] read {self.read_tiles}/{self.total_tiles} ({read_pct:.1f}%)",
            f"infer {self.infer_tiles}/{self.total_tiles} ({infer_pct:.1f}%)",
            f"read_speed {read_speed:.1f} tile/s",
            f"infer_speed {infer_speed:.1f} tile/s",
            f"eta {eta_text}",
            f"detections {int(self.detections if detections is None else detections)}",
        ]
        if queue_size is not None:
            parts.append(f"queue {queue_size}")
        if ready_workers is not None and total_workers is not None:
            parts.append(f"workers {ready_workers}/{total_workers}")

        self.write(", ".join(parts))
        self.last_status_time = now

    def format_eta(self, infer_speed):
        if self.total_tiles <= 0 or infer_speed <= 1e-6:
            return "--"
        remaining_tiles = max(self.total_tiles - self.infer_tiles, 0)
        remaining_seconds = int(round(remaining_tiles / infer_speed))
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def close(self):
        if self.read_bar is not None:
            self.read_bar.n = min(self.read_tiles, self.total_tiles)
            self.read_bar.refresh()
            self.read_bar.close()
        if self.infer_bar is not None:
            self.infer_bar.n = min(self.infer_tiles, self.total_tiles)
            self.infer_bar.refresh()
            self.infer_bar.close()
        self.maybe_log_status(detections=self.detections, force=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-image MMDetection inference with reader and predictor workers."
    )
    parser.add_argument("source", type=str, help="Path to a large image file or a directory.")
    parser.add_argument("--config-file", type=str, required=True, help="MMDetection config path.")
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=True,
        help="MMDetection checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/large_image_predictions.json",
        help="Path to save merged predictions. JSON by default.",
    )
    parser.add_argument(
        "--output-shp",
        type=str,
        default=None,
        help="Optional path to save bbox predictions as a shapefile. Kept for compatibility.",
    )
    parser.add_argument(
        "--output-bbox-shp",
        type=str,
        default=None,
        help="Optional path to save bbox polygons as a shapefile.",
    )
    parser.add_argument(
        "--output-mask-shp",
        type=str,
        default=None,
        help="Optional path to save instance mask polygons as a shapefile.",
    )
    parser.add_argument(
        "--no-save-json",
        action="store_true",
        help="Do not write the merged prediction JSON file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="npu:0" if hasattr(torch, "npu") else "cpu",
        help="Fallback device used when --devices is omitted.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=None,
        help="Device list for predictor workers, for example cpu or npu:0 npu:1.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Number of predictor workers. Defaults to len(--devices) or 2.",
    )
    parser.add_argument(
        "--reader-workers",
        type=int,
        default=1,
        help="Number of reader processes used for tile extraction. This is CPU I/O work.",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=8,
        help="Max number of tile batches buffered between reader and predictors.",
    )
    parser.add_argument(
        "--mp-start",
        type=str,
        default="spawn",
        choices=("spawn", "fork", "forkserver"),
        help="Multiprocessing start method.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="Score threshold applied after model inference.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1280,
        help="Tile size for large-image window reading.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Tile stride. Defaults to tile size when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of tiles per MMDetection batch.",
    )
    parser.add_argument(
        "--global-nms",
        action="store_true",
        help="Apply an extra cross-tile global NMS after tile inference.",
    )
    parser.add_argument(
        "--global-iou",
        type=float,
        default=0.45,
        help="IoU threshold for optional cross-tile global NMS.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class IDs to keep.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable autocast during inference when the backend supports it.",
    )
    parser.add_argument(
        "--small-image-threshold",
        type=int,
        default=12000,
        help="Prefer a lighter single-process path when both image sides are below this threshold.",
    )
    parser.add_argument(
        "--disable-small-image-fastpath",
        action="store_true",
        help="Always use the multiprocessing pipeline even for around-10000x10000 images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories when source is a directory.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip an input file when all expected outputs already exist in directory mode.",
    )
    return parser.parse_args()


class LargeImageReader:
    def __init__(self, path):
        self.path = path
        self.width = None
        self.height = None
        self.backend = None
        self.crs_wkt = None

    def read(self, x1, y1, x2, y2):
        raise NotImplementedError

    def pixel_to_map(self, x, y):
        return float(x), float(y)

    def close(self):
        pass


class RasterioReader(LargeImageReader):
    def __init__(self, path):
        super().__init__(path)
        import rasterio
        from rasterio.windows import Window

        self._Window = Window
        self.ds = rasterio.open(path)
        self.width = self.ds.width
        self.height = self.ds.height
        self.backend = "rasterio"
        self.transform = self.ds.transform
        self.crs_wkt = self.ds.crs.to_wkt() if self.ds.crs else None

    def read(self, x1, y1, x2, y2):
        window = self._Window(x1, y1, x2 - x1, y2 - y1)
        patch = self.ds.read(window=window)
        patch = np.transpose(patch, (1, 2, 0))
        return normalize_patch_channels(patch)

    def pixel_to_map(self, x, y):
        mx, my = self.transform * (x, y)
        return float(mx), float(my)

    def close(self):
        self.ds.close()


class GDALReader(LargeImageReader):
    def __init__(self, path):
        super().__init__(path)
        from osgeo import gdal

        self.ds = gdal.Open(path, gdal.GA_ReadOnly)
        if self.ds is None:
            raise RuntimeError(f"Failed to open image with GDAL: {path}")
        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize
        self.backend = "gdal"
        self.geotransform = self.ds.GetGeoTransform(can_return_null=True)
        self.crs_wkt = self.ds.GetProjection() or None

    def read(self, x1, y1, x2, y2):
        patch = self.ds.ReadAsArray(x1, y1, x2 - x1, y2 - y1)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        else:
            patch = np.transpose(patch, (1, 2, 0))
        return normalize_patch_channels(patch)

    def pixel_to_map(self, x, y):
        if self.geotransform is None:
            return super().pixel_to_map(x, y)
        gt = self.geotransform
        mx = gt[0] + x * gt[1] + y * gt[2]
        my = gt[3] + x * gt[4] + y * gt[5]
        return float(mx), float(my)

    def close(self):
        self.ds = None


class PillowReader(LargeImageReader):
    def __init__(self, path):
        super().__init__(path)
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.backend = "pillow"

    def read(self, x1, y1, x2, y2):
        patch = self.image.crop((x1, y1, x2, y2)).convert("RGB")
        return np.asarray(patch)

    def close(self):
        self.image.close()


def open_reader(path):
    errors = []
    for reader_cls in (RasterioReader, GDALReader, PillowReader):
        try:
            return reader_cls(path)
        except Exception as exc:
            errors.append(f"{reader_cls.__name__}: {exc}")
    raise RuntimeError(
        "Failed to open image with any backend. "
        "Install one of rasterio, gdal, or pillow. "
        f"Details: {' | '.join(errors)}"
    )


def normalize_patch_channels(patch):
    patch = np.asarray(patch)
    if patch.ndim == 2:
        patch = np.repeat(patch[:, :, None], 3, axis=2)
    if patch.shape[2] == 1:
        patch = np.repeat(patch, 3, axis=2)
    elif patch.shape[2] >= 3:
        patch = patch[:, :, :3]

    if patch.dtype != np.uint8:
        patch = np.nan_to_num(patch)
        min_value = float(patch.min()) if patch.size else 0.0
        max_value = float(patch.max()) if patch.size else 0.0
        if max_value <= 255.0 and min_value >= 0.0:
            patch = np.clip(patch, 0, 255).astype(np.uint8)
        else:
            scale = max_value - min_value
            if scale < 1e-6:
                patch = np.zeros_like(patch, dtype=np.uint8)
            else:
                patch = ((patch - min_value) / scale * 255.0).clip(0, 255).astype(np.uint8)

    return np.ascontiguousarray(patch)


def generate_starts(length, tile_size, stride):
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def count_windows(width, height, tile_size, stride):
    return len(generate_starts(width, tile_size, stride)) * len(
        generate_starts(height, tile_size, stride)
    )


def tile_batches(reader, windows, batch_size, tile_size):
    batch_images = []
    batch_windows = []
    for window in windows:
        x1, y1, x2, y2 = window
        patch = reader.read(x1, y1, x2, y2)
        padded = pad_patch_to_size(patch, target_size=tile_size)
        batch_images.append(padded)
        batch_windows.append(window)
        if len(batch_images) == batch_size:
            yield batch_images, batch_windows
            batch_images = []
            batch_windows = []
    if batch_images:
        yield batch_images, batch_windows


def pad_patch_to_size(patch, target_size):
    target_h = max(target_size, patch.shape[0])
    target_w = max(target_size, patch.shape[1])
    if patch.shape[0] == target_h and patch.shape[1] == target_w:
        return np.ascontiguousarray(patch)

    output = np.full((target_h, target_w, patch.shape[2]), 255, dtype=np.uint8)
    output[: patch.shape[0], : patch.shape[1]] = patch
    return np.ascontiguousarray(output)


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = np.maximum(box_area + boxes_area - inter, 1e-12)
    return inter / union


def nms_indices(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        index = order[0]
        keep.append(index)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[index], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return keep


def global_nms(detections, iou_threshold):
    if not detections:
        return detections

    kept = []
    class_ids = sorted({det["class_id"] for det in detections})
    for class_id in class_ids:
        class_dets = [det for det in detections if det["class_id"] == class_id]
        boxes = np.asarray([det["bbox_xyxy"] for det in class_dets], dtype=np.float32)
        scores = np.asarray([det["score"] for det in class_dets], dtype=np.float32)
        keep = nms_indices(boxes, scores, iou_threshold)
        kept.extend(class_dets[idx] for idx in keep)

    kept.sort(key=lambda det: det["score"], reverse=True)
    return kept


def build_test_pipeline(model, first_img):
    cfg = model.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(first_img, np.ndarray):
        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
    return Compose(test_pipeline)


def inference_detector_batch(
    model,
    imgs: ImagesType,
    test_pipeline: Optional[Compose] = None,
) -> Union[DetDataSample, SampleList]:
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    device = model.data_preprocessor.device

    if test_pipeline is None:
        test_pipeline = build_test_pipeline(model, imgs[0])

    if device.type == "cpu":
        for module in model.modules():
            assert not isinstance(
                module, RoIPool
            ), "CPU inference with RoIPool is not supported."

    data_list = []
    for index, img in enumerate(imgs):
        if isinstance(img, np.ndarray):
            data = dict(img=img, img_id=index)
        else:
            data = dict(img_path=img, img_id=index)
        data = test_pipeline(data)
        data_list.append(data)

    batch_inputs = torch.stack([data["inputs"] for data in data_list]).to(device)
    batch_samples = [data["data_samples"] for data in data_list]

    with torch.no_grad():
        result_list = model.test_step({"inputs": batch_inputs, "data_samples": batch_samples})

    return result_list if is_batch else result_list[0]


def get_autocast_ctx(device_type, enabled):
    if not enabled:
        return contextlib.nullcontext()

    if device_type == "cuda":
        return torch.cuda.amp.autocast(enabled=True)

    if device_type == "npu":
        if hasattr(torch, "npu") and hasattr(torch.npu, "amp") and hasattr(torch.npu.amp, "autocast"):
            return torch.npu.amp.autocast(enabled=True)
        return contextlib.nullcontext()

    if hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
        return torch.cpu.amp.autocast(enabled=True)

    return contextlib.nullcontext()


def extract_mask_segment(mask):
    if cv2 is None:
        return None

    mask = np.asarray(mask).astype(np.uint8)
    if mask.ndim != 2 or mask.max() == 0:
        return None

    contours_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if contour.shape[0] < 3:
        return None

    return contour.reshape(-1, 2).astype(np.float32)


def masks_to_numpy(masks):
    if masks is None:
        return None
    if hasattr(masks, "to_ndarray"):
        return masks.to_ndarray()
    if hasattr(masks, "detach"):
        return masks.detach().cpu().numpy()
    return np.asarray(masks)


def convert_results(results, windows, image_width, image_height, class_names, score_thr, classes):
    detections = []
    for result, window in zip(results, windows):
        pred = getattr(result, "pred_instances", None)
        if pred is None or len(pred) == 0:
            continue

        keep = pred.scores >= score_thr
        if classes is not None:
            class_mask = torch.zeros_like(keep, dtype=torch.bool)
            for class_id in classes:
                class_mask |= pred.labels == class_id
            keep &= class_mask

        if keep.sum().item() == 0:
            continue

        boxes = pred.bboxes[keep].detach().cpu().numpy()
        scores = pred.scores[keep].detach().cpu().numpy()
        labels = pred.labels[keep].detach().cpu().numpy().astype(int)

        masks = None
        if hasattr(pred, "masks") and pred.masks is not None:
            try:
                masks = masks_to_numpy(pred.masks[keep])
            except Exception:
                masks = masks_to_numpy(pred.masks)
                if masks is not None and len(masks) == len(getattr(pred, "labels", [])):
                    masks = masks[keep.detach().cpu().numpy()]

        x_offset, y_offset, _, _ = window
        for index, (bbox, score, class_id) in enumerate(zip(boxes, scores, labels)):
            x1, y1, x2, y2 = bbox.tolist()
            gx1 = max(0.0, min(float(image_width), x1 + x_offset))
            gy1 = max(0.0, min(float(image_height), y1 + y_offset))
            gx2 = max(0.0, min(float(image_width), x2 + x_offset))
            gy2 = max(0.0, min(float(image_height), y2 + y_offset))

            segment_xy = None
            if masks is not None and index < len(masks):
                segment = extract_mask_segment(masks[index])
                if segment is not None:
                    segment[:, 0] += x_offset
                    segment[:, 1] += y_offset
                    segment[:, 0] = np.clip(segment[:, 0], 0, image_width)
                    segment[:, 1] = np.clip(segment[:, 1], 0, image_height)
                    segment_xy = segment.tolist()

            class_name = str(class_id)
            if class_names and class_id < len(class_names):
                class_name = str(class_names[class_id])

            detections.append(
                {
                    "bbox_xyxy": [gx1, gy1, gx2, gy2],
                    "score": float(score),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "tile_xyxy": [int(value) for value in window],
                    "segment_xy": segment_xy,
                }
            )

    return detections


def ensure_device(device):
    device_str = str(device)
    if device_str.startswith("npu"):
        if not hasattr(torch, "npu"):
            raise RuntimeError("Requested NPU device, but torch.npu is unavailable.")
        torch.npu.set_device(device_str)
    elif device_str.startswith("cuda") or device_str.startswith("gpu"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda is unavailable.")
        if device_str.startswith("gpu"):
            device_str = device_str.replace("gpu", "cuda", 1)
        torch.cuda.set_device(device_str)


def empty_device_cache(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device.startswith("npu") and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.empty_cache()


def resolve_output_paths(output_arg, output_shp_arg, output_bbox_shp_arg, output_mask_shp_arg):
    output_path = Path(output_arg)
    bbox_shp_path = Path(output_bbox_shp_arg) if output_bbox_shp_arg else None
    mask_shp_path = Path(output_mask_shp_arg) if output_mask_shp_arg else None

    if output_path.suffix.lower() == ".shp":
        if bbox_shp_path is None and output_shp_arg is None:
            bbox_shp_path = output_path
        output_path = output_path.with_suffix(".json")

    if output_shp_arg:
        bbox_shp_path = Path(output_shp_arg)

    return output_path, bbox_shp_path, mask_shp_path


def resolve_output_overlay_path(output_arg, output_mask_overlay_arg):
    if not output_mask_overlay_arg:
        return None

    overlay_path = Path(output_mask_overlay_arg)
    if overlay_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        overlay_path = overlay_path.with_suffix(".png")
    return overlay_path


def iter_input_files(source_path, recursive=False):
    source_path = Path(source_path)
    if source_path.is_file():
        yield source_path
        return

    if not source_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {source_path}")

    pattern = "**/*" if recursive else "*"
    for path in sorted(source_path.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
            yield path


def resolve_batch_output_root(path_arg, default_dir):
    if path_arg is None:
        return Path(default_dir)

    path = Path(path_arg)
    return path.parent if path.suffix else path


def build_batch_output_paths(args, input_file):
    output_root = resolve_batch_output_root(args.output, "./outputs")
    output_root.mkdir(parents=True, exist_ok=True)

    bbox_arg = args.output_bbox_shp or args.output_shp
    bbox_root = resolve_batch_output_root(bbox_arg, output_root) if bbox_arg else None
    mask_root = resolve_batch_output_root(args.output_mask_shp, output_root) if args.output_mask_shp else None

    if bbox_root is not None:
        bbox_root.mkdir(parents=True, exist_ok=True)
    if mask_root is not None:
        mask_root.mkdir(parents=True, exist_ok=True)

    stem = input_file.stem
    output_path = output_root / f"{stem}.json" if not args.no_save_json else None
    output_bbox_shp = bbox_root / f"{stem}_bbox.shp" if bbox_root is not None else None
    output_mask_shp = mask_root / f"{stem}_mask.shp" if mask_root is not None else None
    return output_path, output_bbox_shp, output_mask_shp


def should_skip_outputs(output_path, output_bbox_shp, output_mask_shp):
    expected = [path for path in (output_path, output_bbox_shp, output_mask_shp) if path is not None]
    return bool(expected) and all(path.exists() for path in expected)


def bbox_to_ring_coords(reader, bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return [reader.pixel_to_map(x, y) for x, y in corners]


def mask_to_ring_coords(reader, segment_xy):
    if not segment_xy or len(segment_xy) < 3:
        return None

    ring = [reader.pixel_to_map(x, y) for x, y in segment_xy]
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring if len(ring) >= 4 else None


def _delete_shapefile_sidecars(shp_path):
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        path = shp_path.with_suffix(suffix)
        if path.exists():
            path.unlink()


def _dbf_field_descriptor(name, field_type, length, decimal_count=0):
    name_bytes = name.encode("ascii")[:10]
    name_bytes = name_bytes + b"\x00" * (11 - len(name_bytes))
    return (
        name_bytes
        + field_type.encode("ascii")
        + b"\x00" * 4
        + bytes([length])
        + bytes([decimal_count])
        + b"\x00" * 14
    )


def _dbf_format_value(value, field_type, length, decimal_count=0):
    if field_type == "C":
        text = str(value)[:length]
        return text.ljust(length).encode("utf-8", errors="ignore")[:length]
    if field_type == "N":
        if decimal_count > 0:
            text = f"{float(value):>{length}.{decimal_count}f}"
        else:
            text = f"{int(value):>{length}d}"
        return text.encode("ascii")[:length].rjust(length, b" ")
    raise ValueError(f"Unsupported DBF field type: {field_type}")


def _write_dbf(dbf_path, detections):
    fields = [
        ("class_id", "N", 10, 0),
        ("class_name", "C", 64, 0),
        ("score", "N", 18, 8),
        ("x1", "N", 18, 6),
        ("y1", "N", 18, 6),
        ("x2", "N", 18, 6),
        ("y2", "N", 18, 6),
    ]
    num_records = len(detections)
    num_fields = len(fields)
    header_length = 32 + 32 * num_fields + 1
    record_length = 1 + sum(field[2] for field in fields)
    today = dt.date.today()

    with dbf_path.open("wb") as file:
        file.write(
            struct.pack(
                "<BBBBLHH20x",
                0x03,
                today.year - 1900,
                today.month,
                today.day,
                num_records,
                header_length,
                record_length,
            )
        )
        for field in fields:
            file.write(_dbf_field_descriptor(*field))
        file.write(b"\r")

        for det in detections:
            file.write(b" ")
            values = {
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "score": det["score"],
                "x1": det["bbox_xyxy"][0],
                "y1": det["bbox_xyxy"][1],
                "x2": det["bbox_xyxy"][2],
                "y2": det["bbox_xyxy"][3],
            }
            for name, field_type, length, decimal_count in fields:
                file.write(
                    _dbf_format_value(values[name], field_type, length, decimal_count)
                )
        file.write(b"\x1a")


def _build_polygon_record(ring_coords):
    xs = [point[0] for point in ring_coords]
    ys = [point[1] for point in ring_coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    num_parts = 1
    num_points = len(ring_coords)

    content = bytearray()
    content += struct.pack("<i", 5)
    content += struct.pack("<4d", *bbox)
    content += struct.pack("<2i", num_parts, num_points)
    content += struct.pack("<i", 0)
    for x, y in ring_coords:
        content += struct.pack("<2d", float(x), float(y))
    return bytes(content), bbox


def _write_shp_shx(shp_path, detections, geometry_builder):
    shape_type = 5
    shp_records = []
    shx_records = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    offset_words = 50

    record_number = 0
    for det in detections:
        ring_coords = geometry_builder(det)
        if ring_coords is None:
            continue

        record_number += 1
        content, bbox = _build_polygon_record(ring_coords)
        content_length_words = len(content) // 2
        shp_records.append(struct.pack(">2i", record_number, content_length_words) + content)
        shx_records.append(struct.pack(">2i", offset_words, content_length_words))
        offset_words += 4 + content_length_words
        xmin = min(xmin, bbox[0])
        ymin = min(ymin, bbox[1])
        xmax = max(xmax, bbox[2])
        ymax = max(ymax, bbox[3])

    if not shp_records:
        xmin = ymin = xmax = ymax = 0.0

    shp_file_length_words = 50 + sum(len(record) for record in shp_records) // 2
    shx_file_length_words = 50 + len(shx_records) * 4

    def file_header(file_length_words):
        return (
            struct.pack(">i", 9994)
            + b"\x00" * 20
            + struct.pack(">i", file_length_words)
            + struct.pack("<2i4d4d", 1000, shape_type, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0)
        )

    with shp_path.open("wb") as shp_file:
        shp_file.write(file_header(shp_file_length_words))
        for record in shp_records:
            shp_file.write(record)

    with shp_path.with_suffix(".shx").open("wb") as shx_file:
        shx_file.write(file_header(shx_file_length_words))
        for record in shx_records:
            shx_file.write(record)


def _write_prj(prj_path, crs_wkt):
    if not crs_wkt:
        return
    with prj_path.open("w", encoding="utf-8") as file:
        file.write(crs_wkt)


def _overlay_color_for_class(class_id):
    palette = [
        (255, 80, 80),
        (80, 255, 80),
        (80, 80, 255),
        (255, 200, 80),
        (255, 80, 200),
        (80, 200, 255),
    ]
    return palette[int(class_id) % len(palette)]


def save_mask_overlay_image(image_path, detections, output_path, alpha=0.35):
    if cv2 is None:
        print("[warn] cv2 is unavailable; skip saving mask overlay image.")
        return

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[warn] failed to read image for overlay: {image_path}")
        return

    overlay = image.copy()
    for det in detections:
        segment = det.get("segment_xy_regularized") or det.get("segment_xy")
        if not segment or len(segment) < 3:
            continue
        pts = np.asarray(segment, dtype=np.int32).reshape(-1, 1, 2)
        color = _overlay_color_for_class(det.get("class_id", 0))
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)

    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), blended)


def write_shapefile(shp_path, detections, reader, geometry="bbox"):
    """Thin wrapper around :func:`shapefile_export.write_shapefile`.

    Kept here for backward compatibility — the actual implementation now lives
    in :mod:`shapefile_export` and adds per-detection tqdm progress plus
    vectorised pixel→map conversion.
    """
    from .shapefile_export import write_shapefile as _shared_write_shapefile
    _shared_write_shapefile(shp_path, detections, reader, geometry=geometry)


def build_worker_devices(config):
    base_devices = config.devices if config.devices else [config.device]
    if not base_devices:
        base_devices = ["cpu"]

    worker_count = config.worker_count
    if worker_count is None:
        worker_count = len(base_devices) if config.devices else 2
    if worker_count <= 0:
        raise ValueError("worker_count must be positive.")

    return [base_devices[index % len(base_devices)] for index in range(worker_count)]


def validate_config(config):
    if config.reader_workers <= 0:
        raise ValueError("reader_workers must be positive.")
    if config.tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.queue_size <= 0:
        raise ValueError("queue_size must be positive.")
    if (config.stride or config.tile_size) <= 0:
        raise ValueError("stride must be positive.")
    if not config.config_file:
        raise ValueError("config_file must be provided.")
    if not config.checkpoint_file:
        raise ValueError("checkpoint_file must be provided.")


def partitioned_windows(width, height, tile_size, stride, reader_id, reader_count):
    x_starts = generate_starts(width, tile_size, stride)
    y_starts = generate_starts(height, tile_size, stride)
    rows_per_reader = max(1, math.ceil(len(y_starts) / reader_count))
    y_begin = reader_id * rows_per_reader
    y_end = min(len(y_starts), y_begin + rows_per_reader)

    for y1 in y_starts[y_begin:y_end]:
        y2 = min(y1 + tile_size, height)
        for x1 in x_starts:
            x2 = min(x1 + tile_size, width)
            yield (x1, y1, x2, y2)


def tile_reader_process(
    reader_id,
    source_path,
    tile_size,
    stride,
    batch_size,
    reader_count,
    active_readers,
    image_queue,
    message_queue,
    error_queue,
):
    reader = None
    try:
        set_process_title(read_title("rtmdet", reader_id=reader_id))
        message_queue.put(
            {
                "type": "reader_ready",
                "reader_id": reader_id,
                "pid": os.getpid(),
            }
        )
        reader = open_reader(source_path)
        windows = partitioned_windows(
            reader.width, reader.height, tile_size, stride, reader_id, reader_count
        )
        for batch_index, (batch_images, batch_windows) in enumerate(
            tile_batches(reader, windows, batch_size, tile_size)
        ):
            image_queue.put(
                {
                    "batch_index": batch_index,
                    "images": batch_images,
                    "windows": batch_windows,
                    "reader_id": reader_id,
                }
            )
            message_queue.put(
                {
                    "type": "read",
                    "reader_id": reader_id,
                    "tiles": len(batch_windows),
                }
            )
    except Exception:
        error_queue.put(
            {
                "stage": f"reader:{reader_id}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if reader is not None:
            reader.close()
        with active_readers.get_lock():
            active_readers.value -= 1
        message_queue.put({"type": "reader_done", "reader_id": reader_id})


def predictor_process(
    worker_id,
    device,
    config_file,
    checkpoint_file,
    image_width,
    image_height,
    predict_kwargs,
    active_readers,
    image_queue,
    message_queue,
    error_queue,
):
    try:
        set_process_title(infer_title("rtmdet", device=device, worker_id=worker_id))
        ensure_device(device)
        current_device = device
        if device.startswith("npu") and hasattr(torch, "npu"):
            try:
                current_device = f"npu:{torch.npu.current_device()}"
            except Exception:
                current_device = device
        message_queue.put(
            {
                "type": "worker_ready",
                "worker_id": worker_id,
                "device": device,
                "current_device": current_device,
                "pid": os.getpid(),
            }
        )
        model = init_detector(config_file, checkpoint_file, device=device)
        class_names = model.dataset_meta.get("classes", []) if hasattr(model, "dataset_meta") else []
        test_pipeline = None

        while True:
            try:
                task = image_queue.get(timeout=1)
            except queue.Empty:
                if active_readers.value == 0:
                    break
                continue

            if test_pipeline is None and task["images"]:
                test_pipeline = build_test_pipeline(model, task["images"][0])

            with torch.inference_mode():
                with get_autocast_ctx(model.data_preprocessor.device.type, predict_kwargs["fp16"]):
                    results = inference_detector_batch(
                        model=model,
                        imgs=task["images"],
                        test_pipeline=test_pipeline,
                    )

            detections = convert_results(
                results=results,
                windows=task["windows"],
                image_width=image_width,
                image_height=image_height,
                class_names=class_names,
                score_thr=predict_kwargs["score_thr"],
                classes=predict_kwargs["classes"],
            )

            message_queue.put(
                {
                    "type": "result",
                    "worker_id": worker_id,
                    "batch_index": task["batch_index"],
                    "tiles": len(task["windows"]),
                    "detections": detections,
                }
            )

            del results
            gc.collect()
            empty_device_cache(device)
    except Exception:
        error_queue.put(
            {
                "stage": f"worker:{worker_id}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        message_queue.put({"type": "worker_done", "worker_id": worker_id, "device": device})


def get_error_nowait(error_queue):
    try:
        return error_queue.get_nowait()
    except queue.Empty:
        return None


def terminate_processes(processes):
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2)


def join_processes(processes):
    for process in processes:
        process.join()


def safe_qsize(queue_obj):
    try:
        return queue_obj.qsize()
    except (NotImplementedError, AttributeError, OSError):
        return None


def resolve_runtime_output_paths(config):
    output_arg = config.output or "./outputs/large_image_predictions.json"
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_output_paths(
        output_arg, config.output_shp, config.output_bbox_shp, config.output_mask_shp
    )
    output_mask_overlay_path = None
    if getattr(config, "save_mask_overlay", False) and output_path is not None:
        output_mask_overlay_path = output_path.with_name(f"{output_path.stem}_mask_overlay.png")
    if not config.save_json:
        output_path = None
    return output_path, output_bbox_shp_path, output_mask_shp_path, output_mask_overlay_path


def should_use_small_image_fastpath(config, reader):
    if not config.prefer_single_process_for_small:
        return False
    threshold = max(int(config.small_image_threshold), 1)
    return reader.width <= threshold and reader.height <= threshold


def should_use_direct_image_fastpath(config, reader):
    if not config.prefer_single_process_for_small:
        return False

    threshold = max(int(config.small_image_threshold), 1)
    if reader.width > threshold or reader.height > threshold:
        return False

    pixel_count = int(reader.width) * int(reader.height)
    direct_limit = min(threshold * threshold, 40_000_000)
    return pixel_count <= direct_limit


def maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path):
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_bbox_shp_path is not None:
        output_bbox_shp_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mask_shp_path is not None:
        output_mask_shp_path.parent.mkdir(parents=True, exist_ok=True)


def load_reusable_model(config, worker_device):
    ensure_device(worker_device)
    model = init_detector(config.config_file, config.checkpoint_file, device=worker_device)
    class_names = model.dataset_meta.get("classes", []) if hasattr(model, "dataset_meta") else []
    return ReusableMMDetModel(
        model=model,
        class_names=class_names,
        device=worker_device,
        test_pipeline=None,
    )


def dump_output_files(
    output,
    merged_detections,
    meta_reader,
    output_path,
    output_bbox_shp_path,
    output_mask_shp_path,
    output_mask_overlay_path,
    global_nms_enabled,
    raw_detection_count,
    regularize=None,
):
    # 先做正则化（会改写 merged_detections，写入 segment_xy_regularized），
    # 这样 JSON 与可视化都能反映正则化结果。
    if output_mask_shp_path is not None and regularize:
        from .building_regularize import write_regularized_shapefile
        write_regularized_shapefile(
            output_mask_shp_path, merged_detections, meta_reader, regularize
        )
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
        print(f"[info] saved merged predictions to {output_path}")
    if output_bbox_shp_path is not None:
        write_shapefile(output_bbox_shp_path, merged_detections, meta_reader, geometry="bbox")
        if meta_reader.crs_wkt:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} with source CRS")
        else:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} using pixel coordinates")
    if output_mask_shp_path is not None:
        write_shapefile(output_mask_shp_path, merged_detections, meta_reader, geometry="mask")
        mask_count = sum(1 for det in merged_detections if det.get("segment_xy"))
        if meta_reader.crs_wkt:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} with source CRS "
                f"({mask_count} polygons)"
            )
        else:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} using pixel coordinates "
                f"({mask_count} polygons)"
            )
    if output_mask_overlay_path is not None:
        save_mask_overlay_image(output["image"], merged_detections, output_mask_overlay_path)
        print(f"[info] saved mask overlay image to {output_mask_overlay_path}")

    if global_nms_enabled:
        print(
            f"[info] detections before global NMS: {raw_detection_count}, "
            f"after global NMS: {len(merged_detections)}"
        )
    else:
        print(f"[info] detections exported without extra global NMS: {len(merged_detections)}")


def run_single_process_inference(config, meta_reader, stride, worker_device, reusable_model=None):
    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path, output_mask_overlay_path = resolve_runtime_output_paths(config)
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    print(
        f"[info] single-process tiled path enabled for {meta_reader.width}x{meta_reader.height}. "
        f"Using single-process batched tile inference on {worker_device}."
    )

    reusable_model = reusable_model or load_reusable_model(config, worker_device)
    model = reusable_model.model
    class_names = reusable_model.class_names

    total_tiles = count_windows(meta_reader.width, meta_reader.height, config.tile_size, stride)
    reporter = ProgressReporter(total_tiles)
    all_detections = []
    test_pipeline = reusable_model.test_pipeline
    processed_tiles = 0

    try:
        windows = partitioned_windows(
            meta_reader.width,
            meta_reader.height,
            config.tile_size,
            stride,
            0,
            1,
        )
        for batch_images, batch_windows in tile_batches(
            meta_reader, windows, config.batch_size, config.tile_size
        ):
            if test_pipeline is None and batch_images:
                test_pipeline = build_test_pipeline(model, batch_images[0])
                reusable_model.test_pipeline = test_pipeline

            with torch.inference_mode():
                with get_autocast_ctx(model.data_preprocessor.device.type, config.fp16):
                    results = inference_detector_batch(
                        model=model,
                        imgs=batch_images,
                        test_pipeline=test_pipeline,
                    )

            detections = convert_results(
                results=results,
                windows=batch_windows,
                image_width=meta_reader.width,
                image_height=meta_reader.height,
                class_names=class_names,
                score_thr=config.score_thr,
                classes=config.classes,
            )
            all_detections.extend(detections)
            processed_tiles += len(batch_windows)
            reporter.update_read(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))
            reporter.update_infer(processed_tiles, ready_workers=1, total_workers=1, detections=len(all_detections))

            del results
            gc.collect()
            empty_device_cache(worker_device)
    finally:
        reporter.close()

    if config.global_nms:
        merged_detections = global_nms(all_detections, config.global_iou)
    else:
        merged_detections = all_detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "config_file": config.config_file,
        "checkpoint_file": config.checkpoint_file,
        "devices": [worker_device],
        "reader_workers": 1,
        "worker_count": 1,
        "score_thr": config.score_thr,
        "tile_size": config.tile_size,
        "stride": stride,
        "batch_size": config.batch_size,
        "queue_size": 0,
        "tiles": total_tiles,
        "global_nms": bool(config.global_nms),
        "raw_detections": len(all_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    dump_output_files(
        output=output,
        merged_detections=merged_detections,
        meta_reader=meta_reader,
        output_path=output_path,
        output_bbox_shp_path=output_bbox_shp_path,
        output_mask_shp_path=output_mask_shp_path,
        output_mask_overlay_path=output_mask_overlay_path,
        global_nms_enabled=config.global_nms,
        raw_detection_count=len(all_detections),
        regularize=config.regularize,
    )
    return output


def run_direct_image_inference(config, meta_reader, worker_device, reusable_model=None):
    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path, output_mask_overlay_path = resolve_runtime_output_paths(config)
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    print(
        f"[info] direct-image fast path enabled for {meta_reader.width}x{meta_reader.height}. "
        f"Using single-image inference on {worker_device}."
    )

    reusable_model = reusable_model or load_reusable_model(config, worker_device)
    model = reusable_model.model
    class_names = reusable_model.class_names

    reporter = ProgressReporter(1)
    full_image = meta_reader.read(0, 0, meta_reader.width, meta_reader.height)
    reporter.update_read(1, ready_workers=1, total_workers=1, detections=0)

    if reusable_model.test_pipeline is None:
        reusable_model.test_pipeline = build_test_pipeline(model, full_image)

    with torch.inference_mode():
        with get_autocast_ctx(model.data_preprocessor.device.type, config.fp16):
            result = inference_detector_batch(
                model=model,
                imgs=full_image,
                test_pipeline=reusable_model.test_pipeline,
            )

    detections = convert_results(
        results=[result],
        windows=[(0, 0, meta_reader.width, meta_reader.height)],
        image_width=meta_reader.width,
        image_height=meta_reader.height,
        class_names=class_names,
        score_thr=config.score_thr,
        classes=config.classes,
    )
    reporter.update_infer(1, ready_workers=1, total_workers=1, detections=len(detections))
    reporter.close()

    gc.collect()
    empty_device_cache(worker_device)

    if config.global_nms:
        merged_detections = global_nms(detections, config.global_iou)
    else:
        merged_detections = detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "config_file": config.config_file,
        "checkpoint_file": config.checkpoint_file,
        "devices": [worker_device],
        "reader_workers": 0,
        "worker_count": 1,
        "score_thr": config.score_thr,
        "tile_size": None,
        "stride": None,
        "batch_size": 1,
        "queue_size": 0,
        "tiles": 1,
        "inference_mode": "direct_image",
        "global_nms": bool(config.global_nms),
        "raw_detections": len(detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    dump_output_files(
        output=output,
        merged_detections=merged_detections,
        meta_reader=meta_reader,
        output_path=output_path,
        output_bbox_shp_path=output_bbox_shp_path,
        output_mask_shp_path=output_mask_shp_path,
        output_mask_overlay_path=output_mask_overlay_path,
        global_nms_enabled=config.global_nms,
        raw_detection_count=len(detections),
        regularize=config.regularize,
    )
    return output


def run_multiprocess_inference(config: MultiprocessMMDetInferConfig):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_devices = build_worker_devices(config)

    if len(set(worker_devices)) == 1 and len(worker_devices) > 1:
        print(
            f"[warn] {len(worker_devices)} predictor processes share the same device "
            f"{worker_devices[0]}. This may reduce throughput instead of speeding it up."
        )
    if config.reader_workers > 1:
        print(
            f"[info] using {config.reader_workers} reader processes for tile extraction. "
            "Readers use CPU and storage I/O rather than NPU."
        )

    source_path = Path(config.source)
    output_path, output_bbox_shp_path, output_mask_shp_path, output_mask_overlay_path = resolve_runtime_output_paths(
        config
    )
    maybe_prepare_output_dirs(output_path, output_bbox_shp_path, output_mask_shp_path)

    meta_reader = open_reader(str(source_path))
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print(
            "[warn] no georeference was read from the source image. "
            "The output shapefile will use pixel coordinates."
        )

    if should_use_small_image_fastpath(config, meta_reader):
        output = run_single_process_inference(
            config=config,
            meta_reader=meta_reader,
            stride=stride,
            worker_device=worker_devices[0],
        )
        meta_reader.close()
        return output

    total_tiles = count_windows(meta_reader.width, meta_reader.height, config.tile_size, stride)
    total_row_strips = len(generate_starts(meta_reader.height, config.tile_size, stride))
    print(
        f"[info] total tiles: {total_tiles}, tile_size={config.tile_size}, stride={stride}, "
        f"readers={config.reader_workers}, workers={len(worker_devices)}, devices={worker_devices}"
    )
    if config.reader_workers > total_row_strips:
        print(
            f"[warn] reader_workers={config.reader_workers} exceeds tile rows={total_row_strips}. "
            f"Extra readers will sit idle."
        )

    ctx = mp.get_context(config.mp_start)
    image_queue = ctx.Queue(maxsize=config.queue_size)
    message_queue = ctx.Queue(
        maxsize=max(config.queue_size, len(worker_devices), config.reader_workers) * 4
    )
    error_queue = ctx.Queue()
    active_readers = ctx.Value("i", config.reader_workers)

    predict_kwargs = {
        "score_thr": config.score_thr,
        "classes": config.classes,
        "fp16": config.fp16,
    }

    processes = []
    for reader_id in range(config.reader_workers):
        process = ctx.Process(
            target=tile_reader_process,
            args=(
                reader_id,
                str(source_path),
                config.tile_size,
                stride,
                config.batch_size,
                config.reader_workers,
                active_readers,
                image_queue,
                message_queue,
                error_queue,
            ),
            name=f"mmdet-read-{reader_id}",
        )
        process.daemon = True
        processes.append(process)

    for worker_id, device in enumerate(worker_devices):
        process = ctx.Process(
            target=predictor_process,
            args=(
                worker_id,
                device,
                config.config_file,
                config.checkpoint_file,
                meta_reader.width,
                meta_reader.height,
                predict_kwargs,
                active_readers,
                image_queue,
                message_queue,
                error_queue,
            ),
            name=f"mmdet-infer-{worker_id}",
        )
        process.daemon = True
        processes.append(process)

    for process in processes:
        process.start()

    all_detections = []
    read_tiles = 0
    inferred_tiles = 0
    finished_workers = 0
    ready_readers = 0
    ready_workers = 0
    reporter = ProgressReporter(total_tiles)

    try:
        while finished_workers < len(worker_devices):
            err = get_error_nowait(error_queue)
            if err is not None:
                raise RuntimeError(
                    f"Multiprocess inference failed in {err['stage']}:\n{err['traceback']}"
                )

            try:
                message = message_queue.get(timeout=1)
            except queue.Empty:
                continue

            if message["type"] == "read":
                read_tiles += message["tiles"]
                reporter.update_read(
                    read_tiles,
                    queue_size=safe_qsize(image_queue),
                    ready_workers=ready_workers,
                    total_workers=len(worker_devices),
                )
            elif message["type"] == "reader_ready":
                ready_readers += 1
                reporter.write(
                    f"[info] reader worker {message['reader_id']} ready "
                    f"(pid={message['pid']}) [{ready_readers}/{config.reader_workers}]",
                )
            elif message["type"] == "worker_ready":
                ready_workers += 1
                reporter.write(
                    f"[info] predictor worker {message['worker_id']} ready "
                    f"on {message['current_device']} "
                    f"(requested={message['device']}, pid={message['pid']}) "
                    f"[{ready_workers}/{len(worker_devices)}]",
                )
            elif message["type"] == "result":
                all_detections.extend(message["detections"])
                inferred_tiles += message["tiles"]
                reporter.update_infer(
                    inferred_tiles,
                    queue_size=safe_qsize(image_queue),
                    ready_workers=ready_workers,
                    total_workers=len(worker_devices),
                )
            elif message["type"] == "worker_done":
                finished_workers += 1
    except KeyboardInterrupt:
        terminate_processes(processes)
        raise
    except Exception:
        terminate_processes(processes)
        raise
    finally:
        reporter.close()

    join_processes(processes)

    if config.global_nms:
        merged_detections = global_nms(all_detections, config.global_iou)
    else:
        merged_detections = all_detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": meta_reader.backend,
        "image_size": {"width": meta_reader.width, "height": meta_reader.height},
        "config_file": config.config_file,
        "checkpoint_file": config.checkpoint_file,
        "devices": worker_devices,
        "reader_workers": config.reader_workers,
        "worker_count": len(worker_devices),
        "score_thr": config.score_thr,
        "tile_size": config.tile_size,
        "stride": stride,
        "batch_size": config.batch_size,
        "queue_size": config.queue_size,
        "tiles": total_tiles,
        "global_nms": bool(config.global_nms),
        "raw_detections": len(all_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    dump_output_files(
        output=output,
        merged_detections=merged_detections,
        meta_reader=meta_reader,
        output_path=output_path,
        output_bbox_shp_path=output_bbox_shp_path,
        output_mask_shp_path=output_mask_shp_path,
        output_mask_overlay_path=output_mask_overlay_path,
        global_nms_enabled=config.global_nms,
        raw_detection_count=len(all_detections),
        regularize=config.regularize,
    )

    meta_reader.close()
    return output


def namespace_to_config(args):
    return MultiprocessMMDetInferConfig(
        source=args.source,
        config_file=args.config_file,
        checkpoint_file=args.checkpoint_file,
        output=args.output,
        output_shp=args.output_shp,
        output_bbox_shp=args.output_bbox_shp,
        output_mask_shp=args.output_mask_shp,
        save_json=not args.no_save_json,
        device=args.device,
        devices=args.devices,
        worker_count=args.worker_count,
        reader_workers=args.reader_workers,
        queue_size=args.queue_size,
        mp_start=args.mp_start,
        score_thr=args.score_thr,
        tile_size=args.tile_size,
        stride=args.stride,
        batch_size=args.batch_size,
        global_nms=args.global_nms,
        global_iou=args.global_iou,
        classes=args.classes,
        fp16=args.fp16,
        small_image_threshold=args.small_image_threshold,
        prefer_single_process_for_small=not args.disable_small_image_fastpath,
        save_mask_overlay=args.save_mask_overlay,
    )


def run_cli_for_single_input(args, input_file, output_path=None, output_bbox_shp=None, output_mask_shp=None):
    config = namespace_to_config(args)
    config.source = str(input_file)
    if output_path is not None:
        config.output = str(output_path)
    if output_bbox_shp is not None:
        config.output_bbox_shp = str(output_bbox_shp)
        config.output_shp = None
    else:
        config.output_bbox_shp = None
        config.output_shp = None
    config.output_mask_shp = str(output_mask_shp) if output_mask_shp is not None else None
    return run_multiprocess_inference(config)


def run_reused_single_process_inference(config, reusable_model):
    validate_config(config)
    stride = config.stride or config.tile_size
    worker_devices = build_worker_devices(config)
    worker_device = worker_devices[0]
    source_path = Path(config.source)

    meta_reader = open_reader(str(source_path))
    print(
        f"[info] opened image with {meta_reader.backend}: "
        f"{source_path} ({meta_reader.width}x{meta_reader.height})"
    )
    if meta_reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print(
            "[warn] no georeference was read from the source image. "
            "The output shapefile will use pixel coordinates."
        )

    try:
        if should_use_small_image_fastpath(config, meta_reader):
            return run_single_process_inference(
                config=config,
                meta_reader=meta_reader,
                stride=stride,
                worker_device=worker_device,
                reusable_model=reusable_model,
            )

        return run_multiprocess_inference(config)
    finally:
        meta_reader.close()


def main():
    args = parse_args()
    source_path = Path(args.source)

    if source_path.is_file():
        run_cli_for_single_input(args, source_path)
        return

    input_files = list(iter_input_files(source_path, recursive=args.recursive))
    if not input_files:
        raise RuntimeError(
            f"No supported input files were found in {source_path}. "
            f"Supported suffixes: {sorted(SUPPORTED_INPUT_SUFFIXES)}"
        )

    print(f"[info] found {len(input_files)} input files")
    for index, input_file in enumerate(input_files, start=1):
        output_path, output_bbox_shp, output_mask_shp = build_batch_output_paths(args, input_file)
        if args.skip_existing and should_skip_outputs(output_path, output_bbox_shp, output_mask_shp):
            print(f"[skip] {index}/{len(input_files)} {input_file.name}: outputs already exist")
            continue

        print(f"[info] processing {index}/{len(input_files)}: {input_file}")
        run_cli_for_single_input(
            args,
            input_file,
            output_path=output_path,
            output_bbox_shp=output_bbox_shp,
            output_mask_shp=output_mask_shp,
        )


if __name__ == "__main__":
    main()
