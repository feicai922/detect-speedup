# -*- coding: utf-8 -*-
from . import runtime_warning_filters  # noqa: F401
import argparse
import datetime as dt
import json
import struct
import warnings
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings(
    "ignore",
    message=r"Warning: The .* owner does not match the current owner\.",
    category=UserWarning,
    module=r"torch_npu\.utils\.collect_env",
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch.Tensor.cuda and torch.nn.Module.cuda are replaced.*",
    category=ImportWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*torch.jit.script and torch.jit.script_method will be disabled.*",
    category=RuntimeWarning,
)

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Large-image YOLO inference with tile-by-tile reading."
    )
    parser.add_argument("source", type=str, help="Path to a large image file.")
    parser.add_argument(
        "--model",
        type=str,
        default="best.pt",
        help="YOLO model checkpoint path.",
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
        "--device",
        type=str,
        default="npu:0" if hasattr(torch, "npu") else "cpu",
        help="Inference device, for example cpu, cuda:0, npu:0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold used inside YOLO prediction.",
    )
    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Max detections per tile.",
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
        help="Number of tiles per YOLO batch.",
    )
    parser.add_argument(
        "--global-nms",
        action="store_true",
        help="Apply an extra cross-tile global NMS after YOLO prediction. Disabled by default.",
    )
    parser.add_argument(
        "--global-iou",
        type=float,
        default=0.45,
        help="IoU threshold for optional cross-tile global NMS.",
    )
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS inside YOLO prediction.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Optional class IDs to keep.",
    )
    return parser.parse_args()


class LargeImageReader:
    def __init__(self, path, stretch_to_uint8=False, output_color="rgb"):
        self.path = path
        self.stretch_to_uint8 = stretch_to_uint8
        self.output_color = output_color
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
    def __init__(self, path, stretch_to_uint8=False, output_color="rgb"):
        super().__init__(path, stretch_to_uint8=stretch_to_uint8, output_color=output_color)
        import rasterio
        from rasterio.windows import Window

        self._rasterio = rasterio
        self._Window = Window
        self.ds = rasterio.open(path)
        self.width = self.ds.width
        self.height = self.ds.height
        self.count = self.ds.count
        self.backend = "rasterio"
        self.transform = self.ds.transform
        self.crs_wkt = self.ds.crs.to_wkt() if self.ds.crs else None

    def read(self, x1, y1, x2, y2):
        window = self._Window(x1, y1, x2 - x1, y2 - y1)
        patch = self.ds.read(window=window)
        patch = np.transpose(patch, (1, 2, 0))
        return maybe_convert_output_color(
            normalize_patch_channels(patch, stretch=self.stretch_to_uint8), self.output_color
        )

    def pixel_to_map(self, x, y):
        mx, my = self.transform * (x, y)
        return float(mx), float(my)

    def close(self):
        self.ds.close()


class GDALReader(LargeImageReader):
    def __init__(self, path, stretch_to_uint8=False, output_color="rgb"):
        super().__init__(path, stretch_to_uint8=stretch_to_uint8, output_color=output_color)
        from osgeo import gdal

        self.gdal = gdal
        self.ds = gdal.Open(path, gdal.GA_ReadOnly)
        if self.ds is None:
            raise RuntimeError(f"Failed to open image with GDAL: {path}")
        self.width = self.ds.RasterXSize
        self.height = self.ds.RasterYSize
        self.count = self.ds.RasterCount
        self.backend = "gdal"
        self.geotransform = self.ds.GetGeoTransform(can_return_null=True)
        self.crs_wkt = self.ds.GetProjection() or None

    def read(self, x1, y1, x2, y2):
        patch = self.ds.ReadAsArray(x1, y1, x2 - x1, y2 - y1)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        else:
            patch = np.transpose(patch, (1, 2, 0))
        return maybe_convert_output_color(
            normalize_patch_channels(patch, stretch=self.stretch_to_uint8), self.output_color
        )

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
    def __init__(self, path, stretch_to_uint8=False, output_color="rgb"):
        super().__init__(path, stretch_to_uint8=stretch_to_uint8, output_color=output_color)
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        self._Image = Image
        self.image = Image.open(path)
        self.width, self.height = self.image.size
        self.backend = "pillow"

    def read(self, x1, y1, x2, y2):
        patch = self.image.crop((x1, y1, x2, y2)).convert("RGB")
        return maybe_convert_output_color(np.asarray(patch), self.output_color)

    def close(self):
        self.image.close()


def open_reader(path, stretch_to_uint8=False, output_color="rgb"):
    errors = []
    for reader_cls in (RasterioReader, GDALReader, PillowReader):
        try:
            return reader_cls(path, stretch_to_uint8=stretch_to_uint8, output_color=output_color)
        except Exception as exc:
            errors.append(f"{reader_cls.__name__}: {exc}")
    raise RuntimeError(
        "Failed to open image with any backend. "
        "Install one of rasterio, gdal, or pillow. "
        f"Details: {' | '.join(errors)}"
    )


def maybe_convert_output_color(patch, output_color):
    output_color = (output_color or "rgb").lower()
    if output_color == "rgb":
        return patch
    if output_color == "bgr":
        return patch[:, :, ::-1].copy()
    raise ValueError("Unsupported output_color: {}".format(output_color))


def normalize_patch_channels(patch, stretch=False):
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
        elif not stretch:
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


def generate_windows(width, height, tile_size, stride):
    x_starts = generate_starts(width, tile_size, stride)
    y_starts = generate_starts(height, tile_size, stride)

    for y1 in y_starts:
        for x1 in x_starts:
            x2 = min(x1 + tile_size, width)
            y2 = min(y1 + tile_size, height)
            yield (x1, y1, x2, y2)


def count_windows(width, height, tile_size, stride):
    return len(generate_starts(width, tile_size, stride)) * len(
        generate_starts(height, tile_size, stride)
    )


def tile_batches(reader, windows, batch_size):
    batch_images = []
    batch_windows = []
    for window in windows:
        x1, y1, x2, y2 = window
        batch_images.append(reader.read(x1, y1, x2, y2))
        batch_windows.append(window)
        if len(batch_images) == batch_size:
            yield batch_images, batch_windows
            batch_images = []
            batch_windows = []
    if batch_images:
        yield batch_images, batch_windows


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(box_area + boxes_area - inter, 1e-12)
    return inter / union


def nms_indices(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = compute_iou(boxes[i], boxes[remaining])
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


def clip_image_point(x, y, image_width, image_height):
    return [
        max(0.0, min(float(image_width), float(x))),
        max(0.0, min(float(image_height), float(y))),
    ]


def class_name_from_id(names, class_id):
    if isinstance(names, dict):
        return str(names.get(int(class_id), int(class_id)))
    return str(names[int(class_id)])


def append_detection(
    detections,
    names,
    window,
    image_width,
    image_height,
    xyxy,
    score,
    class_id,
    segment_xy=None,
    obb_xyxyxyxy=None,
):
    x_offset, y_offset, _, _ = window
    x1, y1, x2, y2 = xyxy.tolist()
    gx1, gy1 = clip_image_point(x1 + x_offset, y1 + y_offset, image_width, image_height)
    gx2, gy2 = clip_image_point(x2 + x_offset, y2 + y_offset, image_width, image_height)
    detection = {
        "bbox_xyxy": [gx1, gy1, gx2, gy2],
        "score": float(score),
        "class_id": int(class_id),
        "class_name": class_name_from_id(names, class_id),
        "tile_xyxy": [int(v) for v in window],
        "segment_xy": segment_xy,
    }
    if obb_xyxyxyxy is not None:
        detection["obb_xyxyxyxy"] = obb_xyxyxyxy
        detection["is_obb"] = True
    detections.append(detection)


def convert_results(results, windows, image_width, image_height):
    detections = []
    for result, window in zip(results, windows):
        x_offset, y_offset, _, _ = window
        names = result.names
        obb = getattr(result, "obb", None)
        if obb is not None and len(obb) > 0:
            xyxy_list = obb.xyxy.detach().cpu().numpy()
            conf_list = obb.conf.detach().cpu().numpy()
            cls_list = obb.cls.detach().cpu().numpy().astype(int)
            polygon_list = None
            if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
                polygon_list = obb.xyxyxyxy.detach().cpu().numpy()

            for idx, (xyxy, score, class_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
                segment_xy = None
                if polygon_list is not None and idx < len(polygon_list):
                    points = np.asarray(polygon_list[idx], dtype=np.float32).reshape(-1, 2)
                    segment_xy = [
                        clip_image_point(px + x_offset, py + y_offset, image_width, image_height)
                        for px, py in points
                    ]
                append_detection(
                    detections,
                    names,
                    window,
                    image_width,
                    image_height,
                    xyxy,
                    score,
                    class_id,
                    segment_xy=segment_xy,
                    obb_xyxyxyxy=segment_xy,
                )
            continue

        boxes = result.boxes
        if boxes is None:
            continue

        xyxy_list = boxes.xyxy.detach().cpu().numpy()
        conf_list = boxes.conf.detach().cpu().numpy()
        cls_list = boxes.cls.detach().cpu().numpy().astype(int)
        masks = result.masks
        mask_segments = masks.xy if masks is not None else None

        for idx, (xyxy, score, class_id) in enumerate(zip(xyxy_list, conf_list, cls_list)):
            segment_xy = None
            if mask_segments is not None and idx < len(mask_segments):
                segment_xy = []
                for px, py in mask_segments[idx].tolist():
                    segment_xy.append(
                        clip_image_point(px + x_offset, py + y_offset, image_width, image_height)
                    )
            append_detection(
                detections,
                names,
                window,
                image_width,
                image_height,
                xyxy,
                score,
                class_id,
                segment_xy=segment_xy,
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


def yolo_predict_device(device):
    device = str(device)
    if device.startswith("npu:"):
        return device.split(":", 1)[1]
    if device == "npu":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    if device.startswith("gpu:"):
        return device.split(":", 1)[1]
    if device == "gpu":
        return "0"
    return device


def create_progress(total_tiles):
    if tqdm is not None:
        return tqdm(total=total_tiles, desc="Infer", unit="tile")
    return None


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

    with dbf_path.open("wb") as f:
        f.write(
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
            f.write(_dbf_field_descriptor(*field))
        f.write(b"\r")

        for det in detections:
            f.write(b" ")
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
                f.write(
                    _dbf_format_value(
                        values[name], field_type, length, decimal_count
                    )
                )
        f.write(b"\x1a")


def _build_polygon_record(ring_coords):
    xs = [p[0] for p in ring_coords]
    ys = [p[1] for p in ring_coords]
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
        shp_records.append(
            struct.pack(">2i", record_number, content_length_words) + content
        )
        shx_records.append(struct.pack(">2i", offset_words, content_length_words))
        offset_words += 4 + content_length_words
        xmin = min(xmin, bbox[0])
        ymin = min(ymin, bbox[1])
        xmax = max(xmax, bbox[2])
        ymax = max(ymax, bbox[3])

    if not shp_records:
        xmin = ymin = xmax = ymax = 0.0

    shp_file_length_words = 50 + sum(len(rec) for rec in shp_records) // 2
    shx_file_length_words = 50 + len(shx_records) * 4

    def file_header(file_length_words):
        return (
            struct.pack(">i", 9994)
            + b"\x00" * 20
            + struct.pack(">i", file_length_words)
            + struct.pack("<2i4d4d", 1000, shape_type, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0)
        )

    with shp_path.open("wb") as shp:
        shp.write(file_header(shp_file_length_words))
        for rec in shp_records:
            shp.write(rec)

    with shp_path.with_suffix(".shx").open("wb") as shx:
        shx.write(file_header(shx_file_length_words))
        for rec in shx_records:
            shx.write(rec)


def _write_prj(prj_path, crs_wkt):
    if not crs_wkt:
        return
    with prj_path.open("w", encoding="utf-8") as f:
        f.write(crs_wkt)


def write_shapefile(shp_path, detections, reader, geometry="bbox"):
    shp_path = Path(shp_path)
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    _delete_shapefile_sidecars(shp_path)

    if geometry == "bbox":
        geometry_builder = lambda det: bbox_to_ring_coords(reader, det["bbox_xyxy"])
        export_detections = detections
    elif geometry == "mask":
        geometry_builder = lambda det: mask_to_ring_coords(reader, det.get("segment_xy"))
        export_detections = [det for det in detections if det.get("segment_xy")]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry}")

    _write_shp_shx(shp_path, export_detections, geometry_builder)
    _write_dbf(shp_path.with_suffix(".dbf"), export_detections)
    _write_prj(shp_path.with_suffix(".prj"), reader.crs_wkt)


def main():
    args = parse_args()
    stride = args.stride or args.tile_size
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if args.tile_size <= 0:
        raise ValueError("tile_size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    ensure_device(args.device)

    source_path = Path(args.source)
    output_path, output_bbox_shp_path, output_mask_shp_path = resolve_output_paths(
        args.output, args.output_shp, args.output_bbox_shp, args.output_mask_shp
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reader = open_reader(str(source_path))
    print(
        f"[info] opened image with {reader.backend}: "
        f"{source_path} ({reader.width}x{reader.height})"
    )
    if reader.crs_wkt:
        print("[info] source image CRS detected and will be written to .prj")
    else:
        print(
            "[warn] no georeference was read from the source image. "
            "The output shapefile will use pixel coordinates."
        )

    model = YOLO(args.model)
    total_tiles = count_windows(reader.width, reader.height, args.tile_size, stride)
    print(f"[info] total tiles: {total_tiles}, tile_size={args.tile_size}, stride={stride}")

    all_detections = []
    processed_tiles = 0
    progress = create_progress(total_tiles)
    try:
        windows = generate_windows(reader.width, reader.height, args.tile_size, stride)
        for batch_images, batch_windows in tile_batches(reader, windows, args.batch_size):
            results = model.predict(
                source=batch_images,
                device=yolo_predict_device(args.device),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                classes=args.classes,
                agnostic_nms=args.agnostic_nms,
                verbose=False,
                save=False,
            )
            all_detections.extend(
                convert_results(results, batch_windows, reader.width, reader.height)
            )
            processed_tiles += len(batch_windows)
            if progress is not None:
                progress.update(len(batch_windows))
            else:
                print(f"[info] processed {processed_tiles}/{total_tiles} tiles", flush=True)
    finally:
        if progress is not None:
            progress.close()
        reader.close()

    if args.global_nms:
        merged_detections = global_nms(all_detections, args.global_iou)
    else:
        merged_detections = all_detections

    if output_bbox_shp_path is not None and output_mask_shp_path is None:
        if any(det.get("segment_xy") for det in merged_detections):
            output_mask_shp_path = output_bbox_shp_path.with_name(
                f"{output_bbox_shp_path.stem}_mask.shp"
            )

    output = {
        "image": str(source_path),
        "backend": reader.backend,
        "image_size": {"width": reader.width, "height": reader.height},
        "model": args.model,
        "device": args.device,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "tile_size": args.tile_size,
        "stride": stride,
        "tiles": total_tiles,
        "global_nms": bool(args.global_nms),
        "raw_detections": len(all_detections),
        "merged_detections": len(merged_detections),
        "detections": merged_detections,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[info] saved merged predictions to {output_path}")
    if output_bbox_shp_path is not None:
        write_shapefile(output_bbox_shp_path, merged_detections, reader, geometry="bbox")
        if reader.crs_wkt:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} with source CRS")
        else:
            print(f"[info] saved bbox shapefile to {output_bbox_shp_path} using pixel coordinates")
    if output_mask_shp_path is not None:
        write_shapefile(output_mask_shp_path, merged_detections, reader, geometry="mask")
        mask_count = sum(1 for det in merged_detections if det.get("segment_xy"))
        if reader.crs_wkt:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} with source CRS "
                f"({mask_count} polygons)"
            )
        else:
            print(
                f"[info] saved mask shapefile to {output_mask_shp_path} using pixel coordinates "
                f"({mask_count} polygons)"
            )
    if args.global_nms:
        print(
            f"[info] detections before global NMS: {len(all_detections)}, "
            f"after global NMS: {len(merged_detections)}"
        )
    else:
        print(
            f"[info] detections exported without extra global NMS: {len(merged_detections)}"
        )


if __name__ == "__main__":
    main()
