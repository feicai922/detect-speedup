# -*- coding: utf-8 -*-
from . import runtime_warning_filters  # noqa: F401
import argparse
import datetime as dt
import json
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---- Lightweight image reader (avoid heavy torch/ultralytics imports) ----
def _open_reader(path):
    """Open an image for visualization — tries rasterio, gdal, then PIL."""
    path = str(path)

    # Try rasterio
    try:
        import rasterio
        from rasterio.windows import Window

        ds = rasterio.open(path)
        return _RasterioVisReader(ds)
    except Exception:
        pass

    # Try GDAL
    try:
        from osgeo import gdal

        ds = gdal.Open(path, gdal.GA_ReadOnly)
        if ds is not None:
            return _GDALVisReader(ds)
    except Exception:
        pass

    # Fallback to PIL
    from PIL import Image as PILImage
    PILImage.MAX_IMAGE_PIXELS = None
    img = PILImage.open(path)
    return _PILVisReader(img)


class _RasterioVisReader:
    def __init__(self, ds):
        self.ds = ds
        self.width = ds.width
        self.height = ds.height
        self.crs_wkt = ds.crs.to_wkt() if ds.crs else None

    def read(self, x1, y1, x2, y2):
        import numpy as np
        from rasterio.windows import Window
        patch = self.ds.read(window=Window(x1, y1, x2 - x1, y2 - y1))
        patch = np.transpose(patch, (1, 2, 0))
        patch = _to_rgb(patch)
        return patch

    def pixel_to_map(self, x, y):
        mx, my = self.ds.transform * (x, y)
        return float(mx), float(my)

    def close(self):
        self.ds.close()


class _GDALVisReader:
    def __init__(self, ds):
        self.ds = ds
        self.width = ds.RasterXSize
        self.height = ds.RasterYSize
        self.geotransform = ds.GetGeoTransform(can_return_null=True)
        self.crs_wkt = ds.GetProjection() or None

    def read(self, x1, y1, x2, y2):
        import numpy as np
        patch = self.ds.ReadAsArray(x1, y1, x2 - x1, y2 - y1)
        if patch.ndim == 2:
            patch = patch[:, :, None]
        else:
            patch = np.transpose(patch, (1, 2, 0))
        return _to_rgb(patch)

    def pixel_to_map(self, x, y):
        if self.geotransform is None:
            return float(x), float(y)
        gt = self.geotransform
        return float(gt[0] + x * gt[1] + y * gt[2]), float(gt[3] + x * gt[4] + y * gt[5])

    def close(self):
        self.ds = None


class _PILVisReader:
    def __init__(self, img):
        self.img = img
        self.width, self.height = img.size
        self.crs_wkt = None

    def read(self, x1, y1, x2, y2):
        import numpy as np
        patch = self.img.crop((x1, y1, x2, y2)).convert("RGB")
        return np.asarray(patch)

    def pixel_to_map(self, x, y):
        return float(x), float(y)

    def close(self):
        self.img.close()


def _to_rgb(patch):
    import numpy as np
    patch = np.asarray(patch)
    if patch.ndim == 2:
        patch = np.stack([patch] * 3, axis=-1)
    elif patch.shape[2] >= 3:
        patch = patch[:, :, :3]
    if patch.dtype == np.uint8:
        return patch
    patch = np.nan_to_num(patch, nan=0.0)
    mn, mx = float(patch.min()), float(patch.max())
    if mx <= 255 and mn >= 0:
        return np.clip(patch, 0, 255).astype(np.uint8)
    if mx - mn < 1e-6:
        return np.zeros_like(patch, dtype=np.uint8)
    return ((patch - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)

# ---- end reader ----

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

THRESHOLD_COLORS = {
    0.1: (255, 0, 255),    # Magenta
    0.2: (128, 0, 255),    # Violet
    0.3: (0, 102, 255),    # Blue
    0.4: (0, 255, 255),    # Cyan
    0.5: (0, 255, 0),      # Green
    0.6: (255, 200, 0),    # Yellow-orange
    0.7: (255, 0, 0),      # Red
}
SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize prediction JSON files and export confidence-threshold shapefiles."
    )
    parser.add_argument("input_dir", type=str, help="Directory containing prediction JSON files.")
    parser.add_argument("output_dir", type=str, help="Directory to save visualization JPGs and shapefiles.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Optional image directory used when JSON image paths are missing or moved.",
    )
    parser.add_argument(
        "--geometry",
        choices=("bbox", "mask", "obb"),
        default="bbox",
        help="Draw/export bbox, mask, or YOLO OBB outlines.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Confidence thresholds. Drawn from low to high so high confidence covers low confidence.",
    )
    parser.add_argument("--line-width", type=int, default=3, help="Visualization line width in pixels.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input_dir for JSON files.",
    )
    parser.add_argument(
        "--mask-fallback-bbox",
        action="store_true",
        help="When --geometry mask/obb is selected, draw/export bbox for detections without segment_xy.",
    )
    parser.add_argument(
        "--jpg-quality",
        type=int,
        default=92,
        help="JPEG quality for visualization outputs.",
    )
    parser.add_argument(
        "--no-merged-shp-nms",
        action="store_true",
        help="Disable global NMS before writing merged confidence-threshold shapefiles.",
    )
    parser.add_argument(
        "--merged-shp-iou",
        type=float,
        default=0.45,
        help="IoU threshold for global NMS before writing merged shapefiles.",
    )
    parser.add_argument(
        "--merged-shp-class-aware-nms",
        action="store_true",
        help="Only suppress records with the same class_id when applying merged shapefile NMS.",
    )
    return parser.parse_args()


def threshold_key(threshold):
    return str(threshold).replace(".", "_")


def color_for_threshold(threshold):
    if threshold in THRESHOLD_COLORS:
        return THRESHOLD_COLORS[threshold]
    palette = [
        (0, 102, 255),
        (0, 200, 255),
        (0, 220, 0),
        (255, 170, 0),
        (255, 0, 0),
        (170, 0, 255),
        (255, 0, 180),
    ]
    return palette[abs(hash(threshold)) % len(palette)]


def iter_json_files(input_dir, recursive=False):
    pattern = "**/*.json" if recursive else "*.json"
    yield from sorted(Path(input_dir).glob(pattern))


def build_image_index(image_dir):
    if not image_dir:
        return {}
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return {}
    return {
        path.stem: path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }


def resolve_image_path(prediction, json_path, image_index):
    image_value = prediction.get("image")
    if image_value:
        image_path = Path(image_value)
        if image_path.exists():
            return image_path
        if image_path.stem in image_index:
            return image_index[image_path.stem]

    if json_path.stem in image_index:
        return image_index[json_path.stem]

    for suffix in SUPPORTED_IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def normalize_thresholds(thresholds):
    return sorted({round(float(threshold), 6) for threshold in thresholds})


def det_score(det):
    return float(det.get("score", 0.0))


def normalized_box(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def compute_iou(box, boxes):
    box = normalized_box(box)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    ious = []
    for other in boxes:
        other = normalized_box(other)
        inter_w = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
        inter_h = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
        inter = inter_w * inter_h
        other_area = max(0.0, other[2] - other[0]) * max(0.0, other[3] - other[1])
        union = max(box_area + other_area - inter, 1e-12)
        ious.append(inter / union)
    return ious


def nms_records(records, iou_threshold, agnostic=False):
    if not records:
        return records

    scope_values = sorted({record.get("nms_scope", "__global__") for record in records})
    groups = []
    for scope in scope_values:
        scope_records = [record for record in records if record.get("nms_scope", "__global__") == scope]
        if agnostic:
            groups.append(scope_records)
        else:
            class_ids = sorted({record["class_id"] for record in scope_records})
            groups.extend(
                [record for record in scope_records if record["class_id"] == class_id]
                for class_id in class_ids
            )

    kept = []
    for group_records in groups:
        order = sorted(range(len(group_records)), key=lambda index: group_records[index]["score"], reverse=True)
        while order:
            index = order[0]
            kept.append(group_records[index])
            if len(order) == 1:
                break

            current_box = group_records[index]["map_bbox_xyxy"]
            remaining = order[1:]
            remaining_boxes = [group_records[next_index]["map_bbox_xyxy"] for next_index in remaining]
            ious = compute_iou(current_box, remaining_boxes)
            order = [next_index for next_index, iou in zip(remaining, ious) if iou <= iou_threshold]

    kept.sort(key=lambda record: record["score"], reverse=True)
    return kept


def bbox_points(det):
    x1, y1, x2, y2 = det["bbox_xyxy"]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def segment_points(det):
    segment = det.get("segment_xy")
    if not segment or len(segment) < 3:
        return None
    points = [(float(x), float(y)) for x, y in segment]
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def obb_points(det):
    for key in ("obb_xyxyxyxy", "obb_points", "xyxyxyxy"):
        obb = det.get(key)
        if obb:
            break
    else:
        obb = None

    if obb is None:
        segment = det.get("segment_xy")
        if segment and (det.get("is_obb") or len(segment) == 4):
            return segment_points(det)
        return None

    if len(obb) == 8 and not isinstance(obb[0], (list, tuple)):
        iterator = zip(obb[0::2], obb[1::2])
    else:
        iterator = obb

    points = [(float(x), float(y)) for x, y in iterator]
    if len(points) < 4:
        return None
    points = points[:4]
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def detection_points(det, geometry, mask_fallback_bbox=False):
    if geometry == "bbox":
        return bbox_points(det)

    if geometry == "obb":
        points = obb_points(det)
        if points is None and mask_fallback_bbox:
            return bbox_points(det)
        return points

    points = segment_points(det)
    if points is None and mask_fallback_bbox:
        return bbox_points(det)
    return points


def draw_legend(draw, thresholds, colors, alpha=False):
    """Draw confidence color legend. If alpha=True, draws with semi-transparent background."""
    font = ImageFont.load_default()
    rows = []
    for threshold in sorted(thresholds, reverse=True):
        rows.append((threshold, colors[threshold], "score >= {:.1f}".format(threshold)))

    padding = 8
    swatch = 18
    row_height = 22
    width = 170
    height = padding * 2 + row_height * len(rows) + 18
    x0, y0 = 10, 10

    bg = (255, 255, 255, 160) if alpha else (255, 255, 255)
    outline_color = (0, 0, 0, 120) if alpha else (0, 0, 0)
    text_color = (0, 0, 0, 180) if alpha else (0, 0, 0)
    title_color = (0, 0, 0, 200) if alpha else (0, 0, 0)

    draw.rectangle([x0, y0, x0 + width, y0 + height], fill=bg, outline=outline_color, width=1)
    draw.text((x0 + padding, y0 + padding), "Confidence color", fill=title_color, font=font)
    y = y0 + padding + 18
    for _, color, label in rows:
        if alpha:
            color = (*color, 220)
        draw.line([(x0 + padding, y + 10), (x0 + padding + swatch, y + 10)], fill=color, width=4)
        draw.text((x0 + padding + swatch + 8, y + 3), label, fill=text_color, font=font)
        y += row_height


def polygon_area(points):
    """Shoelace formula for polygon area."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def visualize_one(json_path, image_path, output_dir, thresholds, colors, geometry, line_width,
                  mask_fallback_bbox, jpg_quality, fill_alpha=None, min_mask_area=25,
                  show_legend=False):
    """Visualize predictions — side-by-side: original | result."""
    if fill_alpha is None:
        fill_alpha = 80 if geometry == "mask" else 0

    with json_path.open("r", encoding="utf-8") as file:
        prediction = json.load(file)

    detections = prediction.get("detections", [])
    reader = _open_reader(str(image_path))
    try:
        raw = reader.read(0, 0, reader.width, reader.height)
    finally:
        reader.close()

    # Left: original image
    original = Image.fromarray(raw).convert("RGB")

    # Right: overlay result
    base = Image.fromarray(raw).convert("RGBA")

    if fill_alpha > 0:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        for threshold in thresholds:
            color = colors[threshold]
            fill_rgba = (*color, fill_alpha)
            stroke_rgba = (255, 255, 255, 200)
            for det in detections:
                if det_score(det) < threshold:
                    continue
                points = detection_points(det, geometry, mask_fallback_bbox=mask_fallback_bbox)
                if not points:
                    continue
                if min_mask_area > 0 and geometry == "mask" and polygon_area(points) < min_mask_area:
                    continue
                flat = [coord for p in points for coord in p]
                overlay_draw.polygon(flat, fill=fill_rgba, outline=stroke_rgba)

        if show_legend:
            draw_legend(overlay_draw, thresholds, colors, alpha=True)
        result = Image.alpha_composite(base, overlay).convert("RGB")
    else:
        result = base.convert("RGB")
        draw = ImageDraw.Draw(result)
        for threshold in thresholds:
            color = colors[threshold]
            for det in detections:
                if det_score(det) < threshold:
                    continue
                points = detection_points(det, geometry, mask_fallback_bbox=mask_fallback_bbox)
                if not points:
                    continue
                if min_mask_area > 0 and geometry == "mask" and polygon_area(points) < min_mask_area:
                    continue
                draw.line(points, fill=color, width=line_width, joint="curve")
        if show_legend:
            draw_legend(draw, thresholds, colors, alpha=False)

    # Side-by-side: original | result
    w, h = original.size
    combined = Image.new("RGB", (w * 2, h))
    combined.paste(original, (0, 0))
    combined.paste(result, (w, 0))

    output_path = output_dir / "{}_{}_vis.jpg".format(json_path.stem, geometry)
    combined.save(output_path, quality=max(1, min(int(jpg_quality), 100)))
    return output_path


def pixel_to_map_points(reader, points):
    return [reader.pixel_to_map(x, y) for x, y in points]


def collect_records(json_path, image_path, thresholds, geometry, mask_fallback_bbox):
    with json_path.open("r", encoding="utf-8") as file:
        prediction = json.load(file)
    detections = prediction.get("detections", [])
    reader = _open_reader(str(image_path))
    records = {threshold: [] for threshold in thresholds}
    try:
        nms_scope = "__global__" if reader.crs_wkt else image_path.name
        for det in detections:
            points = detection_points(det, geometry, mask_fallback_bbox=mask_fallback_bbox)
            if not points:
                continue
            map_points = pixel_to_map_points(reader, points)
            bbox_xyxy = det.get("bbox_xyxy", [0, 0, 0, 0])
            map_bbox_points = pixel_to_map_points(reader, bbox_points({"bbox_xyxy": bbox_xyxy})[:-1])
            map_xs = [point[0] for point in map_bbox_points]
            map_ys = [point[1] for point in map_bbox_points]
            map_bbox_xyxy = [min(map_xs), min(map_ys), max(map_xs), max(map_ys)]
            for threshold in thresholds:
                if det_score(det) >= threshold:
                    records[threshold].append(
                        {
                            "image": image_path.name,
                            "class_id": int(det.get("class_id", -1)),
                            "class_name": str(det.get("class_name", "")),
                            "score": det_score(det),
                            "threshold": threshold,
                            "bbox_xyxy": bbox_xyxy,
                            "map_bbox_xyxy": map_bbox_xyxy,
                            "nms_scope": nms_scope,
                            "points": map_points,
                        }
                    )
        return records, reader.crs_wkt
    finally:
        reader.close()


def delete_shapefile_sidecars(shp_path):
    for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        sidecar = shp_path.with_suffix(suffix)
        if sidecar.exists():
            sidecar.unlink()


def dbf_field_descriptor(name, field_type, length, decimal_count=0):
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


def dbf_format_value(value, field_type, length, decimal_count=0):
    if field_type == "C":
        return str(value)[:length].ljust(length).encode("utf-8", errors="ignore")[:length]
    if field_type == "N":
        if decimal_count > 0:
            text = "{0:>{1}.{2}f}".format(float(value), length, decimal_count)
        else:
            text = "{0:>{1}d}".format(int(value), length)
        return text.encode("ascii")[:length].rjust(length, b" ")
    raise ValueError("Unsupported DBF field type: {}".format(field_type))


def write_dbf(dbf_path, records):
    fields = [
        ("image", "C", 80, 0),
        ("class_id", "N", 10, 0),
        ("class_name", "C", 64, 0),
        ("score", "N", 18, 8),
        ("thr", "N", 10, 3),
        ("x1", "N", 18, 6),
        ("y1", "N", 18, 6),
        ("x2", "N", 18, 6),
        ("y2", "N", 18, 6),
    ]
    header_length = 32 + 32 * len(fields) + 1
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
                len(records),
                header_length,
                record_length,
            )
        )
        for field in fields:
            file.write(dbf_field_descriptor(*field))
        file.write(b"\r")

        for record in records:
            x1, y1, x2, y2 = record["bbox_xyxy"]
            values = {
                "image": record["image"],
                "class_id": record["class_id"],
                "class_name": record["class_name"],
                "score": record["score"],
                "thr": record["threshold"],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            file.write(b" ")
            for name, field_type, length, decimal_count in fields:
                file.write(dbf_format_value(values[name], field_type, length, decimal_count))
        file.write(b"\x1a")


def build_polyline_record(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    content = bytearray()
    content += struct.pack("<i", 3)
    content += struct.pack("<4d", *bbox)
    content += struct.pack("<2i", 1, len(points))
    content += struct.pack("<i", 0)
    for x, y in points:
        content += struct.pack("<2d", float(x), float(y))
    return bytes(content), bbox


def write_polyline_shapefile(shp_path, records, crs_wkt=None):
    shp_path = Path(shp_path)
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    delete_shapefile_sidecars(shp_path)

    shape_type = 3
    shp_records = []
    shx_records = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    offset_words = 50

    for index, record in enumerate(records, start=1):
        points = record["points"]
        if len(points) < 2:
            continue
        content, bbox = build_polyline_record(points)
        content_length_words = len(content) // 2
        shp_records.append(struct.pack(">2i", index, content_length_words) + content)
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

    write_dbf(shp_path.with_suffix(".dbf"), records)
    shp_path.with_suffix(".cpg").write_text("UTF-8", encoding="utf-8")
    if crs_wkt:
        shp_path.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")


def write_legend(output_dir, thresholds, colors):
    lines = [
        "Confidence color legend:",
        "Low thresholds are drawn first; higher thresholds are drawn later and cover lower thresholds.",
    ]
    for threshold in sorted(thresholds):
        color = colors[threshold]
        lines.append("score >= {:.1f}: RGB{}".format(threshold, color))
    (output_dir / "confidence_color_legend.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_records(total_records, new_records):
    for threshold, records in new_records.items():
        total_records.setdefault(threshold, []).extend(records)


def run_visualization(config):
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = normalize_thresholds(config.get("thresholds", DEFAULT_THRESHOLDS))
    colors = {threshold: color_for_threshold(threshold) for threshold in thresholds}
    image_index = build_image_index(config.get("image_dir"))
    total_records = {threshold: [] for threshold in thresholds}
    crs_wkt = None
    geometry = config.get("geometry", "bbox")
    if geometry not in {"bbox", "mask", "obb"}:
        raise ValueError("Unsupported geometry: {}".format(geometry))
    line_width = config.get("line_width", 3)
    mask_fallback_bbox = config.get("mask_fallback_bbox", False)
    jpg_quality = config.get("jpg_quality", 92)
    fill_alpha = config.get("fill_alpha", None)
    min_mask_area = config.get("min_mask_area", 25)
    show_legend = config.get("show_legend", False)
    recursive = config.get("recursive", False)
    merged_shp_nms = config.get("merged_shp_nms", not config.get("no_merged_shp_nms", False))
    merged_shp_iou = float(config.get("merged_shp_iou", 0.45))
    merged_shp_agnostic_nms = not bool(config.get("merged_shp_class_aware_nms", False))

    json_files = list(iter_json_files(input_dir, recursive=recursive))
    if not json_files:
        raise RuntimeError("No JSON files found in {}".format(input_dir))

    processed = 0
    skipped = 0
    iterator = tqdm(json_files, desc="Visualize", unit="json", dynamic_ncols=True) if tqdm else json_files
    for json_path in iterator:
        with json_path.open("r", encoding="utf-8") as file:
            prediction = json.load(file)
        image_path = resolve_image_path(prediction, json_path, image_index)
        if image_path is None:
            skipped += 1
            if tqdm is None:
                print("[warn] skip {}, source image not found".format(json_path))
            continue

        visualize_one(
            json_path=json_path,
            image_path=image_path,
            output_dir=output_dir,
            thresholds=thresholds,
            colors=colors,
            geometry=geometry,
            line_width=line_width,
            mask_fallback_bbox=mask_fallback_bbox,
            jpg_quality=jpg_quality,
            fill_alpha=fill_alpha,
            min_mask_area=min_mask_area,
            show_legend=show_legend,
        )
        new_records, image_crs_wkt = collect_records(
            json_path=json_path,
            image_path=image_path,
            thresholds=thresholds,
            geometry=geometry,
            mask_fallback_bbox=mask_fallback_bbox,
        )
        merge_records(total_records, new_records)
        if crs_wkt is None and image_crs_wkt:
            crs_wkt = image_crs_wkt
        processed += 1

    if merged_shp_nms:
        for threshold in thresholds:
            before_count = len(total_records[threshold])
            total_records[threshold] = nms_records(
                total_records[threshold],
                iou_threshold=merged_shp_iou,
                agnostic=merged_shp_agnostic_nms,
            )
            after_count = len(total_records[threshold])
            if before_count != after_count:
                print(
                    "[info] merged SHP NMS threshold {}: {} -> {}".format(
                        threshold, before_count, after_count
                    )
                )

    shp_iterator = tqdm(thresholds, desc="Write SHP", unit="thr", dynamic_ncols=True) if tqdm else thresholds
    for threshold in shp_iterator:
        shp_path = output_dir / "all_{}_conf_{}.shp".format(geometry, threshold_key(threshold))
        write_polyline_shapefile(shp_path, total_records[threshold], crs_wkt=crs_wkt)

    write_legend(output_dir, thresholds, colors)
    print(
        "[summary] processed={}, skipped={}, merged_shp_nms={}, output_dir={}".format(
            processed, skipped, bool(merged_shp_nms), output_dir
        )
    )


def main():
    args = parse_args()
    run_visualization(vars(args))


if __name__ == "__main__":
    main()
