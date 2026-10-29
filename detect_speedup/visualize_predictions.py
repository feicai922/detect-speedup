# -*- coding: utf-8 -*-
from . import runtime_warning_filters  # noqa: F401
import argparse
import datetime as dt
import json
import struct
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
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
        # Expose the geotransform at the top level so vectorised helpers
        # (e.g. ``_reader_transform_6tuple``) can find it without poking
        # at private attributes of the underlying dataset.
        self.transform = ds.transform

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
        # Mirror the geotransform as ``self.transform`` for the vectorised helpers.
        self.transform = self.geotransform

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
        self.transform = None
        self.geotransform = None

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

# 统一使用绿色渲染所有置信度档位，不再按阈值区分颜色。
GREEN_COLOR = (0, 255, 0)
THRESHOLD_COLORS = {
    0.1: GREEN_COLOR,
    0.2: GREEN_COLOR,
    0.3: GREEN_COLOR,
    0.4: GREEN_COLOR,
    0.5: GREEN_COLOR,
    0.6: GREEN_COLOR,
    0.7: GREEN_COLOR,
}
MASK_COLORS = [
    (255, 64, 64),
    (64, 255, 128),
    (64, 160, 255),
    (255, 224, 64),
    (255, 96, 224),
    (96, 255, 255),
    (160, 96, 255),
    (255, 144, 64),
    (128, 255, 64),
    (64, 255, 224),
    (255, 64, 160),
    (224, 255, 96),
]
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
    # 统一使用绿色，不再按阈值区分颜色。
    return GREEN_COLOR


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


def compute_iou_vectorized(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between two ``(N, 4)`` xyxy boxes arrays.

    Returns ``(len(boxes_a), len(boxes_b))`` IoU matrix.
    """
    a = np.asarray(boxes_a, dtype=np.float64)
    b = np.asarray(boxes_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)

    a_x1, a_y1, a_x2, a_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    b_x1, b_y1, b_x2, b_y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    a_area = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)
    b_area = np.maximum(0.0, b_x2 - b_x1) * np.maximum(0.0, b_y2 - b_y1)

    inter_x1 = np.maximum(a_x1[:, None], b_x1[None, :])
    inter_y1 = np.maximum(a_y1[:, None], b_y1[None, :])
    inter_x2 = np.minimum(a_x2[:, None], b_x2[None, :])
    inter_y2 = np.minimum(a_y2[:, None], b_y2[None, :])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    union = np.maximum(a_area[:, None] + b_area[None, :] - inter, 1e-12)
    return inter / union


def nms_records(records, iou_threshold, agnostic=False):
    """Vectorised NMS over the merged-SHP records.

    Records already carry ``map_bbox_xyxy`` in source-CRS units, so the IoU
    computation is meaningful across tiles when ``nms_scope`` is ``__global__``.
    Falls back to a per-record Python loop only for groups of size 1.
    """
    if not records:
        return records

    scope_values = sorted({record.get("nms_scope", "__global__") for record in records})
    groups = []
    for scope in scope_values:
        scope_records = [r for r in records if r.get("nms_scope", "__global__") == scope]
        if agnostic:
            groups.append(scope_records)
        else:
            class_ids = sorted({r["class_id"] for r in scope_records})
            for cid in class_ids:
                groups.append([r for r in scope_records if r["class_id"] == cid])

    kept = []
    for group_records in groups:
        if len(group_records) == 1:
            kept.extend(group_records)
            continue
        # Build numpy views once per group
        boxes = np.asarray([r["map_bbox_xyxy"] for r in group_records], dtype=np.float64)
        scores = np.asarray([float(r["score"]) for r in group_records], dtype=np.float64)
        order = np.argsort(-scores, kind="stable")
        suppressed = np.zeros(len(group_records), dtype=bool)
        for pos, i_idx in enumerate(order):
            if suppressed[i_idx]:
                continue
            kept.append(group_records[i_idx])
            if pos + 1 >= len(order):
                break
            rest_idx = order[pos + 1:]
            rest_idx = rest_idx[~suppressed[rest_idx]]
            if rest_idx.size == 0:
                break
            ious = compute_iou_vectorized(boxes[i_idx:i_idx + 1], boxes[rest_idx])[0]
            suppressed[rest_idx] |= ious > iou_threshold

    kept.sort(key=lambda record: record["score"], reverse=True)
    return kept


def bbox_points(det):
    x1, y1, x2, y2 = det["bbox_xyxy"]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]


def segment_points(det):
    # 优先使用正则化后的轮廓（推理时写入 segment_xy_regularized）
    segment = det.get("segment_xy_regularized") or det.get("segment_xy")
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
        result = Image.new("RGB", (reader.width, reader.height), (0, 0, 0))
    finally:
        reader.close()

    draw = ImageDraw.Draw(result)
    for index, det in enumerate(detections):
        points = detection_points(det, geometry, mask_fallback_bbox=mask_fallback_bbox)
        if not points:
            continue
        if min_mask_area > 0 and geometry == "mask" and polygon_area(points) < min_mask_area:
            continue
        color = MASK_COLORS[index % len(MASK_COLORS)]
        flat = [coord for p in points for coord in p]
        draw.polygon(flat, fill=color)

    output_path = output_dir / "{}_{}_vis.jpg".format(json_path.stem, geometry)
    result.save(output_path, quality=max(1, min(int(jpg_quality), 100)))
    return output_path


def _reader_transform_6tuple(reader) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Best-effort extraction of a 6-tuple geotransform from any reader."""
    transform = getattr(reader, "transform", None)
    if transform is None:
        transform = getattr(reader, "geotransform", None)
    if transform is None:
        return None
    if hasattr(transform, "a"):
        return (
            float(transform.a), float(transform.b), float(transform.c),
            float(transform.d), float(transform.e), float(transform.f),
        )
    if isinstance(transform, (tuple, list)) and len(transform) >= 6:
        return tuple(float(v) for v in transform[:6])
    return None


# Recognised CRS WKTs we can return when the system has no GeoTIFF reader.
# Only includes the few lat/lon CRSs the user is likely to encounter; the
# most common by far is EPSG:4326 (WGS84).
#
# NOTE: The WKT must NOT include AXIS[…] definitions.  Shapefiles always
# store (x, y) = (longitude, latitude) regardless of the official CRS axis
# order, and QGIS / ArcGIS will swap coordinates when a .prj declares
# AXIS["Latitude",NORTH] first — breaking the overlay with the source TIFF.
_WGS84_WKT = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
)


def _parse_tile_bounds_from_filename(tile_path, image_size) -> Optional[Tuple[float, float, float, float, Tuple[float, float, float, float, float, float]]]:
    """Recover the tile's geotransform from the filename convention.

    Two known formats, both of which encode geo values with the dot replaced
    by underscore (e.g. ``-68.122149`` → ``-68_122149``):

    1. ``split_tif.py`` (the user's tool): ``{stem}_{row}_{col}_{xmin}_{ymin}_{xmax}_{ymax}.tif``
       — i.e. ``xmin, ymin, xmax, ymax`` in order.
    2. ``image_splitting.py``: ``{stem}_{row}_{col}_{xmin}_{ymax}_{xmax}_{ymin}.tif``
       — i.e. ``xmin, ymax, xmax, ymin`` in order.

    We try interpretation 1 first; if the resulting values are obviously
    nonsense (outside lat/lon range) we try interpretation 2. Returns
    ``(xmin, ymin, xmax, ymax, (a,b,c,d,e,f))``.
    """
    name = Path(tile_path).stem  # drop .tif
    # Each geo value is split into 2 tokens (sign+int, decimal). Re-join 8 tokens.
    parts = name.split("_")
    if len(parts) < 8:
        return None
    # Take the last 8 tokens and group into 4 pairs.
    last8 = parts[-8:]
    floats = []
    ok = True
    for i in range(0, 8, 2):
        sign, frac = last8[i], last8[i + 1]
        if not frac.isdigit():
            ok = False
            break
        if sign.startswith("-"):
            num_str = "-0." + frac if sign == "-0" else f"-{sign.lstrip('-')}.{frac}"
        else:
            num_str = f"{sign}.{frac}"
        try:
            floats.append(float(num_str))
        except ValueError:
            ok = False
            break
    if not ok or len(floats) != 4:
        return None

    width, height = image_size
    if not width or not height:
        return None

    xmin, a2, xmax, a4 = floats
    # Two possible interpretations:
    for xmin_v, ymin_v, xmax_v, ymax_v in (
        (xmin, a2, xmax, a4),         # split_tif.py: xmin, ymin, xmax, ymax
        (xmin, a4, xmax, a2),         # image_splitting.py: xmin, ymax, xmax, ymin
    ):
        if xmin_v == xmax_v or ymin_v == ymax_v:
            continue
        a = (xmax_v - xmin_v) / width
        e = -(ymax_v - ymin_v) / height
        b = 0.0
        d = 0.0
        c = xmin_v
        f = ymax_v
        # Sanity: looks like lat/lon?
        looks_like_latlon = (
            -180.0 <= xmin_v <= 180.0 and -180.0 <= xmax_v <= 180.0
            and -90.0 <= ymin_v <= 90.0 and -90.0 <= ymax_v <= 90.0
        )
        if looks_like_latlon:
            return (xmin_v, ymin_v, xmax_v, ymax_v, (a, b, c, d, e, f))
    return None


def _sanitize_wkt_for_shapefile(wkt: Optional[str]) -> Optional[str]:
    """Strip ``AXIS[…]`` definitions from a WKT string.

    Shapefiles always store (x, y) = (longitude/easting,
    latitude/northing).  When a .prj declares ``AXIS["Latitude",NORTH]``
    first, QGIS honours that and swaps the coordinates — breaking the
    overlay with the source raster.  Removing the axis declarations makes
    GIS software fall back to the traditional (x, y) order.
    """
    if not wkt:
        return wkt
    import re
    # Remove AXIS["name",DIRECTION] tokens, including any leading comma
    sanitized = re.sub(r',?\s*AXIS\["[^"]*",\s*[A-Z]+\]', '', wkt)
    return sanitized


def _get_source_crs_wkt(crs_source_path) -> Optional[str]:
    """Best-effort: read CRS WKT from ``crs_source_path``.

    Tries rasterio → GDAL → None.  The returned WKT has ``AXIS[…]``
    definitions stripped so the .prj sidecar uses the traditional
    shapefile (x, y) axis order.  WGS84 is the most common fallback
    when everything fails.
    """
    if not crs_source_path:
        return None
    try:
        import rasterio  # noqa: F401
        with rasterio.open(str(crs_source_path)) as ds:
            if ds.crs:
                return _sanitize_wkt_for_shapefile(ds.crs.to_wkt())
            return None
    except Exception:
        pass
    try:
        from osgeo import gdal
        ds = gdal.Open(str(crs_source_path), gdal.GA_ReadOnly)
        if ds is not None:
            try:
                wkt = ds.GetProjection() or None
            finally:
                ds = None
            return _sanitize_wkt_for_shapefile(wkt)
    except Exception:
        pass
    return None


def _inject_crs_from_source(reader, image_path, image_size, crs_source_path) -> bool:
    """If ``reader`` has no CRS, recover one from ``image_path``'s filename
    convention + the optional ``crs_source_path`` for the WKT.

    The PIL fallback returns pixel coordinates as-is, so any map coordinates
    in the SHP would be garbage. We rebuild the tile's affine from the geo
    bounds encoded in the tile filename and grab the CRS WKT from the source
    TIF (if a rasterio/GDAL backend is available). When neither is available
    but the bounds look like lat/lon, we still patch using a built-in
    WGS84 WKT so the SHP coordinates are correct.
    """
    if reader.crs_wkt:
        return False
    if image_size is None:
        return False
    # Accept both ``(w, h)`` tuples and ``{"width": w, "height": h}`` dicts.
    if isinstance(image_size, dict):
        wh = (image_size.get("width"), image_size.get("height"))
    else:
        wh = tuple(image_size)[:2] if image_size else (None, None)
    if not wh or wh[0] is None or wh[1] is None:
        return False
    parsed = _parse_tile_bounds_from_filename(image_path, wh)
    if not parsed:
        return False
    _xmin, _ymin, _xmax, _ymax, gt = parsed
    a, b, c, d, e, f = gt

    # Prefer the source TIF's WKT when we can read it; otherwise assume WGS84.
    crs_wkt = _get_source_crs_wkt(crs_source_path) or _WGS84_WKT

    def _map(x, y):
        return (c + a * x + b * y, f + d * x + e * y)

    reader.crs_wkt = crs_wkt
    reader.pixel_to_map = _map
    # Also expose the affine so the vectorised ``pixel_to_map_points`` path
    # picks it up (it reads ``reader.transform`` rather than calling
    # ``pixel_to_map`` once per point).
    reader.transform = gt
    reader.geotransform = gt
    return True


def pixel_to_map_points(reader, points):
    """Vectorised pixel → map for a list of ``(x, y)`` tuples.

    Reads the reader's ``transform`` / ``geotransform`` once, then runs a
    single numpy matmul — turning ~M Python calls into one.
    """
    if not points:
        return []
    arr = np.asarray(points, dtype=np.float64)
    gt = _reader_transform_6tuple(reader)
    if gt is None:
        # Identity (e.g. PIL fallback with no CRS) — return the same tuples.
        return [(float(x), float(y)) for x, y in arr]
    a, b, c, d, e, f = gt
    mx = c + a * arr[:, 0] + b * arr[:, 1]
    my = f + d * arr[:, 0] + e * arr[:, 1]
    return list(zip(mx.tolist(), my.tolist()))


def collect_records(json_path, image_path, thresholds, geometry, mask_fallback_bbox,
                    crs_source_path=None, reader=None):
    """Collect per-detection records from a prediction JSON.

    ``reader`` is an optional pre-built reader (any object exposing
    ``crs_wkt``, ``width/height`` and a ``pixel_to_map(x, y)`` callable — or
    a ``transform`` 6-tuple that ``pixel_to_map_points`` can pick up). When
    supplied, we skip ``_open_reader`` entirely so callers (e.g. the
    filename-derived ``_FastReader`` in ``run_shp_only``) can avoid
    rasterio/GDAL I/O on huge image tiles.
    """
    with json_path.open("r", encoding="utf-8") as file:
        prediction = json.load(file)
    detections = prediction.get("detections", [])
    image_size = prediction.get("image_size")  # {'width': int, 'height': int} or None
    if reader is None:
        reader = _open_reader(str(image_path))
        crs_patched = _inject_crs_from_source(reader, image_path, image_size, crs_source_path)
    else:
        # Caller already gave us a reader with the geo info baked in.
        crs_patched = False
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
        if crs_patched:
            print(f"[crs] patched CRS for {image_path.name}", flush=True)
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


def write_dbf(dbf_path, records, progress=True):
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

        iterator = (
            tqdm(records, total=len(records), desc=f"dbf {dbf_path.stem}",
                 unit="rec", dynamic_ncols=True)
            if (progress and tqdm is not None and len(records) > 0) else records
        )
        for record in iterator:
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


def build_polygon_record(points):
    """Build a single ESRI polygon (shape type 5) record from a closed ring.

    ``points`` must already be a closed ring (first point == last point).
    A polygon record holds ``(num_parts, num_points, parts_offsets..., xy...)``
    where ``parts_offsets[i]`` is the index of the first point of the i-th
    ring. For a single-ring polygon that's just ``[0]``.
    """
    arr = np.asarray(points, dtype=np.float64)
    xs = arr[:, 0]
    ys = arr[:, 1]
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    num_points = len(points)
    content = bytearray()
    content += struct.pack("<i", 5)                     # shape type = polygon
    content += struct.pack("<4d", *bbox)                # bbox
    content += struct.pack("<i", 1)                     # num_parts
    content += struct.pack("<i", num_points)
    content += struct.pack("<i", 0)                     # part 0 starts at offset 0
    content += arr.reshape(-1).tobytes()
    return bytes(content), bbox


def write_polygon_shapefile(shp_path, records, crs_wkt=None, progress=True):
    """Write ``records`` as a polygon SHP (shape type 5) with progress reporting.

    Records must carry a closed ``points`` ring (first point == last point).
    Each record is also written to the .dbf sidecar with class/score/bbox.
    """
    shp_path = Path(shp_path)
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    delete_shapefile_sidecars(shp_path)

    started = time.monotonic()
    shape_type = 5
    shp_records = []
    shx_records = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    offset_words = 50
    written = 0

    iterator = (
        tqdm(records, total=len(records), desc=f"shp {shp_path.stem}",
             unit="rec", dynamic_ncols=True, mininterval=0.5)
        if (progress and tqdm is not None and len(records) > 0) else records
    )

    for index, record in enumerate(iterator, start=1):
        points = record["points"]
        if len(points) < 3:
            continue
        content, bbox = build_polygon_record(points)
        content_length_words = len(content) // 2
        shp_records.append(struct.pack(">2i", index, content_length_words) + content)
        shx_records.append(struct.pack(">2i", offset_words, content_length_words))
        offset_words += 4 + content_length_words
        if bbox[0] < xmin: xmin = bbox[0]
        if bbox[1] < ymin: ymin = bbox[1]
        if bbox[2] > xmax: xmax = bbox[2]
        if bbox[3] > ymax: ymax = bbox[3]
        written += 1

    if not shp_records:
        xmin = ymin = xmax = ymax = 0.0

    shp_file_length_words = 50 + sum(len(record) for record in shp_records) // 2
    shx_file_length_words = 50 + len(shx_records) * 4

    file_header = (
        struct.pack(">i", 9994)
        + b"\x00" * 20
        + struct.pack(">i", shp_file_length_words)
        + struct.pack("<2i4d4d", 1000, shape_type, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0)
    )
    shx_header = (
        struct.pack(">i", 9994)
        + b"\x00" * 20
        + struct.pack(">i", shx_file_length_words)
        + struct.pack("<2i4d4d", 1000, shape_type, xmin, ymin, xmax, ymax, 0.0, 0.0, 0.0, 0.0)
    )

    with shp_path.open("wb") as shp_file:
        shp_file.write(file_header)
        for record in shp_records:
            shp_file.write(record)

    with shp_path.with_suffix(".shx").open("wb") as shx_file:
        shx_file.write(shx_header)
        for record in shx_records:
            shx_file.write(record)

    write_dbf(shp_path.with_suffix(".dbf"), records, progress=progress)
    shp_path.with_suffix(".cpg").write_text("UTF-8", encoding="utf-8")
    if crs_wkt:
        shp_path.with_suffix(".prj").write_text(crs_wkt, encoding="utf-8")

    elapsed = time.monotonic() - started
    if records:
        print(
            f"[shp] {shp_path.name}: {written} polygon(s) in {elapsed:.1f}s "
            f"({written / max(elapsed, 1e-6):.0f} rec/s)",
            flush=True,
        )


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
    # New: SHP output dir (defaults to sibling 'shps/' of output_dir) and CRS source.
    shp_output_dir = Path(config["shp_output_dir"]) if config.get("shp_output_dir") else output_dir.parent / "shps"
    shp_output_dir.mkdir(parents=True, exist_ok=True)
    crs_source_path = config.get("crs_source")
    if crs_source_path:
        print(f"[crs] fallback source TIF: {crs_source_path}", flush=True)
    if not config.get("shp_output_dir"):
        print(f"[shp] writing polygons to {shp_output_dir}", flush=True)

    json_files = list(iter_json_files(input_dir, recursive=recursive))
    if not json_files:
        raise RuntimeError("No JSON files found in {}".format(input_dir))

    print(f"[info] {len(json_files)} JSON file(s) to process", flush=True)

    processed = 0
    skipped = 0
    iterator = tqdm(json_files, desc="Visualize", unit="json", dynamic_ncols=True, mininterval=0.5) if tqdm else json_files
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
            crs_source_path=crs_source_path,
        )
        merge_records(total_records, new_records)
        if crs_wkt is None and image_crs_wkt:
            crs_wkt = image_crs_wkt
        processed += 1

    if merged_shp_nms:
        nms_iter = tqdm(thresholds, desc="NMS", unit="thr", dynamic_ncols=True) if tqdm else thresholds
        for threshold in nms_iter:
            before_count = len(total_records[threshold])
            t0 = time.monotonic()
            total_records[threshold] = nms_records(
                total_records[threshold],
                iou_threshold=merged_shp_iou,
                agnostic=merged_shp_agnostic_nms,
            )
            after_count = len(total_records[threshold])
            dt_nms = time.monotonic() - t0
            if tqdm:
                nms_iter.set_postfix(removed=before_count - after_count, sec=f"{dt_nms:.1f}")
            print(
                f"[info] NMS threshold {threshold}: {before_count} -> {after_count} "
                f"({dt_nms:.1f}s)",
                flush=True,
            )

    shp_iterator = tqdm(thresholds, desc="Write SHP", unit="thr", dynamic_ncols=True) if tqdm else thresholds
    for threshold in shp_iterator:
        shp_path = shp_output_dir / "all_{}_conf_{}.shp".format(geometry, threshold_key(threshold))
        write_polygon_shapefile(shp_path, total_records[threshold], crs_wkt=crs_wkt)

    write_legend(output_dir, thresholds, colors)
    print(
        "[summary] processed={}, skipped={}, merged_shp_nms={}, vis_dir={}, shp_dir={}".format(
            processed, skipped, bool(merged_shp_nms), output_dir, shp_output_dir
        ),
        flush=True,
    )


def main():
    args = parse_args()
    run_visualization(vars(args))


if __name__ == "__main__":
    main()
