# -*- coding: utf-8 -*-
"""Shapefile export with progress reporting and vectorised CRS conversion.

Used by YOLO / RTMDet / RF-DETR inference. The previous hand-rolled writers
called ``reader.pixel_to_map`` once per vertex — for thousands of instance
masks that adds up to millions of Python calls and a long silent stretch
where users couldn't tell if the run was hung.

This module:

* vectorises bbox + mask vertex conversion through numpy
* wraps the per-detection loop in a tqdm progress bar (when tqdm is
  available) so the user can see N/total advancing
* prints a one-line timing summary after each file is written
* remains API-compatible with the existing ``write_shapefile(...)`` signature
"""

from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


SHAPE_TYPE_POLYGON = 5
BBOX_FIELDS = (
    ("class_id", "N", 10, 0),
    ("class_name", "C", 64, 0),
    ("score", "N", 18, 8),
    ("x1", "N", 18, 6),
    ("y1", "N", 18, 6),
    ("x2", "N", 18, 6),
    ("y2", "N", 18, 6),
)


# ---------------------------------------------------------------------------
# Vectorised pixel -> map conversion
# ---------------------------------------------------------------------------

def _resolve_geotransform(reader) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Pull the 6-tuple geotransform off a reader (rasterio / GDAL / None)."""
    transform = getattr(reader, "transform", None)
    if transform is None:
        return None
    if hasattr(transform, "a"):  # rasterio Affine
        return (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        )
    if isinstance(transform, (tuple, list)) and len(transform) >= 6:
        return tuple(float(v) for v in transform[:6])
    return None


def _pixel_array_to_map(
    transform: Optional[Tuple[float, float, float, float, float, float]],
    xy: np.ndarray,
) -> np.ndarray:
    """Convert an ``(N, 2)`` float64 array of pixel coords to map coords."""
    if transform is None:
        return np.asarray(xy, dtype=np.float64)
    a, b, c, d, e, f = transform
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    mx = c + a * arr[:, 0] + b * arr[:, 1]
    my = f + d * arr[:, 0] + e * arr[:, 1]
    return np.stack([mx, my], axis=1)


def _to_pixel_array(corners: Sequence[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(corners, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 2)
    return arr


# ---------------------------------------------------------------------------
# Geometry builders (vectorised)
# ---------------------------------------------------------------------------

def bbox_to_ring_coords(reader, bbox_xyxy) -> List[Tuple[float, float]]:
    """Return the 5-corner closed ring of a bbox in map coordinates."""
    transform = _resolve_geotransform(reader)
    x1, y1, x2, y2 = bbox_xyxy
    pixels = _to_pixel_array(
        [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    )
    mapped = _pixel_array_to_map(transform, pixels)
    return [tuple(p) for p in mapped]


def mask_to_ring_coords(reader, segment_xy) -> Optional[List[Tuple[float, float]]]:
    """Return the closed ring of an instance mask in map coordinates."""
    if not segment_xy or len(segment_xy) < 3:
        return None
    transform = _resolve_geotransform(reader)
    pixels = _to_pixel_array(segment_xy)
    mapped = _pixel_array_to_map(transform, pixels).tolist()
    if not mapped:
        return None
    if mapped[0] != mapped[-1]:
        mapped.append(mapped[0])
    return [tuple(p) for p in mapped]


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def _build_polygon_record(ring_coords: Sequence[Tuple[float, float]]) -> Tuple[bytes, Tuple[float, float, float, float]]:
    xs = [p[0] for p in ring_coords]
    ys = [p[1] for p in ring_coords]
    bbox = (min(xs), min(ys), max(xs), max(ys))
    num_points = len(ring_coords)
    content = bytearray()
    content += struct.pack("<i", SHAPE_TYPE_POLYGON)
    content += struct.pack("<4d", *bbox)
    content += struct.pack("<2i", 1, num_points)
    content += struct.pack("<i", 0)
    # Pack all vertices in one shot rather than per-vertex struct.pack.
    flat = np.asarray(ring_coords, dtype=np.float64).reshape(-1)
    content += flat.tobytes()
    return bytes(content), bbox


def _dbf_field_descriptor(name: str, field_type: str, length: int, decimal_count: int = 0) -> bytes:
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


def _dbf_format_value(value, field_type: str, length: int, decimal_count: int) -> bytes:
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


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _file_header(file_length_words: int, bbox, shape_type: int = SHAPE_TYPE_POLYGON) -> bytes:
    if bbox is None:
        bbox = (0.0, 0.0, 0.0, 0.0)
    return (
        struct.pack(">i", 9994)
        + b"\x00" * 20
        + struct.pack(">i", file_length_words)
        + struct.pack("<2i4d4d", 1000, shape_type, *bbox, 0.0, 0.0, 0.0, 0.0)
    )


def _delete_shapefile_sidecars(shp_path: Path) -> None:
    for suffix in (".shp", ".shx", ".dbf", ".prj"):
        path = shp_path.with_suffix(suffix)
        if path.exists():
            path.unlink()


def _prj_writer(prj_path: Path, crs_wkt: Optional[str]) -> None:
    """Write the .prj sidecar with AXIS[…] definitions stripped.

    Shapefiles always use (x, y) = (longitude/easting, latitude/northing)
    order.  Including ``AXIS["Latitude",NORTH]`` (as rasterio's OGC WKT
    does for EPSG:4326) causes QGIS to swap the coordinates, breaking the
    overlay with the source raster.
    """
    if not crs_wkt:
        return
    import re
    sanitized = re.sub(r',?\s*AXIS\["[^"]*",\s*[A-Z]+\]', '', crs_wkt)
    prj_path.write_text(sanitized, encoding="utf-8")


def _iter_with_progress(iterable: Iterable, total: int, desc: str, enabled: bool):
    if enabled and tqdm is not None and total:
        return tqdm(iterable, total=total, desc=desc, unit="det", dynamic_ncols=True)
    return iterable


def _write_shp_shx(shp_path: Path, detections, geometry_builder, progress: bool):
    """Write .shp and .shx together with a progress bar over the detections."""
    shp_records: List[bytes] = []
    shx_records: List[bytes] = []
    xmin = ymin = float("inf")
    xmax = ymax = float("-inf")
    offset_words = 50
    record_number = 0

    total = len(detections)
    iterator = _iter_with_progress(
        detections, total=total, desc=f"shp {shp_path.stem}", enabled=progress
    )

    for det in iterator:
        ring_coords = geometry_builder(det)
        if ring_coords is None:
            continue
        record_number += 1
        content, bbox = _build_polygon_record(ring_coords)
        content_length_words = len(content) // 2
        shp_records.append(struct.pack(">2i", record_number, content_length_words) + content)
        shx_records.append(struct.pack(">2i", offset_words, content_length_words))
        offset_words += 4 + content_length_words
        if bbox[0] < xmin: xmin = bbox[0]
        if bbox[1] < ymin: ymin = bbox[1]
        if bbox[2] > xmax: xmax = bbox[2]
        if bbox[3] > ymax: ymax = bbox[3]

    file_bbox = None if record_number == 0 else (xmin, ymin, xmax, ymax)
    shp_file_length_words = 50 + sum(len(r) for r in shp_records) // 2
    shx_file_length_words = 50 + len(shx_records) * 4

    with shp_path.open("wb") as shp_file:
        shp_file.write(_file_header(shp_file_length_words, file_bbox))
        for record in shp_records:
            shp_file.write(record)
    with shp_path.with_suffix(".shx").open("wb") as shx_file:
        shx_file.write(_file_header(shx_file_length_words, file_bbox))
        for record in shx_records:
            shx_file.write(record)
    return record_number


def _write_dbf(dbf_path: Path, detections, progress: bool):
    """Write the .dbf sidecar with progress reporting."""
    num_records = len(detections)
    num_fields = len(BBOX_FIELDS)
    header_length = 32 + 32 * num_fields + 1
    record_length = 1 + sum(field[2] for field in BBOX_FIELDS)

    with dbf_path.open("wb") as file:
        file.write(b"\x03" + b"\x00" * 3)  # version / date placeholders
        file.write(struct.pack("<LHH", num_records, header_length, record_length) + b"\x00" * 20)
        for name, field_type, length, decimal_count in BBOX_FIELDS:
            file.write(_dbf_field_descriptor(name, field_type, length, decimal_count))
        file.write(b"\r")

        iterator = _iter_with_progress(
            detections, total=num_records, desc=f"dbf {dbf_path.stem}", enabled=progress
        )
        for det in iterator:
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
            for name, field_type, length, decimal_count in BBOX_FIELDS:
                file.write(
                    _dbf_format_value(values[name], field_type, length, decimal_count)
                )
        file.write(b"\x1a")


# ---------------------------------------------------------------------------
# Public API (drop-in replacement for the per-backend writers)
# ---------------------------------------------------------------------------

def write_shapefile(shp_path, detections, reader, geometry: str = "bbox", progress: bool = True):
    """Write bbox or mask polygons to a shapefile with progress reporting."""
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

    started = time.monotonic()
    written = _write_shp_shx(shp_path, export_detections, geometry_builder, progress)
    _write_dbf(shp_path.with_suffix(".dbf"), export_detections, progress)
    _prj_writer(shp_path.with_suffix(".prj"), getattr(reader, "crs_wkt", None))

    elapsed = time.monotonic() - started
    if written:
        print(
            f"[shp] {shp_path.name}: {written} {geometry}(s) in {elapsed:.1f}s "
            f"({written / max(elapsed, 1e-6):.0f} det/s)",
            flush=True,
        )
    else:
        print(f"[shp] {shp_path.name}: no {geometry} detections to write", flush=True)