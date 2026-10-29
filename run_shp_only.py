#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone JSON → single-merged-SHP exporter (no visualization).

Consumes the per-tile JSON files written by ``run_demo_config.py`` and
emits **one** polygon SHP (with ``class_id / class_name / score / thr /
bbox`` attributes) per run. The expensive per-tile rasterio/GDAL read
that the full ``run_visualization`` does for JPG rendering is skipped
here — when tiles follow the ``split_tif.py`` / ``image_splitting.py``
filename convention we recover the geotransform straight from the
filename and only open the (single) source TIF to read its CRS WKT.

Examples
--------
    # Mask polygons, threshold 0.3, single SHP
    python run_shp_only.py ./output/yolo/0627_1200/json \\
        --crs-source /data1/.../327.tif --geometry mask --threshold 0.3

    # Bbox only, no NMS
    python run_shp_only.py ./output/yolo/0627_1200/json \\
        --crs-source /data1/.../327.tif --geometry bbox --no-merged-shp-nms


  # 假设已跑过 run_demo_config.py，JSON 在 ./output/rtmdet_balunxiya/json/
  python run_shp_only.py ./output/rtmdet_balunxiya/json/ \
      --crs-source /data1/wzh26/code/detect-speedup/input/巴伦西亚/327.tif \
      --geometry mask --threshold 0.3
  # 产出: ./output/yolo/0627_1200/shps/all_mask.shp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from detect_speedup.visualize_predictions import (
    _get_source_crs_wkt,
    _parse_tile_bounds_from_filename,
    _WGS84_WKT,
    collect_records,
    iter_json_files,
    nms_records,
    polygon_area,
    write_polygon_shapefile,
)


# ---------------------------------------------------------------------------
# Lightweight reader — avoid rasterio/GDAL I/O on huge tile TIFs
# ---------------------------------------------------------------------------

class _FastReader:
    """A drop-in stand-in for ``_RasterioVisReader`` that needs no I/O.

    Built from the geo bounds encoded in the tile filename + the CRS WKT
    pulled from a single source TIF. Exposes the attributes used by
    ``pixel_to_map_points`` and ``collect_records``:
    ``width``, ``height``, ``transform`` (6-tuple), ``geotransform`` (alias),
    ``crs_wkt``, ``pixel_to_map(x, y)`` and a no-op ``close()``.
    """

    __slots__ = ("width", "height", "transform", "geotransform", "crs_wkt")

    def __init__(self, width: int, height: int,
                 transform_6tuple: Tuple[float, float, float, float, float, float],
                 crs_wkt: Optional[str]):
        self.width = int(width)
        self.height = int(height)
        self.transform = transform_6tuple
        # ``geotransform`` is the GDAL-style alias some helpers look for.
        self.geotransform = transform_6tuple
        self.crs_wkt = crs_wkt

    def pixel_to_map(self, x: float, y: float) -> Tuple[float, float]:
        a, b, c, d, e, f = self.transform
        return (c + a * x + b * y, f + d * x + e * y)

    def close(self) -> None:
        # Nothing to release.
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="JSON → single merged polygon SHP (no JPG visualization).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir",
                        help="Directory containing per-tile prediction JSON files")
    parser.add_argument("--output-dir", default=None,
                        help="SHP output directory (default: <input_dir>/../shps)")
    parser.add_argument("--shp-name", default=None,
                        help="Output SHP base name (default: all_<geometry>.shp)")
    parser.add_argument("--crs-source", default=None,
                        help="Source TIF whose CRS WKT is used for the .prj "
                             "sidecar. Strongly recommended for large images.")
    parser.add_argument("--geometry", choices=("mask", "bbox", "obb"),
                        default="mask",
                        help="Polygon geometry (default: mask)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Single confidence threshold (default: 0.3). "
                             "All detections with score >= threshold are written; "
                             "the value is recorded in the SHP 'thr' column.")
    parser.add_argument("--no-merged-shp-nms", action="store_true",
                        help="Skip global NMS before writing the merged SHP.")
    parser.add_argument("--merged-shp-iou", type=float, default=0.45,
                        help="IoU threshold for merged-SHP NMS (default: 0.45).")
    parser.add_argument("--merged-shp-class-aware-nms", action="store_true",
                        help="NMS only suppresses records with the same class_id "
                             "(default: class-agnostic).")
    parser.add_argument("--min-area", type=int, default=25,
                        help="Drop mask polygons with pixel area below N "
                             "(default: 25, 0 disables). mask geometry only.")
    parser.add_argument("--mask-fallback-bbox", action="store_true",
                        help="When geometry=mask but a detection has no segment_xy, "
                             "fall back to its bbox polygon.")
    parser.add_argument("--recursive", action="store_true",
                        help="Recurse into subdirectories for JSON files.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-tile timing breakdown.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Per-tile record collection (with no-file-open fast path)
# ---------------------------------------------------------------------------

def _try_build_fast_reader(json_path, image_path, image_size, crs_wkt):
    """Return a ``_FastReader`` if the tile filename encodes geo bounds; else None."""
    if image_size is None:
        return None
    parsed = _parse_tile_bounds_from_filename(image_path, (image_size["width"], image_size["height"]))
    if not parsed:
        return None
    _xmin, _ymin, _xmax, _ymax, gt = parsed
    return _FastReader(
        width=image_size["width"],
        height=image_size["height"],
        transform_6tuple=gt,
        crs_wkt=crs_wkt,
    )


def _min_area_filter(det, geometry: str, min_area: int) -> bool:
    """Return True if the detection should be kept under the min-area rule."""
    if min_area <= 0 or geometry != "mask":
        return True
    seg = det.get("segment_xy")
    if not seg or len(seg) < 3:
        return True  # let mask_fallback_bbox handle it downstream
    return polygon_area([(float(x), float(y)) for x, y in seg]) >= min_area


def collect_records_for_shp(json_path, image_path, threshold, geometry,
                            mask_fallback_bbox, crs_source_wkt,
                            crs_source_path, min_area, verbose):
    """Collect (single-threshold) records for one JSON, preferring no-I/O path.

    Returns ``(records, crs_wkt)`` mirroring ``collect_records``. ``records``
    is a flat list (not a dict keyed by threshold) because the new script
    only ever uses a single threshold.
    """
    import json as _json
    with json_path.open("r", encoding="utf-8") as f:
        prediction = _json.load(f)
    detections = prediction.get("detections", [])
    image_size = prediction.get("image_size")

    # Optional min-area pre-filter.
    if min_area > 0 and geometry == "mask":
        detections = [d for d in detections if _min_area_filter(d, geometry, min_area)]

    # Fast path: build a filename-only reader.
    fast = _try_build_fast_reader(json_path, image_path, image_size, crs_source_wkt)
    if fast is not None:
        if verbose:
            t0 = time.monotonic()
        # ``collect_records`` still iterates detections itself; we hand it our reader.
        per_thr, returned_crs = collect_records(
            json_path=json_path,
            image_path=image_path,
            thresholds=[threshold],
            geometry=geometry,
            mask_fallback_bbox=mask_fallback_bbox,
            crs_source_path=None,  # fast reader already carries the CRS
            reader=fast,
        )
        if verbose:
            print(f"  [fast] {json_path.name} -> {len(per_thr[threshold])} recs "
                  f"in {(time.monotonic() - t0) * 1000:.1f} ms", flush=True)
        return per_thr[threshold], returned_crs or crs_source_wkt

    # Fallback: use the full path (opens the tile TIF).
    if verbose:
        t0 = time.monotonic()
    per_thr, returned_crs = collect_records(
        json_path=json_path,
        image_path=image_path,
        thresholds=[threshold],
        geometry=geometry,
        mask_fallback_bbox=mask_fallback_bbox,
        crs_source_path=crs_source_path,
    )
    if verbose:
        print(f"  [fallback] {json_path.name} -> {len(per_thr[threshold])} recs "
              f"in {(time.monotonic() - t0) * 1000:.1f} ms", flush=True)
    return per_thr[threshold], returned_crs or crs_source_wkt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    input_dir = Path(os.path.abspath(args.input_dir))
    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / "shps"
    output_dir.mkdir(parents=True, exist_ok=True)

    shp_name = args.shp_name or f"all_{args.geometry}.shp"
    shp_path = output_dir / shp_name

    # Read CRS WKT once (cheap even on huge source TIFs — only metadata).
    crs_source_path = args.crs_source
    if crs_source_path:
        crs_source_wkt = _get_source_crs_wkt(crs_source_path)
        if not crs_source_wkt:
            print(f"[warn] failed to read CRS from {crs_source_path}, "
                  f"falling back to WGS84", flush=True)
            crs_source_wkt = _WGS84_WKT
    else:
        print("[warn] no --crs-source given; coordinates will be written as "
              "WGS84 lat/lon (verify this matches your data!)", flush=True)
        crs_source_wkt = _WGS84_WKT

    threshold = float(args.threshold)
    json_files = list(iter_json_files(input_dir, recursive=args.recursive))
    if not json_files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    print(
        f"[shp] input={input_dir} ({len(json_files)} JSONs)\n"
        f"      output={shp_path}\n"
        f"      geometry={args.geometry}, threshold={threshold}, "
        f"min_area={args.min_area}, nms={'off' if args.no_merged_shp_nms else f'on (IoU={args.merged_shp_iou})'}, "
        f"crs={'source-TIF' if crs_source_path else 'WGS84-default'}",
        flush=True,
    )

    # Iterate tiles — accumulate one flat list of records.
    all_records = []
    final_crs_wkt = crs_source_wkt
    processed = skipped = 0
    t_start = time.monotonic()
    for json_path in json_files:
        with json_path.open("r", encoding="utf-8") as f:
            import json as _json
            prediction = _json.load(f)
        image_value = prediction.get("image")
        image_path = Path(image_value) if image_value else json_path.with_suffix(".tif")

        # If the JSON's image field points to a missing file, the fast path
        # can still extract geo info from the filename (when the filename
        # itself is in the split_tif / image_splitting convention). For the
        # fallback path we need an existing file or we have to skip.
        if not image_path.exists() and not _try_build_fast_reader(
            json_path, image_path, prediction.get("image_size"), crs_source_wkt
        ):
            skipped += 1
            print(f"[warn] skip {json_path.name}: image not found at {image_path}",
                  flush=True)
            continue

        records, returned_crs = collect_records_for_shp(
            json_path=json_path,
            image_path=image_path,
            threshold=threshold,
            geometry=args.geometry,
            mask_fallback_bbox=args.mask_fallback_bbox,
            crs_source_wkt=crs_source_wkt,
            crs_source_path=crs_source_path,
            min_area=args.min_area,
            verbose=args.verbose,
        )
        all_records.extend(records)
        if returned_crs and final_crs_wkt is None:
            final_crs_wkt = returned_crs
        processed += 1

    elapsed = time.monotonic() - t_start
    print(
        f"[info] collected {len(all_records)} record(s) from {processed} JSON(s) "
        f"(skipped {skipped}) in {elapsed:.1f}s",
        flush=True,
    )

    if not all_records:
        print("[shp] no records to write — nothing emitted", flush=True)
        return

    # Optional global NMS.
    if not args.no_merged_shp_nms:
        before = len(all_records)
        t0 = time.monotonic()
        all_records = nms_records(
            all_records,
            iou_threshold=args.merged_shp_iou,
            agnostic=not args.merged_shp_class_aware_nms,
        )
        print(
            f"[nms] {before} -> {len(all_records)} records "
            f"(IoU={args.merged_shp_iou}, agnostic={not args.merged_shp_class_aware_nms}, "
            f"{time.monotonic() - t0:.1f}s)",
            flush=True,
        )

    # Write one merged SHP.
    write_polygon_shapefile(shp_path, all_records, crs_wkt=final_crs_wkt)
    print(
        f"[done] {shp_path} ({len(all_records)} polygon(s)); "
        f"sidecars: .shx .dbf .prj",
        flush=True,
    )


if __name__ == "__main__":
    main()
