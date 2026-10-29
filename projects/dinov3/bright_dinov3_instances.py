#!/usr/bin/env python3
"""Convert DINOv3 semantic label GeoTIFFs into building instances.

Each connected building component receives the majority non-background class
from the semantic prediction. Output is a GeoJSON FeatureCollection suitable
for GIS inspection and downstream conversion to SHP.
"""
from __future__ import annotations

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import mapping, shape
from scipy import ndimage


NAMES = {1: "intact", 2: "damaged", 3: "destroyed"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("labels_dir", help="Directory containing *_pred_label.tif files")
    p.add_argument("output", help="Output GeoJSON path")
    p.add_argument("--min-area", type=int, default=9,
                   help="Minimum component area in pixels (default: 9)")
    p.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    return p.parse_args()


def convert_one(path: Path, min_area: int, connectivity: int):
    with rasterio.open(path) as src:
        labels = src.read(1)
        transform = src.transform
        crs = src.crs

    structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
    building_mask = labels > 0
    component_ids, count = ndimage.label(building_mask, structure=structure)
    features = []
    for component_id in range(1, count + 1):
        component = component_ids == component_id
        area = int(component.sum())
        if area < min_area:
            continue
        values, counts = np.unique(labels[component], return_counts=True)
        valid = values > 0
        if not valid.any():
            continue
        class_id = int(values[valid][np.argmax(counts[valid])])
        polygons = list(shapes(component.astype(np.uint8), mask=component, transform=transform))
        for geom, value in polygons:
            if value != 1:
                continue
            geometry = shape(geom)
            if geometry.is_empty:
                continue
            minx, miny, maxx, maxy = geometry.bounds
            features.append({
                "type": "Feature",
                "geometry": mapping(geometry),
                "properties": {
                    "source": path.name,
                    "class_id": class_id,
                    "class_name": NAMES[class_id],
                    "area_px": area,
                    "bbox": [minx, miny, maxx, maxy],
                },
            })
    return features, crs


def main():
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    paths = sorted(labels_dir.glob("*_pred_label.tif"))
    if not paths:
        raise SystemExit(f"No *_pred_label.tif files found in {labels_dir}")
    all_features = []
    crs = None
    for path in paths:
        features, file_crs = convert_one(path, args.min_area, args.connectivity)
        all_features.extend(features)
        crs = crs or file_crs
        print(f"[instances] {path.name}: {len(features)} instances", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {"type": "FeatureCollection", "features": all_features}
    if crs:
        document["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    output.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    print(f"[instances] wrote {len(all_features)} instances to {output}", flush=True)


if __name__ == "__main__":
    main()
