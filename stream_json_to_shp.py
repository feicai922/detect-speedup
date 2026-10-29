#!/usr/bin/env python
"""Bounded-memory exporter for one streamed large-image prediction JSON.

The exporter makes two passes over ``detections``: the first calculates the
SHP headers, the second writes .shp/.shx/.dbf records.  It deliberately skips
global NMS, because global NMS requires retaining all detections in RAM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import time
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # Export remains usable in minimal environments.
    tqdm = None

from detect_speedup.shapefile_export import (
    BBOX_FIELDS,
    _build_polygon_record,
    _dbf_field_descriptor,
    _dbf_format_value,
    _file_header,
    _prj_writer,
    bbox_to_ring_coords,
    mask_to_ring_coords,
)
from detect_speedup.yolo_image_io_and_export import open_reader


def parse_args():
    parser = argparse.ArgumentParser(description="Stream one prediction JSON into a shapefile.")
    parser.add_argument("json_path", help="Completed streamed prediction JSON (not .partial)")
    parser.add_argument("--source", required=True, help="Source raster, used only for CRS/geotransform metadata")
    parser.add_argument("--output", required=True, help="Destination .shp path")
    parser.add_argument("--geometry", choices=("mask", "bbox"), default="mask")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument(
        "--recover-truncated-tail",
        action="store_true",
        help=(
            "Export every complete detection when a streamed input has a "
            "truncated tail or a malformed detection. Malformed detections "
            "are skipped only when a following complete record can be found."
        ),
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show per-pass JSON read progress when tqdm is installed (default: enabled).",
    )
    return parser.parse_args()


def _next_record_start(buffer, start):
    """Find the next top-level detection after a malformed flat record.

    YOLO streamed detections contain no nested JSON objects: geometry is
    represented by arrays.  Therefore the ``},{`` separator is unambiguous
    here and lets recovery discard only the bad record rather than retaining
    the rest of a multi-gigabyte file in ``buffer``.
    """
    match = re.search(r",\s*\{", buffer[start:])
    return start + match.end() - 1 if match else None


def iter_detections(
    json_path,
    chunk_size=1024 * 1024,
    recover_truncated_tail=False,
    progress=None,
):
    """Yield items from the top-level detections array without loading it all.

    In recovery mode, malformed records are skipped when a subsequent object
    boundary is already available in the current chunk. EOF is also accepted
    after complete records when only structural closing bytes are missing.
    """
    decoder = json.JSONDecoder()
    marker = re.compile(r'"detections"\s*:\s*\[')
    with Path(json_path).open("r", encoding="utf-8") as file:
        def update_progress():
            # ``tell`` reflects consumed source characters and avoids an extra
            # read or a full-file pre-scan.  It is intentionally independent
            # from the number of detections, which is unknown for damaged JSON.
            if progress is not None:
                position = file.tell()
                if position > progress.n:
                    progress.update(position - progress.n)

        buffer = ""
        while True:
            marker_match = marker.search(buffer)
            if marker_match:
                buffer = buffer[marker_match.end():]
                break
            chunk = file.read(chunk_size)
            if not chunk:
                raise ValueError(f"{json_path} does not contain a detections array")
            buffer += chunk
            update_progress()

        yielded_count = 0
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            while True:
                # A chunk can end exactly after a decoded object.  Once the
                # next chunk arrives it may begin with the array terminator;
                # check delimiters before asking JSONDecoder to decode it.
                if buffer.startswith("]"):
                    return
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                    continue
                if not buffer:
                    chunk = file.read(chunk_size)
                    if not chunk:
                        if recover_truncated_tail and yielded_count:
                            print(
                                f"[shp] warning: {json_path} is missing its final array/root "
                                f"closing bytes; exporting {yielded_count} complete detections",
                                flush=True,
                            )
                            return
                        raise ValueError(f"Truncated or invalid detections array in {json_path}")
                    buffer += chunk
                    update_progress()
                    continue
                try:
                    detection, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    if not isinstance(detection, dict):
                        raise ValueError(f"Invalid non-object detection in {json_path}")
                    yielded_count += 1
                    yield detection
                    break
                except json.JSONDecodeError as error:
                    # A syntax error followed by `,{` well before the current
                    # buffer end is a malformed record, not a chunk split.
                    # Recover immediately so a bad record cannot cause the
                    # remaining gigabytes to accumulate in memory.
                    if recover_truncated_tail and error.pos < len(buffer) - 1:
                        next_start = _next_record_start(buffer, max(error.pos, 1))
                        if next_start is not None:
                            print(
                                f"[shp] warning: skipped malformed detection near byte "
                                f"{file.tell() - len(buffer) + error.pos}: {error.msg}",
                                flush=True,
                            )
                            buffer = buffer[next_start:]
                            continue
                    chunk = file.read(chunk_size)
                    if not chunk:
                        # Only after EOF is it safe to treat a decode failure
                        # as corruption rather than an object split at a
                        # chunk boundary.  In recovery mode, discard the
                        # malformed object if a later complete object exists.
                        if recover_truncated_tail:
                            next_start = _next_record_start(buffer, max(error.pos, 1))
                            if next_start is not None:
                                print(
                                    f"[shp] warning: skipped malformed detection near byte "
                                    f"{file.tell() - len(buffer) + error.pos}: {error.msg}",
                                    flush=True,
                                )
                                buffer = buffer[next_start:]
                                continue
                        # Do not silently turn arbitrary invalid JSON into an
                        # export.  Recovery is allowed only if the last parsed
                        # item was complete and the remaining tail is empty or
                        # a single separator comma.
                        if recover_truncated_tail and yielded_count and buffer.strip() in ("", ","):
                            print(
                                f"[shp] warning: {json_path} is missing its final array/root "
                                f"closing bytes; exporting {yielded_count} complete detections",
                                flush=True,
                            )
                            return
                        offset = file.tell() - len(buffer) + error.pos
                        raise ValueError(
                            f"Truncated or invalid detection near byte {offset} in {json_path}: "
                            f"{error.msg}"
                        ) from error
                    buffer += chunk
                    update_progress()


def iter_records(
    json_path,
    reader,
    geometry,
    threshold,
    recover_truncated_tail=False,
    progress=None,
):
    for detection in iter_detections(
        json_path,
        recover_truncated_tail=recover_truncated_tail,
        progress=progress,
    ):
        if float(detection.get("score", 0.0)) < threshold:
            continue
        if geometry == "mask":
            ring = mask_to_ring_coords(reader, detection.get("segment_xy"))
        else:
            ring = bbox_to_ring_coords(reader, detection["bbox_xyxy"])
        if ring is None:
            continue
        content, bbox = _build_polygon_record(ring)
        yield detection, content, bbox


def write_dbf_header(file, count):
    header_length = 32 + 32 * len(BBOX_FIELDS) + 1
    record_length = 1 + sum(field[2] for field in BBOX_FIELDS)
    file.write(b"\x03" + b"\x00" * 3)
    file.write(struct.pack("<LHH", count, header_length, record_length) + b"\x00" * 20)
    for field in BBOX_FIELDS:
        file.write(_dbf_field_descriptor(*field))
    file.write(b"\r")


def write_dbf_record(file, detection):
    values = {
        "class_id": detection["class_id"], "class_name": detection["class_name"],
        "score": detection["score"], "x1": detection["bbox_xyxy"][0],
        "y1": detection["bbox_xyxy"][1], "x2": detection["bbox_xyxy"][2],
        "y2": detection["bbox_xyxy"][3],
    }
    file.write(b" ")
    for name, field_type, length, decimals in BBOX_FIELDS:
        file.write(_dbf_format_value(values[name], field_type, length, decimals))


def _patch_shp_header(path, file_length_words, bounds):
    with path.open("r+b") as file:
        file.seek(24)
        file.write(struct.pack(">i", file_length_words))
        file.seek(36)
        file.write(struct.pack("<4d", *bounds))


def _patch_dbf_count(path, count):
    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<L", count))


def make_progress(enabled, json_path, description):
    """Create a byte-oriented progress bar, or ``None`` when unavailable."""
    if not enabled or tqdm is None:
        return None
    return tqdm(
        total=json_path.stat().st_size,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        mininterval=0.5,
    )


def main():
    args = parse_args()
    json_path, shp_path = Path(args.json_path), Path(args.output)
    if not json_path.is_file() or json_path.name.endswith(".partial"):
        raise SystemExit(f"Expected a completed JSON file, got: {json_path}")
    shp_path.parent.mkdir(parents=True, exist_ok=True)
    reader = open_reader(args.source)
    started = time.monotonic()
    try:
        count = 0
        content_words = 0
        bounds = [float("inf"), float("inf"), float("-inf"), float("-inf")]
        progress = make_progress(args.progress, json_path, "SHP pass 1/2 (scan)")
        try:
            for _detection, content, bbox in iter_records(
                json_path, reader, args.geometry, args.threshold,
                args.recover_truncated_tail, progress,
            ):
                count += 1
                content_words += len(content) // 2 + 4  # record header + content
                bounds[0] = min(bounds[0], bbox[0]); bounds[1] = min(bounds[1], bbox[1])
                bounds[2] = max(bounds[2], bbox[2]); bounds[3] = max(bounds[3], bbox[3])
        finally:
            if progress is not None:
                progress.close()
        if not count:
            print("[shp] no matching detections; nothing exported", flush=True)
            return

        # ESRI Shapefile headers use signed 32-bit word lengths.  Split large
        # exports into bounded parts; the first part retains the requested
        # output name for compatibility.
        max_part_words = 900_000_000
        max_part_records = max(1, min(100_000, max_part_words // max(1, content_words // count)))
        file_bounds = tuple(bounds)
        progress = make_progress(args.progress, json_path, "SHP pass 2/2 (write)")
        part = 0; written = 0; offset_words = 50; part_count = 0
        shp = shx = dbf = None; part_paths = []
        try:
            for detection, content, _bbox in iter_records(
                json_path, reader, args.geometry, args.threshold,
                args.recover_truncated_tail, progress,
            ):
                if shp is None or part_count >= max_part_records:
                    if shp is not None:
                        dbf.write(b"\x1a"); shp.close(); shx.close(); dbf.close()
                        _patch_shp_header(part_paths[-1][0], offset_words, file_bounds)
                        _patch_shp_header(part_paths[-1][1], 50 + part_count * 4, file_bounds)
                        _patch_dbf_count(part_paths[-1][2], part_count)
                    part += 1; part_count = 0; offset_words = 50
                    stem = shp_path.stem if part == 1 else f"{shp_path.stem}_part{part:03d}"
                    base = shp_path.with_name(f"{stem}.partial")
                    ps, px, pd = (base.with_name(f"{base.name}{s}") for s in (".shp", ".shx", ".dbf"))
                    shp, shx, dbf = ps.open("wb"), px.open("wb"), pd.open("wb")
                    shp.write(_file_header(50, file_bounds)); shx.write(_file_header(50, file_bounds)); write_dbf_header(dbf, 0)
                    pp = base.with_name(f"{base.name}.prj")
                    _prj_writer(pp, getattr(reader, "crs_wkt", None))
                    part_paths.append((ps, px, pd, pp))
                content_length = len(content) // 2
                shp.write(struct.pack(">2i", part_count + 1, content_length) + content)
                shx.write(struct.pack(">2i", offset_words, content_length)); offset_words += 4 + content_length
                write_dbf_record(dbf, detection); part_count += 1; written += 1
        finally:
            if progress is not None: progress.close()
        if shp is not None:
            dbf.write(b"\x1a"); shp.close(); shx.close(); dbf.close()
            _patch_shp_header(part_paths[-1][0], offset_words, file_bounds); _patch_shp_header(part_paths[-1][1], 50 + part_count * 4, file_bounds); _patch_dbf_count(part_paths[-1][2], part_count)
        for paths in part_paths:
            final_stem = paths[3].name.removesuffix('.partial.prj')
            finals = [shp_path.with_name(f"{final_stem}{s}") for s in ('.shp', '.shx', '.dbf', '.prj')]
            for partial, final in zip(paths, finals):
                if partial.exists(): os.replace(partial, final)
        print(f"[shp] wrote {written} {args.geometry} polygons in {len(part_paths)} parts in {time.monotonic() - started:.1f}s", flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()
