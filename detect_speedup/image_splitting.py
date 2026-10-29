# -*- coding: utf-8 -*-
"""Memory-safe image splitting for very large inputs.

The inference backends already perform windowed reads internally, but the
fallback Pillow reader triggers a decompression-bomb check on ``Image.open``.
When the source image is huge (e.g. >~180 Mpx) the Pillow fallback and any
pre-processing tool that materialises the full image will fail.

This module provides a streaming splitter:

* Reads dimensions from the file header only (rasterio / GDAL / tifffile)
* Reads each tile as a window directly from disk (no full pixel load)
* Saves tiles as PNG using geo-coordinate-aware naming so that downstream
  inference produces outputs in the standard ``{stem}_{row}_{col}_{bounds}``
  format
* Caches tiles next to the input file by default — the **output directory is
  never written to**
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from PIL import Image as _PILImage
except ImportError:  # pragma: no cover - PIL is part of the install
    _PILImage = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore


DEFAULT_MAX_SIDE = 2000
TILE_DIR_SUFFIX = "_auto_tiles"


# ---------------------------------------------------------------------------
# Backend probing
# ---------------------------------------------------------------------------

def _try_rasterio():
    try:
        import rasterio  # noqa: F401
        from rasterio.windows import Window  # noqa: F401
        return rasterio
    except Exception:
        return None


def _try_gdal():
    try:
        from osgeo import gdal  # noqa: F401
        return gdal
    except Exception:
        return None


def _try_tifffile():
    try:
        import tifffile  # noqa: F401
        return tifffile
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dimensions + geotransform
# ---------------------------------------------------------------------------

class _ImageMeta:
    __slots__ = ("width", "height", "transform", "crs_wkt", "backend")

    def __init__(self, width, height, transform=None, crs_wkt=None, backend=None):
        self.width = int(width)
        self.height = int(height)
        self.transform = transform  # 6-tuple or None
        self.crs_wkt = crs_wkt
        self.backend = backend


def get_image_meta(path: Path) -> _ImageMeta:
    """Return dimensions and (if available) the source geotransform/CRS."""
    path = Path(path)

    rasterio = _try_rasterio()
    if rasterio is not None:
        try:
            with rasterio.open(path) as ds:
                transform = ds.transform
                gt = None
                if transform is not None:
                    gt = (
                        float(transform.a),
                        float(transform.b),
                        float(transform.c),
                        float(transform.d),
                        float(transform.e),
                        float(transform.f),
                    )
                return _ImageMeta(
                    width=ds.width,
                    height=ds.height,
                    transform=gt,
                    crs_wkt=ds.crs.to_wkt() if ds.crs else None,
                    backend="rasterio",
                )
        except Exception:
            pass

    gdal = _try_gdal()
    if gdal is not None:
        try:
            ds = gdal.Open(str(path), gdal.GA_ReadOnly)
            if ds is not None:
                try:
                    gt = ds.GetGeoTransform(can_return_null=True)
                    crs = ds.GetProjection() or None
                    return _ImageMeta(
                        width=ds.RasterXSize,
                        height=ds.RasterYSize,
                        transform=tuple(gt) if gt else None,
                        crs_wkt=crs,
                        backend="gdal",
                    )
                finally:
                    ds = None
        except Exception:
            pass

    tifffile = _try_tifffile()
    if tifffile is not None and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                return _ImageMeta(
                    width=page.imagewidth,
                    height=page.imagelength,
                    transform=None,
                    crs_wkt=None,
                    backend="tifffile",
                )
        except Exception:
            pass

    # Last resort: PIL header only. May trigger bomb check on huge files.
    if _PILImage is None:
        raise RuntimeError(f"Cannot read metadata for {path}: no backend available")
    prev = _PILImage.MAX_IMAGE_PIXELS
    _PILImage.MAX_IMAGE_PIXELS = None
    try:
        with _PILImage.open(path) as img:
            return _ImageMeta(width=img.width, height=img.height)
    finally:
        _PILImage.MAX_IMAGE_PIXELS = prev


def get_image_dimensions(path: Path) -> Tuple[int, int]:
    meta = get_image_meta(path)
    return meta.width, meta.height


def _pixel_to_map(transform, x: float, y: float) -> Tuple[float, float]:
    """Apply a 6-tuple geotransform to convert pixel (x, y) to map (mx, my)."""
    if not transform:
        return float(x), float(y)
    a, b, c, d, e, f = transform
    mx = c + a * x + b * y
    my = f + d * x + e * y
    return float(mx), float(my)


def _format_geo(value: float) -> str:
    """Render a coordinate as a compact ``-68_133136``-style token.

    Mirrors the user's existing split convention (underscore separators,
    no scientific notation, sign preserved).
    """
    if value < 0:
        sign = "-"
        abs_value = -value
    else:
        sign = ""
        abs_value = value
    text = f"{abs_value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return f"{sign}{text.replace('.', '_')}"


def _tile_bounds_token(transform, x1: int, y1: int, x2: int, y2: int) -> str:
    """Return ``{xmin}_{ymax}_{xmax}_{ymin}`` in the source CRS units."""
    xmin, ymax = _pixel_to_map(transform, x1, y1)        # top-left
    xmax, ymin = _pixel_to_map(transform, x2, y2)        # bottom-right
    return "_".join(
        [_format_geo(xmin), _format_geo(ymax), _format_geo(xmax), _format_geo(ymin)]
    )


# ---------------------------------------------------------------------------
# Window reads (no full image load)
# ---------------------------------------------------------------------------

def read_window(path: Path, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Read a sub-region as ``(H, W, C)`` uint8 array (no full image load)."""
    path = Path(path)
    width = max(0, int(x2) - int(x1))
    height = max(0, int(y2) - int(y1))
    if width == 0 or height == 0:
        return np.zeros((height, width, 1), dtype=np.uint8)

    rasterio = _try_rasterio()
    if rasterio is not None:
        try:
            from rasterio.windows import Window
            with rasterio.open(path) as ds:
                window = Window(int(x1), int(y1), width, height)
                patch = ds.read(window=window)
            patch = np.asarray(patch)
            if patch.ndim == 2:
                patch = patch[:, :, None]
            else:
                patch = np.transpose(patch, (1, 2, 0))
            if patch.dtype != np.uint8:
                patch = _normalize_dtype(patch)
            return np.ascontiguousarray(patch)
        except Exception:
            pass

    gdal = _try_gdal()
    if gdal is not None:
        try:
            ds = gdal.Open(str(path), gdal.GA_ReadOnly)
            if ds is not None:
                try:
                    patch = ds.ReadAsArray(int(x1), int(y1), width, height)
                finally:
                    ds = None
            else:
                patch = None
            if patch is not None:
                patch = np.asarray(patch)
                if patch.ndim == 2:
                    patch = patch[:, :, None]
                else:
                    patch = np.transpose(patch, (1, 2, 0))
                if patch.dtype != np.uint8:
                    patch = _normalize_dtype(patch)
                return np.ascontiguousarray(patch)
        except Exception:
            pass

    tifffile = _try_tifffile()
    if tifffile is not None and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tif:
                page = tif.pages[0]
                try:
                    with tifffile.memmap(str(path)) as mm:
                        data = np.asarray(mm[int(y1):int(y2), int(x1):int(x2)])
                except Exception:
                    data = page.asarray()[int(y1):int(y2), int(x1):int(x2)]
            if data.ndim == 2:
                data = data[:, :, None]
            if data.dtype != np.uint8:
                data = _normalize_dtype(data)
            return np.ascontiguousarray(data)
        except Exception:
            pass

    # Pillow fallback — only safe for small images.
    if _PILImage is None:
        raise RuntimeError(f"Cannot read window from {path}: no backend available")
    prev = _PILImage.MAX_IMAGE_PIXELS
    _PILImage.MAX_IMAGE_PIXELS = None
    try:
        with _PILImage.open(path) as img:
            tile = img.crop((int(x1), int(y1), int(x2), int(y2)))
            return np.asarray(tile)
    finally:
        _PILImage.MAX_IMAGE_PIXELS = prev


def _normalize_dtype(patch: np.ndarray) -> np.ndarray:
    arr = np.asarray(patch)
    if arr.dtype == np.uint8:
        return arr
    arr = np.nan_to_num(arr)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    finite_min = float(arr.min())
    finite_max = float(arr.max())
    if finite_max <= 255.0 and finite_min >= 0.0:
        return np.clip(arr, 0, 255).astype(np.uint8)
    span = finite_max - finite_min
    if span < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - finite_min) / span * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Tile writing
# ---------------------------------------------------------------------------

def _array_to_pil(arr: np.ndarray) -> "Image.Image":
    if _PILImage is None:
        raise RuntimeError("Pillow is required to save tiles")
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        return _PILImage.fromarray(arr)
    channels = arr.shape[2]
    if channels == 1:
        return _PILImage.fromarray(arr[:, :, 0])
    if channels == 2:
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=arr.dtype)
        rgba[:, :, :2] = arr
        rgba[:, :, 3] = 255
        return _PILImage.fromarray(rgba, mode="RGBA")
    if channels == 3:
        return _PILImage.fromarray(arr[:, :, :3])
    return _PILImage.fromarray(arr[:, :, :4])


def _save_tile(args) -> Path:
    image_path, x1, y1, x2, y2, tile_path = args
    if tile_path.exists() and tile_path.stat().st_size > 0:
        return tile_path
    arr = read_window(image_path, x1, y1, x2, y2)
    img = _array_to_pil(arr)
    img.save(tile_path)
    return tile_path


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_tiles(width: int, height: int, max_side: int) -> List[Tuple[int, int, int, int, int, int]]:
    """Return list of ``(x1, y1, x2, y2, row, col)`` covering the image."""
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    cols = max(1, math.ceil(width / max_side))
    rows = max(1, math.ceil(height / max_side))
    tiles = []
    for row in range(rows):
        y1 = row * max_side
        y2 = min(y1 + max_side, height)
        for col in range(cols):
            x1 = col * max_side
            x2 = min(x1 + max_side, width)
            tiles.append((int(x1), int(y1), int(x2), int(y2), row, col))
    return tiles


def _resolve_tile_extension(image_path: Path) -> str:
    """PNG is lossless. JPEG saves space but is lossy; only use for photographic JPGs."""
    suf = image_path.suffix.lower()
    if suf in {".jpg", ".jpeg"}:
        return ".jpg"
    return ".png"


def _resolve_cache_dir(image_path: Path, override: Optional[Path]) -> Path:
    """Pick where tiles should be written — never inside the output dir.

    Order:
    1. ``override`` if provided
    2. ``<input_dir>/<stem>_auto_tiles/`` (sibling of the source file)
    """
    if override is not None:
        return Path(override)
    return image_path.parent / f"{image_path.stem}{TILE_DIR_SUFFIX}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_image_to_tiles(
    image_path: Path,
    max_side: int = DEFAULT_MAX_SIDE,
    num_workers: int = 1,
    overwrite: bool = False,
    cache_dir: Optional[Path] = None,
) -> List[Path]:
    """Split ``image_path`` into ``max_side``-sized tiles.

    Tiles are written under ``cache_dir`` (default: sibling of the source).
    Returns the list of tile paths. If the image already fits within
    ``max_side`` on both sides the original path is returned unchanged.
    Silent: callers (e.g. :func:`expand_input_paths`) are responsible for
    printing a single aggregate log line so we don't flood the terminal.
    """
    image_path = Path(image_path)
    try:
        meta = get_image_meta(image_path)
    except Exception:
        return [image_path]

    if max(meta.width, meta.height) <= max_side:
        return [image_path]

    tile_root = _resolve_cache_dir(image_path, cache_dir)
    if overwrite and tile_root.exists():
        for old in tile_root.glob("*"):
            try:
                old.unlink()
            except Exception:
                pass
    tile_root.mkdir(parents=True, exist_ok=True)

    ext = _resolve_tile_extension(image_path)
    tiles = plan_tiles(meta.width, meta.height, max_side)

    jobs = []
    for (x1, y1, x2, y2, row, col) in tiles:
        bounds = _tile_bounds_token(meta.transform, x1, y1, x2, y2)
        tile_name = f"{image_path.stem}_{row:03d}_{col:03d}_{bounds}{ext}"
        jobs.append((image_path, x1, y1, x2, y2, tile_root / tile_name))

    workers = max(1, int(num_workers or 1))
    if workers == 1 or len(jobs) <= 1:
        results = [_save_tile(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_save_tile, jobs))

    return [Path(r) for r in results]


def maybe_split(
    image_path: Path,
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
    num_workers: int = 1,
    cache_dir: Optional[Path] = None,
) -> List[Path]:
    """Split the image if it exceeds ``max_side``; otherwise return ``[path]``."""
    if max_side is None or max_side <= 0:
        return [Path(image_path)]
    return split_image_to_tiles(
        image_path=image_path,
        max_side=max_side,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )


def expand_input_paths(
    paths: Iterable[Path],
    max_side: Optional[int] = DEFAULT_MAX_SIDE,
    num_workers: int = 1,
    cache_dir: Optional[Path] = None,
    show_progress: bool = True,
) -> Tuple[List[Path], List[dict]]:
    """Apply :func:`maybe_split` to each input path.

    Returns ``(expanded_paths, split_records)`` where ``split_records`` is a
    list of ``{"source": Path, "tiles": int, "cache": Path}`` for every input
    that was actually split — used by the caller to emit a single summary log
    line without spamming the terminal.

    A tqdm bar wraps the file loop so the user sees ``auto-split: N/M``
    advance while tiles are written to disk; without it the run can look
    frozen for tens of seconds on directories of large rasters.
    """
    paths_list = list(paths)
    expanded: List[Path] = []
    records: List[dict] = []
    if not paths_list:
        return expanded, records

    use_tqdm = show_progress and tqdm is not None and len(paths_list) > 1
    iterator = tqdm(
        paths_list,
        total=len(paths_list),
        desc="auto-split",
        unit="file",
        dynamic_ncols=True,
    ) if use_tqdm else paths_list

    for path in iterator:
        path = Path(path)
        try:
            meta = get_image_meta(path)
        except Exception:
            expanded.append(path)
            if use_tqdm:
                iterator.set_postfix(file=path.name, status="skip")
            continue

        if max_side is None or max_side <= 0 or max(meta.width, meta.height) <= max_side:
            expanded.append(path)
            if use_tqdm:
                iterator.set_postfix(file=path.name, status="kept")
            continue

        try:
            tiles = split_image_to_tiles(
                image_path=path,
                max_side=max_side,
                num_workers=num_workers,
                cache_dir=cache_dir,
            )
        except Exception as exc:
            records.append({
                "source": path,
                "tiles": 0,
                "cache": None,
                "error": str(exc),
            })
            expanded.append(path)
            if use_tqdm:
                iterator.set_postfix(file=path.name, status="error")
            continue

        if len(tiles) == 1 and tiles[0] == path:
            expanded.append(path)
            if use_tqdm:
                iterator.set_postfix(file=path.name, status="kept")
            continue

        expanded.extend(tiles)
        cache_root = tiles[0].parent if tiles else None
        records.append({
            "source": path,
            "tiles": len(tiles),
            "cache": cache_root,
            "size": (meta.width, meta.height),
        })
        if use_tqdm:
            iterator.set_postfix(file=path.name, tiles=len(tiles))
    if use_tqdm:
        iterator.close()
    return expanded, records