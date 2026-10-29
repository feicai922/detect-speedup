#!/usr/bin/env python3
"""流式生成建筑物分布热力图（适用于大量拆分 SHP）。

依赖：fiona shapely pyproj rasterio numpy scipy matplotlib tqdm
"""
from __future__ import annotations

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[2]))  # repo root
import argparse
from pathlib import Path
import warnings

import fiona
import numpy as np
import rasterio
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from pyproj import CRS, Transformer
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:  # 允许不安装 tqdm 时仍可运行
    def tqdm(iterable, **kwargs):
        total = kwargs.get("total", "?")
        for n, item in enumerate(iterable, 1):
            print(f"进度: {n}/{total}", flush=True)
            yield item


def shp_files(folder: Path):
    return sorted(folder.rglob("*.shp"))


def get_extent(files, label="扫描范围"):
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    crs_text = None
    for path in tqdm(files, total=len(files), desc=label, unit="shp"):
        try:
            with fiona.open(path) as src:
                b = src.bounds
                xmin, ymin = min(xmin, b[0]), min(ymin, b[1])
                xmax, ymax = max(xmax, b[2]), max(ymax, b[3])
                if crs_text is None:
                    crs_text = src.crs_wkt or src.crs
        except Exception as exc:
            warnings.warn(f"跳过无法读取的 {path}: {exc}")
    if not np.isfinite(xmin):
        raise RuntimeError("没有找到可读取的 SHP")
    return (xmin, ymin, xmax, ymax), crs_text


def one_dataset(folder: Path, out_prefix: Path, max_size: int, batch: int,
                sigma: float, target_crs: str | None):
    files = shp_files(folder)
    if not files:
        raise RuntimeError(f"{folder} 下没有 SHP 文件")
    print(f"[{folder.name}] 开始扫描 SHP 范围，共 {len(files)} 个文件…", flush=True)
    bounds, source_crs = get_extent(files, label=f"{folder.name} 扫描")
    print(f"[{folder.name}] 范围扫描完成，开始栅格化…", flush=True)
    # 默认直接在经纬度网格中统计；也支持指定米制投影（如 EPSG:32648）。
    if target_crs:
        src = CRS.from_user_input(source_crs or "EPSG:4326")
        dst = CRS.from_user_input(target_crs)
        project = Transformer.from_crs(src, dst, always_xy=True).transform
        corners = [project(bounds[0], bounds[1]), project(bounds[0], bounds[3]),
                    project(bounds[2], bounds[1]), project(bounds[2], bounds[3])]
        bounds = (min(x for x, _ in corners), min(y for _, y in corners),
                  max(x for x, _ in corners), max(y for _, y in corners))
    else:
        project = None
    w0, h0 = bounds[2] - bounds[0], bounds[3] - bounds[1]
    scale = min(1.0, max_size / max(w0, h0))
    width, height = max(1, int(np.ceil(w0 * scale))), max(1, int(np.ceil(h0 * scale)))
    tr = from_bounds(*bounds, width, height)
    counts = np.zeros((height, width), dtype=np.uint32)
    def add_batch(items):
        # 直接写入累计栅格，避免每批再分配一个同样大小的临时数组。
        rasterize(items, out=counts, transform=tr, fill=0,
                  dtype="uint32", merge_alg=MergeAlg.add)
    total = 0
    file_iter = tqdm(files, total=len(files), desc=folder.name, unit="shp")
    for i, path in enumerate(file_iter, 1):
        if hasattr(file_iter, "set_postfix"):
            file_iter.set_postfix(file=path.name[:24], buildings=f"{total:,}")
        try:
            with fiona.open(path) as src:
                geoms = []
                for feat in src:
                    if not feat.get("geometry"):
                        continue
                    try:
                        geom = shape(feat["geometry"])
                        if project:
                            geom = shp_transform(project, geom)
                        if not geom.is_empty:
                            geoms.append((geom, 1))
                    except Exception:
                        continue
                    if len(geoms) >= batch:
                        add_batch(geoms)
                        total += len(geoms); geoms.clear()
                if geoms:
                    add_batch(geoms)
                    total += len(geoms)
        except Exception as exc:
            warnings.warn(f"跳过 {path}: {exc}")
        # tqdm 的 postfix 会在同一行显示累计要素数，不产生大量日志。
        if hasattr(file_iter, "set_postfix"):
            file_iter.set_postfix(buildings=f"{total:,}")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": height, "width": width, "count": 1,
               "dtype": "uint32", "transform": tr, "compress": "zstd", "nodata": 0}
    if target_crs:
        profile["crs"] = target_crs
    with rasterio.open(str(out_prefix) + "_count.tif", "w", **profile) as dst:
        dst.write(counts, 1)
    density = gaussian_filter(counts.astype(np.float32), sigma=sigma) if sigma > 0 else counts.astype(np.float32)
    nz = density[density > 0]
    vmax = float(np.percentile(nz, 99.5)) if nz.size else 1.0
    # 透明背景便于叠加到地图；颜色上限用 99.5 分位数，避免少数超密区域压扁整体对比度。
    norm = plt.Normalize(vmin=0, vmax=max(vmax, 1e-6))
    rgba = plt.get_cmap("inferno")(norm(density))
    rgba[..., 3] = (density > 0).astype(np.float32)
    plt.imsave(str(out_prefix) + ".png", rgba, origin="upper")
    print(f"完成: {out_prefix}.png  ({width}x{height}), {total:,} 个要素")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", type=Path, required=True, help="一个或多个输出目录")
    ap.add_argument("--output-dir", type=Path, default=Path("output/guizhou_heatmaps"))
    ap.add_argument("--max-size", type=int, default=10000, help="最长边像素数，高清可设 16000")
    ap.add_argument("--batch", type=int, default=5000, help="每批栅格化的要素数")
    ap.add_argument("--sigma", type=float, default=2.0, help="热力平滑像素半径，0 表示不平滑")
    ap.add_argument("--target-crs", default=None, help="可选米制投影，如 EPSG:32648")
    args = ap.parse_args()
    for folder in args.input:
        one_dataset(folder, args.output_dir / folder.name, args.max_size, args.batch,
                    args.sigma, args.target_crs)


if __name__ == "__main__":
    main()
