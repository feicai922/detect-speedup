# detect-speedup

大图（超大 GeoTIFF）滑窗推理工具：**检测 / 实例分割**，统一入口 `run_large.py`，
后端可插拔（原生：YOLO / RTMDet / RF-DETR；插件：RT-DETR(HF) / SKYDET / 以后继续加）。

## 统一入口

```bash
# 看当前有哪些后端
python run_large.py --list-backends

# 原生后端（走原有 多进程多卡 流水线）
python run_large.py --backend yolo \
    --model /path/to/best.pt --input /path/to/pre.tif --out ./output/xxx \
    --tile 1024 --imgsz 640 --conf 0.3 --batch 8 --device cuda:0

python run_large.py --backend rtmdet --model cfg.pth --backend-config cfg.py \
    --input pre.tif --out ./output/xxx --device cuda:0
python run_large.py --backend rfdetr --model ckpt.pth --input pre.tif --out ./output/xxx --imgsz 648

# 插件后端（通用滑窗循环，见 detect_speedup/backends/）
python run_large.py --backend skydet \
    --model /data1/wzh26/runs/skydet_gzda_v1_640_bs8/best_stg1.pth \
    --backend-config /data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml \
    --input pre.tif --out ./output/skydet --tile 640 --stride 512
python run_large.py --backend rtdetr-hf --model /path/to/hf_model_dir \
    --input pre.tif --out ./output/rtdetr

# 参数也可以全部写进 JSON/YAML
python run_large.py -c my_run.yml

# 环境变量兜底：DS_BACKEND / DS_INPUT / DS_OUT / DS_MODEL / DS_DEVICE / DS_TILE / DS_STRIDE / DS_BATCH / DS_CONF
```

输出统一为 `<out>/json/*.json`（检测结果，含 bbox/score/class/segment_xy）、
`<out>/shps/all_*_conf_*.shp`（按阈值分档的合并矢量，EPSG:4326）、`<out>/vis/*_vis.jpg`。
`--no-vis` 只跑推理，`--no-json` 不落 JSON。

## 目录结构

```
detect-speedup/
├── run_large.py                     # ★ 统一入口（后端无关）
├── detect_speedup/
│   ├── backends/                    # ★ 后端插件层
│   │   ├── __init__.py              #   注册表：register() / get_backend() / all_backends()
│   │   ├── base.py                  #   TileBackend / TilePrediction 基类
│   │   ├── runner.py                #   通用滑窗循环（切块→组批→NMS→JSON→SHP/可视化）
│   │   ├── rtdetr_hf.py             #   插件：HuggingFace RT-DETR
│   │   ├── skydet.py                #   插件：SKYDET（DINOv3-ConvNeXt + D-FINE 系）
│   │   └── _template.py             #   ★ 新增后端照着抄
│   ├── run_large_image_inference.py # 原生后端统一 API（run_yolo/run_rtmdet/run_rfdetr）
│   ├── model_backend_adapters.py    # 原生后端适配
│   ├── batch_multi_device_runner.py # 多卡多设备调度
│   ├── yolo_large_image_inference.py / rtmdet_* / rfdetr_*   # 各原生后端大图核心
│   ├── yolo_image_io_and_export.py  # 图像读写 / 窗口生成 / global NMS / 检测框落盘
│   ├── shapefile_export.py          # Shapefile 写出
│   ├── building_regularize.py       # 建筑物轮廓正则化
│   ├── building_vector_export.py    # 区域裁切与矢量导出
│   ├── image_splitting.py           # 大图预切块
│   ├── visualize_predictions.py     # 可视化 + 合并 SHP
│   └── runtime_warning_filters.py / process_title_utils.py
├── projects/                        # 一次性/项目专用脚本（不是通用入口）
│   ├── pre1023/                     #   1023 测试影像 pre.tif 的各模型推理脚本
│   ├── guizhou/  dinov3/  p2pformer/
├── archive/legacy_entrypoints/      # 被 run_large.py 取代的旧入口（保留备查）
├── tools/nvidia.py                  # 小工具
├── tmp/                             # 临时产物（QA 切片、旧中间结果）
├── logs/                            # 运行日志
├── input/  output/                  # 输入影像 / 推理产物
└── run_vis.py  run_shp_only.py  split_tif.py  stream_json_to_shp.py  run_all.sh  # 仍然可用的小工具
```

## 新增一个后端

1. 复制 `detect_speedup/backends/_template.py` 为 `backends/你的名字.py`；
2. 继承 `TileBackend`，实现两个方法：
   - `load(device)`：把权重加载到 `self.model`；
   - `infer(tiles, windows, device)`：一批瓦片（`HxWx3` uint8 RGB）→ 每块 `TilePrediction(boxes, scores, labels[, segment_xy])`，
     坐标用**瓦片内像素**（runner 会自动加回窗口偏移）；
3. 文件末尾 `register("你的名字", YourBackend)`；
4. 在 `backends/__init__.py` 末尾加一行 `from . import 你的名字  # noqa`；
5. `python run_large.py --list-backends` 能看到即可用。

切块、组批、置信度过滤、global NMS、JSON 落盘、Shapefile 导出、可视化都由
`backends/runner.py` 统一完成，插件里**不要**自己过滤分数或做 NMS。

> 说明：`yolo / rtmdet / rfdetr` 仍走框架原有的多进程多卡流水线（性能更好），
> 在 `backends/__init__.py` 的 `NATIVE_BACKENDS` 里登记；插件后端走单进程通用循环。

## 快速开始

### 1. 修改配置（旧方式，仍可用：`run_demo_config.py` 里改 config dict；推荐用上面的 `run_large.py`）

```python
# 选择后端: "yolo" | "rtmdet" | "rfdetr"
BACKEND = "yolo"

# 通用配置
COMMON_BASE_CONFIG = {
    "input_path": "/path/to/your/image_or_dir",
    "output_dir": "./output/demo",
    "devices": ["cuda:0"],  # GPU 推理
    ...
}

# 模型配置（三选一，按需填写）
YOLO_CONFIG = {"model_path": "/path/to/best.pt", ...}
RTMDET_CONFIG = {"config_file": "...", "checkpoint_file": "...", ...}
RFDETR_CONFIG = {"checkpoint": "/path/to/checkpoint.pth", ...}
```

### 2. 运行

```bash
python run_demo.py
```

推理完成后会自动运行可视化，生成 `*_vis.jpg` 和汇总 SHP 文件。

### 3. 环境变量覆盖

```bash
INFER_BACKEND=rtmdet \
INFER_INPUT_PATH=/path/to/images \
INFER_OUTPUT_DIR=./output/rtmdet \
INFER_DEVICES=cuda:0,cuda:1 \
python run_demo.py
```

## 建筑物轮廓正则化

推理完成后会对掩膜检测结果做建筑物轮廓正则化（面积过滤 →
Douglas-Peucker 简化 → 主方向检测 → 直角化 → 对称化 → 边界平滑 →
拓扑修复），算法来自 `building-regularize` 项目。启用方式：在
`run_demo_config.py` / `run_yolo.py` 中给 `COMMON["regularize"]` 填一个
dict（或直接使用内置的 `REGULARIZE` 配置块）：

```python
COMMON = {
    ...
    "regularize": {
        "min_area": 20.0,        # 最小面积 (m²)，过滤误检小碎块
        "dp_tolerance": 0.5,     # Douglas-Peucker 简化容差 (m)
        "angle_threshold": 10.0, # 直角化角度吸附阈值 (°)
        "snap_angles": [0, 45, 90, 135],
        "enable_symmetry": False,
        "smooth_iterations": 0,  # Chaikin 平滑迭代次数
        "fix_topology": True,
    },
}
```

- 每个影像的掩膜经过正则化后额外写出 `*_mask_regularized.shp`（含
  .shp/.shx/.dbf/.prj），同时检测 JSON 中会写入 `segment_xy_regularized`
  （正则化后的像素轮廓），可视化 / 合并 SHP 会优先绘制正则化结果。
  不填（None）则不启用。
- 输入为 WGS84 经纬度且安装了 `pyproj` 时，自动投影到 UTM（容差单位为米）；
  平面/像素坐标数据直接在源坐标系下执行；WGS84 数据但缺少 `pyproj` 会跳过
  并给出提示（避免用米制容差处理经纬度）。

## Python API

```python
from detect_speedup import run_inference, run_yolo, run_rtmdet, run_rfdetr

common = {"input_path": "/path/to/images", "output_dir": "./out", "devices": ["cuda:0"]}
yolo = {"model_path": "best.pt", "imgsz": 1024, "conf": 0.3}

run_yolo(common=common, yolo=yolo)
```

## 可视化

- 结果统一使用**绿色**渲染（不再按置信度阈值区分颜色），掩膜填充/描边均为绿色。
- 可设置 `fill_alpha` 控制填充透明度（0=仅描边）。

```python
from detect_speedup.visualize_predictions import run_visualization

run_visualization({
    "input_dir": "./output/demo",
    "output_dir": "./vis_output",
    "image_dir": "/path/to/source_images",
    "geometry": "bbox",       # bbox | mask | obb
    "thresholds": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
})
```

也支持 CLI：

```bash
python -m detect_speedup.visualize_predictions ./output/json_dir ./vis_output \
  --image-dir /path/to/images --geometry bbox
```

## 设备说明

| 设备前缀 | 示例 | 说明 |
|---------|------|------|
| `cuda:N` | `cuda:0`, `cuda:1` | NVIDIA GPU |
| `gpu:N` | `gpu:0` | 自动映射为 `cuda:N` |
| `npu:N` | `npu:0`, `npu:1` | 华为昇腾 NPU |
| `cpu` | `cpu` | CPU 推理 |

## 输出

每张图默认生成：

- `*.json` — 检测结果
- `*_bbox.shp` — 检测框（含 .shp/.shx/.dbf/.prj）
- `*_mask.shp` — 分割掩膜（如有）
- `*_mask_regularized.shp` — 建筑物轮廓正则化后的掩膜（开启正则化时）
- `*_vis.jpg` — 可视化结果（统一绿色）

## RF-DETR Seg Medium 建筑实例与区域导出

RF-DETR 后端现可按训练检查点 `model_name` / `model_config` 自动恢复 Seg 变体，使用严格权重匹配；不再将分割权重误装为检测 Large。结果 JSON 的 `mask_geometry` 保留孔洞与多连通区域，`segment_xy` 保留主要轮廓供旧可视化兼容。可用 common 中 `fp16=True` 开启 CUDA 混合精度推理。

针对两个国内最佳模型与 1023 测试影像的复现入口：`run_building_best_pre.py`。区域裁切与矢量导出模块为 `detect_speedup/building_vector_export.py`，提供独立模型的全图和 EPSG:4549 园头村结果。输出说明见 `/data1/wzh26/重研/outputs/pre_20261028_rfdetr_best/README.md`。
