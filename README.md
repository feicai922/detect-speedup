# detect-speedup

统一大图推理工具，支持 **YOLO / RTMDet / RF-DETR** 三种后端，GPU 推理 + 结果可视化。

## 目录结构

```
detect-speedup/
├── run_demo.py              # 唯一入口，修改配置后直接运行
└── detect_speedup/           # 所有核心代码
    ├── run_large_image_inference.py   # 统一推理 API
    ├── model_backend_adapters.py      # YOLO / RTMDet / RF-DETR 后端适配
    ├── batch_multi_device_runner.py   # 多卡多设备调度
    ├── yolo_large_image_inference.py  # YOLO 大图推理核心
    ├── rtmdet_large_image_inference.py # RTMDet 大图推理核心
    ├── rfdetr_large_image_inference.py # RF-DETR 大图推理核心
    ├── yolo_image_io_and_export.py    # 图像读写 & Shapefile 导出
    ├── visualize_predictions.py       # 结果可视化
    ├── runtime_warning_filters.py     # NPU 警告过滤
    └── process_title_utils.py         # 进程标题工具
```

## 快速开始

### 1. 修改 `run_demo.py` 中的配置

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

## Python API

```python
from detect_speedup import run_inference, run_yolo, run_rtmdet, run_rfdetr

common = {"input_path": "/path/to/images", "output_dir": "./out", "devices": ["cuda:0"]}
yolo = {"model_path": "best.pt", "imgsz": 1024, "conf": 0.3}

run_yolo(common=common, yolo=yolo)
```

## 可视化

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
- `*_vis.jpg` — 可视化结果
