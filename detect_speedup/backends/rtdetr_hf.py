"""HuggingFace transformers 的 RT-DETR（``RTDetrForObjectDetection``）。

对应权重示例：
``/data1/wzh26/重研/code/runs/down-rt-detr``（Yifeng-Liu/rt-detr-finetuned-for-satellite-image-roofs-detection）

依赖：``transformers``（本仓库的 ``rtdetr`` conda 环境已有）。
"""

from __future__ import annotations

import time
from typing import List

import numpy as np

from . import register
from .base import TileBackend, TilePrediction, Window


class RTDetrHFBackend(TileBackend):
    name = "rtdetr-hf"
    task = "detect"
    defaults = {
        "model_dir": None,
        "size": 640,          # 模型固定输入尺寸
        "class_names": {0: "roof"},
    }

    def __init__(self, config):
        super().__init__(config)
        self.class_names = dict(self.config.get("class_names") or {0: "building"})
        self.processor = None
        self.model = None

    def load(self, device: str) -> None:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        model_dir = self.config.get("model_dir")
        if not model_dir:
            raise ValueError("rtdetr-hf 需要 model_dir（HF 模型目录）")
        t0 = time.time()
        self.processor = RTDetrImageProcessor.from_pretrained(model_dir)
        self.model = RTDetrForObjectDetection.from_pretrained(model_dir).to(device).eval()
        self._torch = torch
        print(f"[rtdetr-hf] loaded in {time.time() - t0:.1f}s from {model_dir}", flush=True)

    def infer(self, tiles: List[np.ndarray], windows: List[Window], device: str) -> List[TilePrediction]:
        torch = self._torch
        inputs = self.processor(images=list(tiles), return_tensors="pt")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = [(w[3] - w[1], w[2] - w[0]) for w in windows]
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.01
        )
        out = []
        for res in results:
            out.append(
                TilePrediction(
                    boxes=res["boxes"].detach().cpu().numpy(),
                    scores=res["scores"].detach().cpu().numpy(),
                    labels=res["labels"].detach().cpu().numpy().astype(int),
                )
            )
        return out

    def close(self) -> None:
        self.model = None


register("rtdetr-hf", RTDetrHFBackend, aliases=("rtdetr_hf", "hf-rtdetr"))
