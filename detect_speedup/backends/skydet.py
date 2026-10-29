"""SKYDET（DINOv3-ConvNeXt + CrossFusedEncoder + DFINETransformer，D-FINE 系）。

对应权重示例：``/data1/wzh26/runs/skydet_gzda_v1_640_bs8/best_stg1.pth``
配置：``/data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml``

依赖：SKYDET 仓库（``/data1/wzh26/code/SKYDET``）与它自己的依赖目录
``PYTHONPATH=/home/wzh26/skydet_pkgs``；用 ``yolo`` conda 环境即可。

注意：SKYDET 自带 ``tools/inference/torch_inf.py`` 的预处理漏了训练时的
``dinov3_norm``（``/255`` + ImageNet 归一化），这里按训练口径补上。
"""

from __future__ import annotations

import sys
import time
from typing import List

import numpy as np
from PIL import Image

from . import register
from .base import TileBackend, TilePrediction, Window

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class SkydetBackend(TileBackend):
    name = "skydet"
    task = "detect"
    defaults = {
        "repo": "/data1/wzh26/code/SKYDET",
        "config": "/data1/wzh26/code/SKYDET/configs/skydet/skydet_gzda_v1_640_bs8.yml",
        "checkpoint": "/data1/wzh26/runs/skydet_gzda_v1_640_bs8/best_stg1.pth",
        "size": 640,
        "class_names": {0: "building"},
    }

    def __init__(self, config):
        super().__init__(config)
        self.class_names = dict(self.config.get("class_names") or {0: "building"})
        self.model = None

    def load(self, device: str) -> None:
        import torch
        import torch.nn as nn

        repo = self.config["repo"]
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from engine.core import YAMLConfig

        cfg_path = self.config["config"]
        ckpt_path = self.config["checkpoint"]
        t0 = time.time()
        cfg = YAMLConfig(cfg_path, resume=ckpt_path)
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_sizes):
                return self.postprocessor(self.model(images), orig_sizes)

        self._torch = torch
        self.model = Model().to(device).eval()
        print(f"[skydet] loaded in {time.time() - t0:.1f}s from {ckpt_path}", flush=True)

    def _preprocess(self, tiles: List[np.ndarray], size: int) -> "np.ndarray":
        batch = np.empty((len(tiles), 3, size, size), dtype=np.float32)
        for i, tile in enumerate(tiles):
            im = Image.fromarray(tile).resize((size, size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
            batch[i] = (arr - MEAN) / STD
        return batch

    def infer(self, tiles: List[np.ndarray], windows: List[Window], device: str) -> List[TilePrediction]:
        torch = self._torch
        size = int(self.config.get("size", 640))
        x = torch.from_numpy(self._preprocess(tiles, size)).to(device, non_blocking=True)
        orig = torch.tensor([[w[3] - w[1], w[2] - w[0]] for w in windows], device=device)
        with torch.no_grad():
            labels, boxes, scores = self.model(x, orig)
        out = []
        for lab, box, scr in zip(labels, boxes, scores):
            out.append(
                TilePrediction(
                    boxes=box.detach().cpu().numpy(),
                    scores=scr.detach().cpu().numpy(),
                    labels=lab.detach().cpu().numpy().astype(int),
                )
            )
        return out

    def close(self) -> None:
        self.model = None


register("skydet", SkydetBackend)
