"""**新增后端模板** —— 复制这个文件改名后填空即可。

步骤
----
1. 复制成 ``backends/myx.py``；
2. 改 ``name`` / ``task`` / ``defaults`` / ``class_names``；
3. 填 ``load()``（把权重加载到 ``self.model``）与 ``infer()``（一批瓦片 → 每块框）；
4. 文件末尾 ``register("myx", MyBackend)``；
5. 在 ``backends/__init__.py`` 末尾加 ``from . import myx  # noqa``；
6. ``python run_large.py --list-backends`` 能看到 ``myx`` 就成功了。

约定
----
* ``tiles`` 是 ``list[np.ndarray]``，每个 ``HxWx3`` uint8 RGB（边缘瓦片可能小于 tile_size）；
* ``windows`` 与之等长，元素 ``(x1, y1, x2, y2)``，单位是**源影像像素**；
* ``infer()`` 返回的框坐标必须是**瓦片内**像素坐标，runner 会自己加回 window 偏移；
* 置信度阈值、global NMS、JSON 落盘、SHP 导出、可视化都由 runner 负责，
  所以 ``infer()`` 里不要自己过滤分数（返回全量即可，runner 按 ``--conf`` 过滤）。
"""

from __future__ import annotations

from typing import List

import numpy as np

from . import register
from .base import TileBackend, TilePrediction, Window


class MyBackend(TileBackend):
    name = "myx"
    task = "detect"          # 有掩膜就改成 "segment"，并在 infer() 里填 segment_xy
    defaults = {
        "checkpoint": None,
        "size": 640,
        "class_names": {0: "building"},
    }

    def __init__(self, config):
        super().__init__(config)
        self.class_names = dict(self.config.get("class_names") or {0: "building"})
        self.model = None

    def load(self, device: str) -> None:
        import torch  # noqa: F401
        
        # TODO: 加载权重
        # self.model = build_my_model(self.config["checkpoint"]).to(device).eval()
        raise NotImplementedError

    def infer(self, tiles: List[np.ndarray], windows: List[Window], device: str) -> List[TilePrediction]:
        # TODO: 前向 + 后处理，返回每个瓦片的 TilePrediction（瓦片内坐标）
        # 有掩膜时：
        #   TilePrediction(boxes, scores, labels, segment_xy=[[[x, y], ...], ...])
        raise NotImplementedError

    def close(self) -> None:
        self.model = None


# register("myx", MyBackend)
