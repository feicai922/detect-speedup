"""后端插件基类与数据结构。

新增一个后端只需要三步：
1. 在 ``detect_speedup/backends/`` 下新建 ``<name>.py``；
2. 继承 :class:`TileBackend`，实现 ``load()`` 与 ``infer()``；
3. 在文件末尾 ``register("name", MyBackend)``（或 ``register_aliases``）。

切块、组批、global NMS、JSON 落盘、Shapefile 导出、可视化都由
:mod:`detect_speedup.backends.runner` 统一完成，后端只需要"给一批瓦片，返回每块的框"。
可参考 ``backends/_template.py``。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# 瓦片用 (x1, y1, x2, y2) 表示，单位是源影像的像素
Window = Sequence[int]


@dataclass
class TilePrediction:
    """单个瓦片的推理结果，坐标是**瓦片内**像素坐标。"""

    boxes: np.ndarray                      # (N, 4) float, xyxy
    scores: np.ndarray                     # (N,) float
    labels: np.ndarray                     # (N,) int
    segment_xy: Optional[List[List[List[float]]]] = None  # 可选：(N, P, 2) 掩膜多边形
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        if not (len(self.boxes) == len(self.scores) == len(self.labels)):
            raise ValueError("boxes / scores / labels 数量不一致")

    def keep(self, mask: np.ndarray) -> "TilePrediction":
        seg = None
        if self.segment_xy is not None:
            seg = [s for s, k in zip(self.segment_xy, mask) if k]
        return TilePrediction(self.boxes[mask], self.scores[mask], self.labels[mask], seg)


class TileBackend(abc.ABC):
    """大图滑窗推理后端。

    子类只需关心"模型怎么加载"和"一批瓦片怎么变成框"，其余交给 runner。
    """

    #: CLI / 注册表里使用的名字
    name: str = "base"
    #: "detect" 或 "segment"（segment 时 JSON 里会带 segment_xy，可视化可画掩膜）
    task: str = "detect"
    #: 后端默认参数，CLI 未指定时使用
    defaults: Dict[str, Any] = {}
    #: 类别 id → 名称，写进 JSON 的 class_name
    class_names: Dict[int, str] = {0: "building"}

    def __init__(self, config: Dict[str, Any]):
        self.config = {**self.defaults, **(config or {})}

    # ---- 必须实现 ----------------------------------------------------
    @abc.abstractmethod
    def load(self, device: str) -> None:
        """加载权重到 ``device``，结果存到 ``self.model``。"""

    @abc.abstractmethod
    def infer(self, tiles: List[np.ndarray], windows: List[Window], device: str) -> List[TilePrediction]:
        """一批瓦片（HxWx3 uint8 RGB）→ 每个瓦片的 :class:`TilePrediction`。"""

    # ---- 可选 --------------------------------------------------------
    def close(self) -> None:
        """释放显存等资源。"""

    def describe(self) -> str:
        return f"{self.name} (task={self.task})"
