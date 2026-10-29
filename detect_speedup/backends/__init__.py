"""后端注册表。

内置的 ``yolo / rtmdet / rfdetr`` 走框架原有的多进程多卡流水线（``NATIVE_BACKENDS``），
其余后端（如 ``rtdetr-hf`` / ``skydet``）走 :mod:`detect_speedup.backends.runner` 的
通用滑窗循环（``PLUGIN_BACKENDS``）。

新增插件的写法见 ``backends/_template.py``；只要模块被 import 时调用了 ``register()``，
CLI 的 ``--backend`` 就能直接用，无需改动别处。
"""

from __future__ import annotations

from typing import Callable, Dict, List, Type

from .base import TileBackend, TilePrediction, Window

#: 框架原生后端（多进程/多卡优化过的老流水线，``--backend`` 直接透传）
NATIVE_BACKENDS: List[str] = ["yolo", "rtmdet", "rfdetr"]
NATIVE_ALIASES = {"mmdet": "rtmdet", "rf-detr": "rfdetr"}

_REGISTRY: Dict[str, Type[TileBackend]] = {}


def register(name: str, backend_cls: Type[TileBackend], aliases: tuple = ()) -> Type[TileBackend]:
    """把后端类登记到注册表（可同时登记别名）。"""
    for key in (name, *aliases):
        _REGISTRY[key.lower()] = backend_cls
    return backend_cls


def get_backend(name: str, config: Dict) -> TileBackend:
    """按名字实例化一个插件后端。"""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"未知后端 {name!r}；可用插件后端: {sorted(_REGISTRY)}；"
            f"原生后端: {NATIVE_BACKENDS}"
        )
    return _REGISTRY[key](config)


def is_native(name: str) -> bool:
    key = name.lower()
    if key in NATIVE_ALIASES:
        key = NATIVE_ALIASES[key]
    return key in NATIVE_BACKENDS


def normalize_native(name: str) -> str:
    key = name.lower()
    return NATIVE_ALIASES.get(key, key)


def plugin_names() -> List[str]:
    return sorted(_REGISTRY)


def all_backends() -> List[str]:
    return NATIVE_BACKENDS + plugin_names()


# 导入即注册内置插件后端（新增后端时在这里加一行 import）
from . import rtdetr_hf  # noqa: E402,F401
from . import skydet  # noqa: E402,F401

__all__ = [
    "NATIVE_BACKENDS",
    "TileBackend",
    "TilePrediction",
    "Window",
    "all_backends",
    "get_backend",
    "is_native",
    "normalize_native",
    "plugin_names",
    "register",
]
