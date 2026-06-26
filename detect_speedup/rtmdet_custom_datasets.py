from . import runtime_warning_filters  # noqa: F401

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module(force=True)
class GreenhouseDataset(CocoDataset):
    METAINFO = {
        "classes": ("greenhouse",),
        "palette": [(0, 255, 0)],
    }


@DATASETS.register_module(force=True)
class PowerPoleDetectDataset(CocoDataset):
    METAINFO = {
        "classes": ("powepole",),
        "palette": [(0, 255, 0)],
    }
