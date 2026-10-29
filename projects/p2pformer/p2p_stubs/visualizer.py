"""Stub for the optional ``visualizer`` package (luo3300612/Visualizer).

P2PFormer's head decorates a method with ``@get_local('bboxes')`` purely to
cache tensors for its attention-map demo.  Inference does not need it, so we
provide a no-op decorator here instead of pulling in an extra dependency.
"""


def get_local(*args, **kwargs):
    def _deco(fn):
        return fn
    return _deco


def get_local_cuda(*args, **kwargs):
    return get_local(*args, **kwargs)
