import warnings

_SUPPRESSED_SUBSTRINGS = (
    "torch.Tensor.cuda and torch.nn.Module.cuda are replaced",
    "torch.jit.script and torch.jit.script_method will be disabled by transfer_to_npu",
    "owner does not match the current owner",
    "resource_tracker: There appear to be",
    "torch.distributed.reduce_op is deprecated",
)

_ORIGINAL_WARN = getattr(warnings, "_unify_original_warn", warnings.warn)
_ORIGINAL_SHOWWARNING = getattr(warnings, "_unify_original_showwarning", warnings.showwarning)


def _should_suppress(message):
    text = str(message)
    return any(item in text for item in _SUPPRESSED_SUBSTRINGS)


def _quiet_warn(message, category=None, stacklevel=1, source=None):
    if category is ResourceWarning or _should_suppress(message):
        return None
    return _ORIGINAL_WARN(message, category=category, stacklevel=stacklevel, source=source)


def _quiet_showwarning(message, category, filename, lineno, file=None, line=None):
    if category is ResourceWarning or _should_suppress(message):
        return None
    return _ORIGINAL_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


warnings._unify_original_warn = _ORIGINAL_WARN
warnings._unify_original_showwarning = _ORIGINAL_SHOWWARNING
warnings.warn = _quiet_warn
warnings.showwarning = _quiet_showwarning

warnings.filterwarnings("ignore", category=ResourceWarning)
