try:
    from setproctitle import setproctitle
except ImportError:
    setproctitle = None


def compact_device(device):
    text = str(device).replace(":", "")
    return text.replace("/", "_")


def set_process_title(title):
    if setproctitle is None:
        return
    try:
        setproctitle(str(title)[:31])
    except Exception:
        pass


def infer_title(backend, device=None, worker_id=None):
    parts = [str(backend), "infer"]
    if device is not None:
        parts.append(compact_device(device))
    if worker_id is not None:
        parts.append("w{}".format(worker_id))
    return "-".join(parts)


def read_title(backend, reader_id=None):
    parts = [str(backend), "read"]
    if reader_id is not None:
        parts.append("r{}".format(reader_id))
    return "-".join(parts)


def dir_title(backend, device=None, worker_id=None):
    parts = [str(backend), "dir"]
    if device is not None:
        parts.append(compact_device(device))
    if worker_id is not None:
        parts.append("w{}".format(worker_id))
    return "-".join(parts)
