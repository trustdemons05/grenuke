"""Run a training step that writes its result to the cache.

Without a GPU lock file a step runs in this process. With one (`--gpu-lock FILE`), every GPU step
runs in a fresh process while this process holds an exclusive `flock` on FILE, so other jobs that
use the same lock never share the GPU with it, and the GPU memory XGBoost and CatBoost keep until
their process exits is released before the lock is.
"""

from __future__ import annotations

import contextlib
import importlib
import os


@contextlib.contextmanager
def gpu_lock(path: str):
    import fcntl  # Unix only

    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def call(target: str, kwargs: dict) -> None:
    """`target` is `"module:function"`; the function writes its own result."""
    module, name = target.split(":")
    getattr(importlib.import_module(module), name)(**kwargs)


def run(target: str, kwargs: dict, *, device: str, lock: str | None) -> None:
    if device != "cuda" or not lock:
        call(target, kwargs)
        return
    import multiprocessing as mp

    with gpu_lock(lock):
        child = mp.get_context("spawn").Process(target=call, args=(target, kwargs))
        child.start()
        child.join()
    if child.exitcode != 0:
        raise RuntimeError(f"{target} failed (exit code {child.exitcode})")


def save(path, array) -> None:
    """Atomic `np.save`: a reader never sees a half-written file."""
    import numpy as np

    path = str(path)
    tmp = f"{path}.tmp{os.getpid()}.npy"
    np.save(tmp, array)
    os.replace(tmp, path)
