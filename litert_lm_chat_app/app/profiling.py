"""NVTX-range helper for nsys timeline labelling.

Wraps a code block in a named NVTX range so nsys-ui renders it as a labelled
band on the timeline. When nvtx isn't installed (the common case in
production), this falls back to a no-op contextmanager — zero overhead, zero
behaviour change.

Usage:
    from app.profiling import nvtx_range
    with nvtx_range("gemma_prefill"):
        ...

The category is optional and groups related ranges in nsys (e.g. all
"engine.*" ranges share a color).
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    import nvtx as _nvtx  # type: ignore
    _NVTX_AVAILABLE = True
except ImportError:
    _NVTX_AVAILABLE = False


@contextmanager
def nvtx_range(name: str, category: str = "nova"):
    if _NVTX_AVAILABLE:
        with _nvtx.annotate(message=name, category=category):
            yield
    else:
        yield
