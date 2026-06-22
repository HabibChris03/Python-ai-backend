"""Lightweight timing context manager used across all AI endpoints."""

import time
from contextlib import contextmanager
from typing import Generator


class _TimingState:
    def __init__(self):
        self.elapsed_ms: float = 0.0


@contextmanager
def timed() -> Generator[_TimingState, None, None]:
    """
    Usage:
        with timed() as t:
            do_work()
        print(t.elapsed_ms)  # milliseconds elapsed
    """
    state = _TimingState()
    start = time.perf_counter()
    try:
        yield state
    finally:
        state.elapsed_ms = (time.perf_counter() - start) * 1000.0
