"""Tracy profiler utilities for Python.

Provides a @tracy_zone decorator and helper functions for Tracy integration.
When Tracy is disabled at build time, all functions become no-ops.
"""

import functools
import inspect
import os

import alphazero


def tracy_zone(func):
    """Decorator to wrap a function in a Tracy zone.

    Zone name format: "module.Class.method" or "module.function"
    Source location: "filename:line"

    All metadata is computed at decoration time for zero overhead at call time.
    """
    # Pre-compute all metadata at decoration time
    name = func.__qualname__  # e.g., "GameRunner.batch_builder" or "self_play"
    try:
        filepath = inspect.getfile(func)
        filename = os.path.basename(filepath)  # Just "game_runner.py"
        _, line = inspect.getsourcelines(func)
    except (TypeError, OSError):
        filename = "<unknown>"
        line = 0

    source_loc = f"{filename}:{line}"

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        alphazero._tracy_zone_begin(name, source_loc, line)
        try:
            return func(*args, **kwargs)
        finally:
            alphazero._tracy_zone_end()

    return wrapper


class TracyZone:
    """Context manager for inline Tracy zones.

    Usage:
        with TracyZone("my_operation"):
            # code to profile
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        alphazero._tracy_zone_begin(self.name, "", 0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        alphazero._tracy_zone_end()
        return False


def tracy_thread(name: str):
    """Set the current thread's name in Tracy for better visualization.

    Call this at the start of each thread's main function.
    """
    alphazero._tracy_set_thread_name(name)


def tracy_frame():
    """Mark a frame boundary (e.g., per training iteration).

    Call this at the end of each training iteration to help Tracy
    group work into frames for analysis.
    """
    alphazero.tracy_frame_mark()
