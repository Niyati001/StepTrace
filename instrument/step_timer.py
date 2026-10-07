"""Step timing on a single device timeline.

CUDA execution is asynchronous, so host clocks around GPU work measure
*launch* time, not execution time. On CUDA every mark is a ``torch.cuda.Event``
recorded on the current stream; offsets are resolved with
``Event.elapsed_time`` only after ``torch.cuda.synchronize()`` at step end.
On CPU (Gloo development path) execution is synchronous and marks are
``time.perf_counter_ns`` values.
"""

from __future__ import annotations

import time

import torch


def host_clock_ns() -> int:
    """System-wide high-resolution clock comparable across processes on one host.

    Linux: CLOCK_MONOTONIC (shared by all processes). Elsewhere perf_counter_ns
    (QueryPerformanceCounter on Windows, also system-wide). Never compare
    across hosts.
    """
    if hasattr(time, "clock_gettime_ns") and hasattr(time, "CLOCK_MONOTONIC"):
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    return time.perf_counter_ns()


class Clock:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.cuda = device.type == "cuda"
        self.source = "cuda_event" if self.cuda else "host_clock"

    def mark(self):
        if self.cuda:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()  # current stream of the calling thread
            return ev
        return time.perf_counter_ns()

    def ms(self, start, end) -> float:
        if self.cuda:
            return start.elapsed_time(end)
        return (end - start) / 1e6

    def synchronize(self) -> None:
        if self.cuda:
            torch.cuda.synchronize(self.device)


class StepTimer:
    """Collect named marks for one step; resolve to ms offsets from the start mark."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.marks: dict = {}

    def begin(self) -> None:
        self.host_start_ns = time.perf_counter_ns()
        self.monotonic_start_ns = host_clock_ns()  # same-host cross-process alignment
        self.start = self.clock.mark()
        self.marks = {}

    def mark(self, name: str) -> None:
        self.marks[name] = self.clock.mark()

    def offsets(self) -> dict[str, float]:
        """Call only after Clock.synchronize()."""
        return {k: self.clock.ms(self.start, v) for k, v in self.marks.items()}
