"""torch.profiler cross-check of the hook/event measurements.

The profiler (Kineto/CUPTI) timestamps every GPU kernel. From a Chrome trace
we compute, per ``ProfilerStep#N`` window:

* ``nccl_kernel_ms``  - union of NCCL kernel intervals
* ``other_gpu_ms``    - union of all non-NCCL kernels + memcpy/memset
* ``exposed_nccl_ms`` - NCCL time not overlapped by any non-NCCL GPU activity
* ``step_ms``         - CPU span of the ProfilerStep (includes profiler overhead)

This is an independent method: it uses kernel timestamps, not our CUDA events.
Expected, documented differences vs. the hook measurement:
* NCCL kernels start when launched and may spin waiting for the peer rank, so
  kernel time >= pure transfer time; the hook's derived start excludes queueing
  but includes the same peer wait.
* "Not overlapped by any kernel" counts compute-stream *gaps* (e.g. launch-bound
  stretches) as exposed even if communication did not cause them.
The trace analysis functions are pure and tested on CPU with fixtures.
"""

from __future__ import annotations

import gzip
import json
import statistics
from pathlib import Path

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def load_trace(path: str | Path) -> dict:
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as f:
        return json.load(f)


def merge(iv: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def length(merged: list[tuple[float, float]]) -> float:
    return sum(e - s for s, e in merged)


def subtract(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """Total length of merged intervals ``a`` not covered by merged intervals ``b``."""
    total, j = 0.0, 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                total += b[k][0] - cur
            cur = max(cur, b[k][1])
            if cur >= e:
                break
            k += 1
        if cur < e:
            total += e - cur
    return total


def _clip(iv, lo, hi):
    return [(max(s, lo), min(e, hi)) for s, e in iv if e > lo and s < hi]


def analyze_trace(trace: dict) -> dict:
    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    gpu = [e for e in events if e.get("cat") in GPU_CATS]
    nccl = [(e["ts"], e["ts"] + e["dur"]) for e in gpu if "nccl" in e.get("name", "").lower()]
    other = [(e["ts"], e["ts"] + e["dur"]) for e in gpu if "nccl" not in e.get("name", "").lower()]
    steps = sorted((e for e in events if e.get("name", "").startswith("ProfilerStep#")),
                   key=lambda e: e["ts"])
    windows = [(e["ts"], e["ts"] + e["dur"]) for e in steps]
    if not windows and gpu:  # no annotations: treat the whole trace as one window
        windows = [(min(e["ts"] for e in gpu), max(e["ts"] + e["dur"] for e in gpu))]

    per_step = []
    for lo, hi in windows:
        n, o = merge(_clip(nccl, lo, hi)), merge(_clip(other, lo, hi))
        n_ms = length(n) / 1e3
        exp_ms = subtract(n, o) / 1e3
        per_step.append({
            "step_ms": (hi - lo) / 1e3,
            "nccl_kernel_ms": n_ms,
            "other_gpu_ms": length(o) / 1e3,
            "exposed_nccl_ms": exp_ms,
            "overlapped_nccl_ms": n_ms - exp_ms,
        })
    med = {k: statistics.median(s[k] for s in per_step) for k in per_step[0]} if per_step else {}
    return {
        "gpu_kernels_found": bool(gpu),
        "nccl_kernels_found": bool(nccl),
        "n_steps": len(per_step),
        "nccl_kernel_names": sorted({e["name"] for e in gpu if "nccl" in e.get("name", "").lower()})[:10],
        "per_step": per_step,
        "median": med,
    }


def make_profiler(device_is_cuda: bool, wait: int, warmup: int, active: int, trace_path: Path):
    import torch.profiler as tp

    acts = [tp.ProfilerActivity.CPU] + ([tp.ProfilerActivity.CUDA] if device_is_cuda else [])
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return tp.profile(
        activities=acts,
        schedule=tp.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
        record_shapes=False, with_stack=False,
    )


def gzip_file(path: Path) -> Path:
    gz = path.with_suffix(path.suffix + ".gz")
    with open(path, "rb") as src, gzip.open(gz, "wb") as dst:
        dst.write(src.read())
    path.unlink()
    return gz
