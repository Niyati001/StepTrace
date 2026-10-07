import gzip
import json

import pytest

from instrument.profiler import analyze_trace, load_trace, merge, subtract


def test_merge_and_subtract():
    a = merge([(0, 10), (5, 12), (20, 30)])
    assert a == [(0, 12), (20, 30)]
    b = merge([(2, 4), (8, 25)])
    # a minus b: [0,2) + [4,8) + [25,30) = 2 + 4 + 5
    assert subtract(a, b) == 11
    assert subtract(a, []) == 22
    assert subtract([], b) == 0


def _k(name, ts, dur, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur}


def trace():
    ev = [
        {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#3", "ts": 0, "dur": 1000},
        {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#4", "ts": 1000, "dur": 1000},
        # step 3: compute 0-600, nccl 400-800 -> 200 us overlapped, 200 us exposed
        _k("volta_sgemm", 0, 600), _k("ncclDevKernel_AllReduce_Sum_f32_RING_LL", 400, 400),
        # step 4: compute 1000-1500, memcpy 1700-1750, nccl 1600-1800 -> exposed 150
        _k("conv", 1000, 500), _k("Memcpy DtoD", 1700, 50, cat="gpu_memcpy"),
        _k("ncclKernel_AllReduce", 1600, 200),
        {"ph": "X", "cat": "cpu_op", "name": "aten::add", "ts": 10, "dur": 5},
    ]
    return {"traceEvents": ev}


def test_analyze_trace_per_step_overlap():
    a = analyze_trace(trace())
    assert a["gpu_kernels_found"] and a["nccl_kernels_found"] and a["n_steps"] == 2
    s3, s4 = a["per_step"]
    assert s3["nccl_kernel_ms"] == pytest.approx(0.4) and s3["exposed_nccl_ms"] == pytest.approx(0.2)
    assert s4["nccl_kernel_ms"] == pytest.approx(0.2) and s4["exposed_nccl_ms"] == pytest.approx(0.15)
    assert s4["other_gpu_ms"] == pytest.approx(0.55)
    assert a["median"]["exposed_nccl_ms"] == pytest.approx(0.175)


def test_cpu_only_trace_reports_no_kernels():
    a = analyze_trace({"traceEvents": [{"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#0",
                                        "ts": 0, "dur": 10}]})
    assert not a["gpu_kernels_found"] and a["median"]["nccl_kernel_ms"] == 0


def test_load_gz(tmp_path):
    p = tmp_path / "t.json.gz"
    with gzip.open(p, "wt") as f:
        json.dump(trace(), f)
    assert analyze_trace(load_trace(p))["n_steps"] == 2
