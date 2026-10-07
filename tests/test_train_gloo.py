"""Integration: real 2-process DDP training on CPU/Gloo with the full instrumentation path.

No GPU, no downloads. Validates plumbing and invariants, not performance.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from analysis.summary import summarize
from analysis.validation import validate

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, *sets):
    args = [sys.executable, "-m", "workloads.train", "--spawn", "2", "--out-root", str(tmp_path),
            "--set", "experiment.name=it", "--set", "workload.model=cnn_tiny",
            "--set", "workload.batch_size=8", "--set", "measurement.warmup_steps=2",
            "--set", "measurement.measured_steps=4", "--set", "measurement.repeats=2"]
    for s in sets:
        args += ["--set", s]
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "PYTHONPATH": str(ROOT)}
    r = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    files = sorted((tmp_path / "it").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_ddp_and_nosync_with_profiler(tmp_path):
    doc = _run(tmp_path, "measurement.profile.enabled=true", "measurement.profile.active=2")
    assert doc["schema_errors"] == []
    assert doc["backend"] == "gloo" and doc["world_size"] == 2
    # 2 repeats x 2 modes x 4 steps x 2 ranks
    assert len(doc["steps"]) == 32
    assert [b["mode"] for b in doc["blocks"]] == ["ddp", "nosync", "nosync", "ddp"]
    ddp = [s for s in doc["steps"] if s["mode"] == "ddp"]
    nos = [s for s in doc["steps"] if s["mode"] == "nosync"]
    assert {s["communication_bytes"] for s in ddp} == {doc["model"]["grad_bytes_fp32"]}
    assert all(s["communication_bytes"] == 0 for s in nos)
    assert {s["timing_source"] for s in doc["steps"]} == {"host_clock"}
    assert len(doc["profile"]) == 2 and all(p["hook_steps_during_profile"] for p in doc["profile"])
    checks = {c["name"]: c for c in validate([doc])}
    gates = [c for c in checks.values() if c["kind"] == "gate"]
    assert all(c["passed"] for c in gates), gates
    s = summarize([doc])
    assert s["ablation_exposed_ms"]["n"] == 2


def test_hook_disabled_and_synthetic_host_loader(tmp_path):
    doc = _run(tmp_path, "measurement.comm_hook=false", "workload.dataset=synthetic_host",
               "workload.num_workers=0")
    assert doc["schema_errors"] == []
    ddp = [s for s in doc["steps"] if s["mode"] == "ddp"]
    assert all(s["communication_time_ms"] is None for s in ddp)
    assert all(s["host_data_wait_ms"] >= 0 for s in doc["steps"])
    summarize([doc])  # must not crash on unmeasured communication


def test_gradient_accumulation(tmp_path):
    doc = _run(tmp_path, "workload.grad_accum_steps=2")
    ddp = [s for s in doc["steps"] if s["mode"] == "ddp"]
    assert all(s["accumulation_ms"] > 0 for s in ddp)
    # one gradient sync per optimizer step regardless of accumulation
    assert {s["communication_bytes"] for s in ddp} == {doc["model"]["grad_bytes_fp32"]}
