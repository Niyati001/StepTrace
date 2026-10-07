"""Measurement-validity checks (pre-registered; METHODOLOGY.md §Validation).

``gate`` checks must pass before GPU budget is spent on the pilot/baseline.
``info`` checks are reported, never used to accept/reject (their definitions
legitimately differ; see METHODOLOGY.md).
"""

from __future__ import annotations

import math

import numpy as np

from analysis.summary import profile_cross_check

PROFILER_COMM_REL_TOL = 0.20     # hook comm busy vs profiler NCCL kernel time
PROFILER_COMM_ABS_TOL_MS = 0.5
WALL_EVENT_REL_TOL = 0.05        # host wall step vs CUDA-event step
WALL_EVENT_ABS_TOL_MS = 1.0
HOOK_OVERHEAD_REL_TOL = 0.03     # timed hook vs DDP built-in all-reduce


def _check(name, passed, kind, value, criterion):
    return {"name": name, "passed": bool(passed), "kind": kind, "value": value, "criterion": criterion}


def _ddp_steps(docs):
    return [s for d in docs for s in d["steps"] if s["mode"] == "ddp"]


def validate(docs: list[dict], hook_off_docs: list[dict] | None = None) -> list[dict]:
    out = []
    errs = sum(len(d.get("schema_errors", [])) for d in docs + (hook_off_docs or []))
    out.append(_check("schema_valid", errs == 0, "gate", errs, "0 schema errors"))

    steps = _ddp_steps(docs)
    timed = [s for s in steps if s.get("communication_time_ms") is not None]
    gpu = any(s["timing_source"] == "cuda_event" for s in steps)

    if timed:
        resid = max(abs(s["data_wait_ms"] + s["compute_time_ms"] + s["exposed_communication_time_ms"]
                        + s["ddp_finalize_ms"] - s["step_time_ms"]) for s in timed)
        out.append(_check("decomposition_closes", resid < 1e-3, "gate", resid,
                          "data_wait + compute + exposed + finalize == step (|resid| < 1e-3 ms)"))
        viol = sum(s["comm_causality_violations"] for s in timed)
        out.append(_check("comm_causality", viol == 0, "gate", viol,
                          "no collective ends before its bucket is ready"))
        expected = {d["model"]["grad_bytes_fp32"] for d in docs}
        got = {s["communication_bytes"] for s in timed}
        out.append(_check("comm_bytes_equal_grad_bytes", got == expected, "gate",
                          {"hook": sorted(got), "model": sorted(expected)},
                          "hook bytes per step == fp32 gradient bytes of the model"))
        buckets = {s["bucket_count"] for s in timed}
        out.append(_check("bucket_count_stable", len(buckets) == 1, "info", sorted(buckets),
                          "same bucket count every measured step"))

    diffs = np.array([s["wall_step_ms"] - s["step_time_ms"] for s in steps])
    med_step = float(np.median([s["step_time_ms"] for s in steps]))
    lim = max(WALL_EVENT_ABS_TOL_MS, WALL_EVENT_REL_TOL * med_step)
    med_diff = float(np.median(diffs))
    out.append(_check("event_vs_wall_step", -0.05 <= med_diff <= lim, "gate",
                      {"median_wall_minus_event_ms": med_diff, "median_step_ms": med_step},
                      f"0 <= median(wall - event) <= {lim:.3f} ms"))
    out.append(_check("timing_source", gpu or not any(d["backend"] == "nccl" for d in docs), "gate",
                      sorted({s["timing_source"] for s in steps}), "CUDA events on GPU runs"))
    out.append(_check("loss_finite", all(math.isfinite(s["loss"]) for d in docs for s in d["steps"]),
                      "gate", None, "every recorded loss finite"))

    for r in profile_cross_check(docs):
        if not r["gpu_kernels_found"]:
            out.append(_check(f"profiler_rank{r['rank']}_kernels", not gpu, "gate" if gpu else "info",
                              None, "GPU kernels present in profiler trace (GPU runs)"))
            continue
        tol = max(PROFILER_COMM_REL_TOL * r["prof_nccl_kernel_ms"], PROFILER_COMM_ABS_TOL_MS)
        d = r["hook_comm_busy_ms"] - r["prof_nccl_kernel_ms"]
        out.append(_check(f"profiler_rank{r['rank']}_comm_busy_agrees", abs(d) <= tol, "gate",
                          {"hook_ms": r["hook_comm_busy_ms"], "nccl_kernel_ms": r["prof_nccl_kernel_ms"],
                           "diff_ms": d},
                          f"|hook busy - NCCL kernel time| <= {tol:.3f} ms"))
        out.append(_check(f"profiler_rank{r['rank']}_exposed", True, "info",
                          {"hook_exposed_ms": r["hook_exposed_ms"],
                           "prof_exposed_nccl_ms": r["prof_exposed_nccl_ms"],
                           "hook_step_ms": r["hook_step_ms"], "prof_step_ms": r["prof_step_ms"],
                           "prof_other_gpu_ms": r["prof_other_gpu_ms"]},
                          "reported only (definitions differ, see METHODOLOGY.md)"))

    nccl = [d.get("nccl") for d in docs if d.get("backend") == "nccl"]
    if nccl:
        tr = sorted({t for n in nccl if n for t in n["transports"]})
        ver = sorted({n["runtime_version"] for n in nccl if n and n["runtime_version"]})
        out.append(_check("nccl_transport_recorded", bool(tr), "gate", {"transports": tr, "versions": ver},
                          "NCCL-reported transport captured for every GPU run"))

    if hook_off_docs:
        def launch_medians(ds):
            return [float(np.median([s["step_time_ms"] for s in _ddp_steps([d])])) for d in ds]
        on, off = launch_medians(docs), launch_medians(hook_off_docs)
        rel = (np.median(on) - np.median(off)) / np.median(off)
        noise = float(np.hypot(np.std(on, ddof=1) if len(on) > 1 else 0.0,
                               np.std(off, ddof=1) if len(off) > 1 else 0.0) / np.median(off))
        # A gate must be able to fail: it passes only if the comparison is precise enough
        # (noise <= tolerance) AND the difference is within tolerance. CPU timings: info only.
        precise = noise <= HOOK_OVERHEAD_REL_TOL
        out.append(_check("hook_overhead", precise and abs(rel) <= HOOK_OVERHEAD_REL_TOL,
                          "gate" if gpu else "info",
                          {"hook_on_launch_medians_ms": on, "hook_off_launch_medians_ms": off,
                           "relative_diff": float(rel), "relative_noise": noise},
                          f"launch noise <= {HOOK_OVERHEAD_REL_TOL} and "
                          f"|median(on) - median(off)| / median(off) <= {HOOK_OVERHEAD_REL_TOL}"))
    return out


def format_checks(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        tag = "PASS" if c["passed"] else ("FAIL" if c["kind"] == "gate" else "note")
        lines.append(f"[{tag}] ({c['kind']}) {c['name']}: {c['value']}  -- {c['criterion']}")
    ok = all(c["passed"] for c in checks if c["kind"] == "gate")
    lines.append(f"GATES: {'ALL PASS' if ok else 'FAILED'}")
    return "\n".join(lines)
