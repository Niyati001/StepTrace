"""Turn raw run documents into processed metrics (pure numpy; CPU-testable).

Levels:
  step   (run, block, step, rank)        raw record
  cstep  (run, block, step)              cluster step: ranks combined
  block  (run, block)                    median/IQR over measured steps
  mode   (experiment, mode)              statistics over block medians (= repeats)

Cluster step: step_time = max over ranks (the slowest rank bounds a
synchronous step); component times = mean over ranks; skews = max - min.

Exposed communication is reported two ways (METHODOLOGY.md):
  * timeline  - per-step hook/event measurement (exposed_communication_time_ms)
  * ablation  - paired difference of block medians, ddp - nosync, same repeat
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

COMPONENTS = ("step_time_ms", "data_wait_ms", "compute_time_ms", "forward_ms",
              "backward_compute_ms", "optimizer_ms", "communication_time_ms",
              "exposed_communication_time_ms", "ddp_finalize_ms", "host_data_wait_ms",
              "wall_step_ms", "communication_bytes", "bucket_count",
              "gpu_utilization_pct", "gpu_memory_mb", "comm_causality_violations")
OUTLIER_IQR_K = 3.0  # flag steps > Q3 + 3*IQR within a block (reported, never dropped)


def _q(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q))


def describe(values) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p10": _q(a, 10), "p25": _q(a, 25), "p75": _q(a, 75), "p90": _q(a, 90),
            "iqr": _q(a, 75) - _q(a, 25), "min": float(a.min()), "max": float(a.max())}


def cluster_steps(steps: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in steps:
        groups[(s["run_id"], s["block_id"], s["step"])].append(s)
    out = []
    for (run_id, block_id, step), rs in sorted(groups.items()):
        row = {"run_id": run_id, "block_id": block_id, "step": step,
               "repeat": rs[0]["repeat"], "mode": rs[0]["mode"], "n_ranks": len(rs)}
        for k in COMPONENTS:
            vals = [r.get(k) for r in rs if r.get(k) is not None]
            row[k] = float(np.mean(vals)) if vals else None
        st = [r["step_time_ms"] for r in rs]
        row["step_time_ms"] = max(st)
        row["rank_step_skew_ms"] = max(st) - min(st)
        cp = [r["compute_time_ms"] for r in rs if r.get("compute_time_ms") is not None]
        row["rank_compute_skew_ms"] = (max(cp) - min(cp)) if cp else None
        ts = [r["step_start_monotonic_ns"] for r in rs]
        row["rank_start_skew_ms"] = (max(ts) - min(ts)) / 1e6
        out.append(row)
    return out


def block_table(csteps: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in csteps:
        groups[(r["run_id"], r["block_id"])].append(r)
    out = []
    for (run_id, block_id), rows in sorted(groups.items()):
        st = np.array([r["step_time_ms"] for r in rows])
        q1, q3 = np.percentile(st, [25, 75])
        b = {"run_id": run_id, "block_id": block_id, "repeat": rows[0]["repeat"],
             "mode": rows[0]["mode"], "n_steps": len(rows),
             "outlier_steps": int((st > q3 + OUTLIER_IQR_K * (q3 - q1)).sum()),
             "step_time_iqr_ms": float(q3 - q1)}
        for k in COMPONENTS + ("rank_step_skew_ms", "rank_compute_skew_ms", "rank_start_skew_ms"):
            vals = [r[k] for r in rows if r.get(k) is not None]
            b[k] = float(np.median(vals)) if vals else None
        out.append(b)
    return out


def summarize(docs: list[dict]) -> dict:
    steps = [s for d in docs for s in d["steps"]]
    cs = cluster_steps(steps)
    blocks = block_table(cs)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for b in blocks:
        by_mode[b["mode"]].append(b)

    modes = {}
    for mode, bs in by_mode.items():
        modes[mode] = {k: describe([b[k] for b in bs]) for k in
                       COMPONENTS + ("rank_step_skew_ms", "rank_compute_skew_ms", "rank_start_skew_ms")}
        modes[mode]["n_blocks"] = len(bs)
        modes[mode]["outlier_steps"] = sum(b["outlier_steps"] for b in bs)
        modes[mode]["n_steps"] = sum(b["n_steps"] for b in bs)

    # Paired ablation: same launch and repeat index, ddp vs nosync.
    pair = defaultdict(dict)
    for b in blocks:
        pair[(b["run_id"], b["repeat"])][b["mode"]] = b
    diffs, comp_diffs = [], []
    for p in pair.values():
        if "ddp" in p and "nosync" in p:
            diffs.append(p["ddp"]["step_time_ms"] - p["nosync"]["step_time_ms"])
            if p["ddp"]["compute_time_ms"] is not None:
                comp_diffs.append(p["ddp"]["compute_time_ms"] - p["nosync"]["compute_time_ms"])

    out = {"n_runs": len(docs), "n_step_records": len(steps), "modes": modes, "blocks": blocks}
    if "ddp" in modes:
        d = modes["ddp"]
        step = d["step_time_ms"]["median"]

        def frac(k):
            v = d[k].get("median")
            return None if v is None else v / step

        out["ddp_fractions"] = {
            "communication_busy": frac("communication_time_ms"),
            "exposed_communication_timeline": frac("exposed_communication_time_ms"),
            "data_wait": frac("data_wait_ms"),
            "compute": frac("compute_time_ms"),
            "ddp_finalize": frac("ddp_finalize_ms"),
        }
    if diffs:
        abl = describe(diffs)
        abl["all_positive"] = bool(all(x > 0 for x in diffs))
        abl["values"] = diffs
        abl["fraction_of_ddp_step"] = abl["median"] / modes["ddp"]["step_time_ms"]["median"]
        out["ablation_exposed_ms"] = abl
        out["compute_interference_ms"] = describe(comp_diffs)  # compute slowdown under concurrent comm
    out["measurability"] = measurability(out)
    out["profile_cross_check"] = profile_cross_check(docs)
    out["gpu_state"] = gpu_state_summary(docs)
    out["nccl"] = sorted({(t, (d.get("nccl") or {}).get("runtime_version")) for d in docs
                          for t in (d.get("nccl") or {}).get("transports", [])})
    return out


def gpu_state_summary(docs: list[dict]) -> dict:
    """Range of SM clock / temperature / power over block boundaries, plus throttle reasons seen."""
    snaps = [x[k] for d in docs for x in d.get("gpu_state", []) for k in ("before", "after") if x.get(k)]
    if not snaps:
        return {"available": False}

    def rng(key):
        v = [s[key] for s in snaps if s.get(key) is not None]
        return [min(v), max(v)] if v else None

    return {"available": True, "n_snapshots": len(snaps), "sm_clock_mhz": rng("sm_clock_mhz"),
            "temperature_c": rng("temperature_c"), "power_w": rng("power_w"),
            "clock_event_reasons": sorted({s.get("clock_event_reasons") for s in snaps} - {None})}


def measurability(summary: dict, k_noise: float = 3.0, min_fraction: float = 0.10) -> dict:
    """Pre-registered pilot criterion (METHODOLOGY.md §Pilot):

    resolvable: every paired (ddp - nosync) difference > 0 AND
                median difference > k_noise * sqrt(sd_ddp^2 + sd_nosync^2),
                sd = std of block medians across repeats (needs >= 2 repeats);
    material:   median ablation exposure >= min_fraction of the ddp step.
    """
    abl = summary.get("ablation_exposed_ms")
    m = summary["modes"]
    if not abl or abl["n"] < 2 or "nosync" not in m:
        return {"status": "insufficient_repeats", "resolvable": None, "material": None}
    noise = float(np.hypot(m["ddp"]["step_time_ms"]["std"], m["nosync"]["step_time_ms"]["std"]))
    resolvable = abl["all_positive"] and abl["median"] > k_noise * noise
    material = abl["fraction_of_ddp_step"] >= min_fraction
    return {"status": "ok", "noise_ms": noise, "k_noise": k_noise, "min_fraction": min_fraction,
            "resolvable": bool(resolvable), "material": bool(material),
            "communication_relevant": bool(resolvable and material)}


def profile_cross_check(docs: list[dict]) -> list[dict]:
    """Per rank: profiler kernel-trace metrics vs hook metrics on the SAME profiled steps."""
    rows = []
    for d in docs:
        for p in d.get("profile") or []:
            if not p:
                continue
            hs = p["hook_steps_during_profile"]
            med = p["trace_analysis"].get("median", {})
            rows.append({
                "run_id": d["run_id"], "rank": p["rank"],
                "gpu_kernels_found": p["trace_analysis"]["gpu_kernels_found"],
                "n_steps": len(hs),
                "hook_step_ms": float(np.median([h["step_time_ms"] for h in hs])),
                "hook_comm_busy_ms": float(np.median([h["communication_time_ms"] for h in hs])),
                "hook_exposed_ms": float(np.median([h["exposed_communication_time_ms"] for h in hs])),
                "prof_step_ms": med.get("step_ms"),
                "prof_nccl_kernel_ms": med.get("nccl_kernel_ms"),
                "prof_exposed_nccl_ms": med.get("exposed_nccl_ms"),
                "prof_other_gpu_ms": med.get("other_gpu_ms"),
            })
    return rows
