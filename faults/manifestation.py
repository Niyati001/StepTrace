"""Mechanism-level manifestation criteria (independent of the diagnoser).

MANIFESTATION asks whether an injected mechanism acted on the system as intended.
It never uses the diagnoser's verdict, its robust-z scores or its thresholds. Each
check compares a mechanism-specific observable with the injected amount (from ground
truth) or with the same-session healthy reference.

A run is DID_NOT_MANIFEST only if BOTH
  (a) its mechanism check fails, AND
  (b) its median cluster step time lies inside the healthy reference range of block
      medians (no measurable performance effect at all).
If the step time moved but the check failed, the run counts as MANIFESTED (unexpected
pathway) and its diagnosis is scored. A slowed-down run that was misdiagnosed can never
be excused as "did not manifest". This file is hashed into the frozen rule record.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from analysis.summary import cluster_steps

MIN_FRACTION_OF_INTENDED = 0.5   # a check passes if >= 50 % of the injected amount is observed


def _ddp(steps):
    return [s for s in steps if s["mode"] == "ddp"]


def _median(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else float("nan")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def reference_profile(ref_docs: list[dict]) -> dict:
    """Healthy, same-session quantities the checks compare against."""
    steps = [s for d in ref_docs for s in _ddp(d["steps"])]
    cs = [c for d in ref_docs for c in cluster_steps(d["steps"]) if c["mode"] == "ddp"]
    block_steps = defaultdict(list)
    for c in cs:
        block_steps[(c["run_id"], c["block_id"])].append(c["step_time_ms"])
    medians = [float(np.median(v)) for v in block_steps.values()]
    # consumer inter-batch period: consecutive step starts within a block, per rank
    cycles = []
    by = defaultdict(list)
    for s in steps:
        by[(s["run_id"], s["block_id"], s["rank"])].append((s["step"], s["step_start_monotonic_ns"]))
    for seq in by.values():
        seq.sort()
        cycles += [(b[1] - a[1]) / 1e6 for a, b in zip(seq, seq[1:]) if b[0] == a[0] + 1]
    transports = sorted({t for d in ref_docs for t in (d.get("nccl") or {}).get("transports", [])})
    return {"step_block_min": min(medians), "step_block_max": max(medians),
            "step_median": _median(c["step_time_ms"] for c in cs),
            "cycle_median_ms": _median(cycles),
            "comm_busy_ms": _median(s["communication_time_ms"] for s in steps),
            "host_data_wait_ms": _median(s["host_data_wait_ms"] for s in steps),
            "host_data_wait_mean_ms": _mean(s["host_data_wait_ms"] for s in steps),
            "bucket_counts": sorted({s["bucket_count"] for s in steps}),
            "comm_bytes": sorted({s["communication_bytes"] for s in steps}),
            "transports": transports}


def _rank_compute_gap(steps, target_rank):
    by = defaultdict(dict)
    for s in steps:
        by[(s["block_id"], s["step"])][s["rank"]] = s["compute_time_ms"]
    gaps = [v[target_rank] - np.mean([c for r, c in v.items() if r != target_rank])
            for v in by.values() if target_rank in v and len(v) > 1]
    return _median(gaps)


def check(doc: dict, ref: dict, host_ref_data_wait_ms: float | None = None,
          host_ref: dict | None = None) -> dict:
    """host_ref: reference_profile of the same-session healthy host-loader runs (used by the
    rate-limited worker stalls; falls back to `ref` if absent)."""
    gt = doc["ground_truth"]
    mech, p = gt["fault_mechanism"], gt["fault_parameters"]
    steps = _ddp(doc["steps"])
    cs = [c for c in cluster_steps(doc["steps"]) if c["mode"] == "ddp"]
    step_med = _median(c["step_time_ms"] for c in cs)
    perf_effect = not (ref["step_block_min"] <= step_med <= ref["step_block_max"])
    ev: dict = {"step_median_ms": step_med, "healthy_step_block_range": [ref["step_block_min"],
                                                                         ref["step_block_max"]]}
    hdw_base = ref["host_data_wait_ms"] if host_ref_data_wait_ms is None else host_ref_data_wait_ms

    if mech == "none":
        ok = True
    elif mech == "single_bucket":
        obs = sorted({s["bucket_count"] for s in steps})
        ev.update(bucket_counts=obs, healthy_bucket_counts=ref["bucket_counts"])
        ok = obs != ref["bucket_counts"]
    elif mech == "small_batch":
        eff = doc["effective_config"]["workload"]["batch_size"]
        req = doc["config"]["workload"]["batch_size"]
        nbytes = sorted({s["communication_bytes"] for s in steps})
        ev.update(batch_requested=req, batch_effective=eff, comm_bytes=nbytes, healthy_comm_bytes=ref["comm_bytes"])
        ok = eff < req and nbytes == ref["comm_bytes"]
    elif mech == "emulated_bandwidth":
        inj = _median(s.get("injected_comm_delay_ms") for s in steps)
        d_busy = _median(s["communication_time_ms"] for s in steps) - ref["comm_busy_ms"]
        ev.update(injected_ms=inj, comm_busy_increase_ms=d_busy)
        ok = inj == inj and inj > 0 and d_busy >= MIN_FRACTION_OF_INTENDED * inj
    elif mech == "shm_disable":
        tr = sorted((doc.get("nccl") or {}).get("transports", []))
        ev.update(transports=tr, healthy_transports=ref["transports"])
        ok = bool(tr) and tr != ref["transports"]
    elif mech in ("sleep", "compute"):
        gap = _rank_compute_gap(steps, p["rank"])
        ev.update(target_rank_compute_excess_ms=gap, intended_ms=p["delay_ms"])
        ok = gap >= MIN_FRACTION_OF_INTENDED * p["delay_ms"]
    elif mech == "fetch_sleep":
        d = _median(s["host_data_wait_ms"] for s in steps) - hdw_base
        ev.update(host_data_wait_increase_ms=d, intended_ms=p["delay_ms"])
        ok = d >= MIN_FRACTION_OF_INTENDED * p["delay_ms"]
    elif mech in ("loader_sleep", "cpu_preprocess"):
        # Workers deliver in pairs, so per-step waits are bimodal: compare the MEAN over all steps
        # and ranks (not the median) with the rate-based expectation d/W - healthy host-loader cycle.
        w = doc["effective_config"]["workload"]["num_workers"]
        base = host_ref or ref
        cycle = base["cycle_median_ms"]
        intended = max(0.0, p["delay_ms"] / max(w, 1) - cycle)
        d = _mean(s["host_data_wait_ms"] for s in steps) - base["host_data_wait_mean_ms"]
        ev.update(host_data_wait_increase_ms=d, intended_ms=intended, consumer_cycle_ms=cycle)
        ok = intended > 0 and d >= MIN_FRACTION_OF_INTENDED * intended
    else:
        raise ValueError(f"no manifestation criterion for mechanism {mech!r}")

    if ok:
        status = "MANIFESTED"
    elif perf_effect:
        status = "MANIFESTED_UNEXPECTED_PATHWAY"
    else:
        status = "DID_NOT_MANIFEST"
    return {"run_id": doc["run_id"], "mechanism": mech, "mechanism_check": bool(ok),
            "performance_effect": bool(perf_effect), "status": status, "evidence": ev}
