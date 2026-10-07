"""Optimization metrics: throughput and training-correctness comparison (pure; CPU-tested).

Throughput = samples per second per GPU = batch_size * grad_accum_steps / cluster step time,
taken as the median over a run's blocks of the block-median cluster step (slowest rank).

Correctness compares runs with IDENTICAL seeds and data order:
  * loss curve: rank-0 per-step loss in execution order;
  * parameter fingerprint: per-tensor L2 norms at the end of the measured blocks.
The tolerated deviation is the larger of a fixed floor and K_NOISE times the deviation
between two identical baseline runs (GPU nondeterminism, e.g. cuDNN autotuning/atomics).
Semantics-changing candidates (e.g. gradient accumulation, which changes the effective
batch per optimizer step) are compared on loss versus SAMPLES processed instead of steps,
and only on the final-loss criterion.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from analysis.summary import cluster_steps

K_NOISE = 3.0
CURVE_REL_FLOOR = 0.02        # median relative loss-curve deviation tolerated
FINAL_LOSS_REL_TOL = 0.02     # relative difference of mean loss over the final 10 % of training
FINGERPRINT_REL_FLOOR = 1e-3  # max per-tensor relative parameter-norm difference


def throughput(doc: dict) -> dict:
    cfg = doc["effective_config"]["workload"]
    samples = cfg["batch_size"] * cfg["grad_accum_steps"]
    per_block = defaultdict(list)
    for c in cluster_steps(doc["steps"]):
        if c["mode"] == "ddp":
            per_block[c["block_id"]].append(c["step_time_ms"])
    med = float(np.median([np.median(v) for v in per_block.values()]))
    return {"step_ms": med, "samples_per_step": samples, "samples_per_s_per_gpu": samples / (med / 1e3)}


def loss_curve(doc: dict) -> np.ndarray:
    rows = sorted((s["block_id"], s["step"], s["loss"]) for s in doc["steps"]
                  if s["rank"] == 0 and s["mode"] == "ddp")
    return np.array([r[2] for r in rows], dtype=float)


def curve_deviation(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.median(np.abs(a[:n] - b[:n]) / np.maximum(np.abs(a[:n]), 1e-8)))


def final_loss(curve: np.ndarray) -> float:
    k = max(1, len(curve) // 10)
    return float(np.mean(curve[-k:]))


def fingerprint_deviation(fa: dict, fb: dict) -> float:
    return max(abs(fa[k] - fb[k]) / max(abs(fa[k]), 1e-12) for k in fa)


def correctness(baseline_a: dict, baseline_b: dict, candidate: dict, semantics_preserving: bool) -> dict:
    """baseline_a/_b: two identical-seed baseline runs (noise floor); candidate: same seed."""
    ca, cb, cc = loss_curve(baseline_a), loss_curve(baseline_b), loss_curve(candidate)
    noise_curve = curve_deviation(ca, cb)
    fl_a, fl_c = final_loss(ca), final_loss(cc)
    final_rel = abs(fl_c - fl_a) / max(abs(fl_a), 1e-8)
    noise_final = abs(final_loss(cb) - fl_a) / max(abs(fl_a), 1e-8)
    final_tol = max(FINAL_LOSS_REL_TOL, K_NOISE * noise_final)
    out = {"semantics_preserving": semantics_preserving, "final_loss_baseline": fl_a, "final_loss_candidate": fl_c,
           "final_loss_rel_diff": final_rel, "final_loss_tol": final_tol,
           "noise_curve_deviation": noise_curve, "noise_final_rel": noise_final}
    checks = {"final_loss": final_rel <= final_tol}
    if semantics_preserving:
        dev = curve_deviation(ca, cc)
        tol = max(CURVE_REL_FLOOR, K_NOISE * noise_curve)
        out.update(curve_deviation=dev, curve_tol=tol)
        checks["loss_curve"] = dev <= tol
        fa = (baseline_a.get("fingerprint_by_rank") or [None])[0]
        fb = (baseline_b.get("fingerprint_by_rank") or [None])[0]
        fc = (candidate.get("fingerprint_by_rank") or [None])[0]
        if fa and fb and fc:
            fdev, fnoise = fingerprint_deviation(fa, fc), fingerprint_deviation(fa, fb)
            ftol = max(FINGERPRINT_REL_FLOOR, K_NOISE * fnoise)
            out.update(fingerprint_deviation=fdev, fingerprint_noise=fnoise, fingerprint_tol=ftol)
            checks["fingerprint"] = fdev <= ftol
    out["checks"] = checks
    out["passed"] = all(checks.values())
    return out
