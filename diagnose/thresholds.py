"""Baseline-derived thresholds (spec §18).

The *formulas and constants* below are the rule set that gets frozen before
held-out evaluation. The *reference statistics* they are applied to are
re-estimated from the healthy reference runs of the SAME GPU session
(METHODOLOGY.md §M2: same-session reference).

For a feature x with healthy reference blocks r_1..r_n:
    center = median(r)
    scale  = max(1.4826 * MAD(r), SCALE_FLOOR_REL * |center|, SCALE_FLOOR_ABS[feature])
    z      = (x - center) / scale                       (robust z-score)
A feature is ELEVATED iff
    z >= K_ROBUST_Z                                      (statistically unusual for this session)
    AND (x - center) >= practical floor                  (large enough to matter)
practical floor: PRACTICAL_STEP_FRACTION * median healthy step time for ms features,
                 PRACTICAL_FRACTION_PP (absolute) for fraction features.
The scale floors stop a near-constant reference (e.g. data wait ~0.01 ms with
device-resident data) from making microscopic changes look like 100-sigma events;
the practical floor is the second guard.
"""

from __future__ import annotations

import numpy as np

from diagnose.features import FEATURES

K_ROBUST_Z = 4.0
PRACTICAL_STEP_FRACTION = 0.03
PRACTICAL_FRACTION_PP = 0.03
SCALE_FLOOR_REL = 0.01
SCALE_FLOOR_ABS = {"step_ms": 0.05, "data_wait_ms": 0.05, "compute_skew_ms": 0.05,
                   "exposed_min_ms": 0.05, "exposed_min_frac": 0.002, "comm_busy_ms": 0.05,
                   "comm_GBps": 0.01, "compute_ms": 0.05}
MIN_REFERENCE_BLOCKS = 6


def reference_stats(ref_blocks: list[dict]) -> dict:
    if len(ref_blocks) < MIN_REFERENCE_BLOCKS:
        raise ValueError(f"need >= {MIN_REFERENCE_BLOCKS} healthy reference blocks, got {len(ref_blocks)}")
    stats = {}
    for k in FEATURES:
        v = np.array([b[k] for b in ref_blocks], dtype=float)
        v = v[np.isfinite(v)]
        center = float(np.median(v))
        mad = float(np.median(np.abs(v - center)))
        stats[k] = {"center": center, "mad": mad, "n": int(v.size),
                    "scale": max(1.4826 * mad, SCALE_FLOOR_REL * abs(center), SCALE_FLOOR_ABS[k]),
                    "min": float(v.min()), "max": float(v.max())}
    stats["_practical_ms"] = PRACTICAL_STEP_FRACTION * stats["step_ms"]["center"]
    stats["_n_blocks"] = len(ref_blocks)
    stats["_runs"] = sorted({b["run_id"] for b in ref_blocks})
    return stats


def score_feature(x: float, feature: str, stats: dict) -> dict:
    s = stats[feature]
    delta = x - s["center"]
    z = delta / s["scale"]
    floor = PRACTICAL_FRACTION_PP if feature.endswith("_frac") else stats["_practical_ms"]
    return {"feature": feature, "value": x, "reference_center": s["center"],
            "reference_scale": s["scale"], "delta": delta, "z": z,
            "z_threshold": K_ROBUST_Z, "practical_threshold": floor,
            "elevated": bool(z >= K_ROBUST_Z and delta >= floor)}
