"""Observable features for diagnosis (raw measurement -> derived metric).

The diagnoser sees ONLY ``observable_view(doc)``: the run id, world size, model
size and a whitelist of measured per-step fields. Ground truth, configuration,
fault parameters and injector bookkeeping are never visible to it.

Per cluster step (ranks combined), then per block (median over steps):

| feature            | definition (per step)                                   | targets          |
|--------------------|---------------------------------------------------------|------------------|
| step_ms            | max over ranks of step_time_ms                          | supporting       |
| data_wait_ms       | mean over ranks of data_wait_ms                         | DATA_STALL       |
| compute_skew_ms    | max - min over ranks of compute_time_ms                 | STRAGGLER        |
| exposed_min_ms     | MIN over ranks of exposed_communication_time_ms         | COMMUNICATION    |
| exposed_min_frac   | exposed_min_ms / step_ms                                | COMMUNICATION    |
| comm_busy_ms       | mean over ranks of communication_time_ms                | supporting       |
| comm_GBps          | communication_bytes / comm_busy                         | supporting       |
| compute_ms         | mean over ranks of compute_time_ms                      | supporting       |

Why MIN over ranks for exposed communication: in a synchronous collective a fast
rank waits inside the collective for a slow rank, so its "exposed communication"
contains straggler wait. The rank that arrived last waited least; its exposed time
is the best per-step estimate of communication on the critical path itself.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from instrument.schema import STEP_FIELDS

OBSERVABLE_RUN_KEYS = ("run_id", "world_size", "model")
FEATURES = ("step_ms", "data_wait_ms", "compute_skew_ms", "exposed_min_ms", "exposed_min_frac",
            "comm_busy_ms", "comm_GBps", "compute_ms")


def observable_view(doc: dict) -> dict:
    """Strip everything except measurements. The diagnoser must work from this alone."""
    view = {k: doc[k] for k in OBSERVABLE_RUN_KEYS}
    view["steps"] = [{k: s[k] for k in STEP_FIELDS if k in s} for s in doc["steps"]]
    return view


def _cluster_rows(steps: list[dict]) -> dict[tuple, list[dict]]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for s in steps:
        if s["mode"] == "ddp" and s.get("communication_time_ms") is not None:
            g[(s["block_id"], s["step"])].append(s)
    return g


def block_features(obs: dict) -> list[dict]:
    per_block: dict[int, list[dict]] = defaultdict(list)
    for (block, _), rs in sorted(_cluster_rows(obs["steps"]).items()):
        step = max(r["step_time_ms"] for r in rs)
        busy = float(np.mean([r["communication_time_ms"] for r in rs]))
        nbytes = float(np.mean([r["communication_bytes"] for r in rs]))
        exp_min = min(r["exposed_communication_time_ms"] for r in rs)
        comp = [r["compute_time_ms"] for r in rs]
        per_block[block].append({
            "step_ms": step,
            "data_wait_ms": float(np.mean([r["data_wait_ms"] for r in rs])),
            "compute_skew_ms": max(comp) - min(comp),
            "exposed_min_ms": exp_min,
            "exposed_min_frac": exp_min / step,
            "comm_busy_ms": busy,
            "comm_GBps": (nbytes / (busy / 1e3) / 1e9) if busy > 0 else float("nan"),
            "compute_ms": float(np.mean(comp)),
        })
    out = []
    for block, rows in sorted(per_block.items()):
        f = {k: float(np.median([r[k] for r in rows])) for k in FEATURES}
        f.update(run_id=obs["run_id"], block_id=block, n_steps=len(rows))
        out.append(f)
    return out
