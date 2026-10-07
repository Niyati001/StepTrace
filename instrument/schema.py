"""Raw result schema (version 3).

A run document (one launch) is JSON:

    {schema_version, run_id, experiment_id, launch_index, timestamp_utc,
     command, environment, config, model, blocks, steps, profile}

``steps`` holds one record per (block, measured step, rank). Communication
fields are 0 in ``nosync`` mode (no collective was issued), measurements in
``ddp`` mode, and null only when the comm hook was disabled
(``measurement.comm_hook: false``, used to measure instrumentation overhead).
"""

from __future__ import annotations

SCHEMA_VERSION = 3  # v3: provenance, effective_config, ground_truth

RUN_KEYS = ("schema_version", "run_id", "experiment_id", "launch_index", "timestamp_utc",
            "provenance", "environment", "config", "effective_config", "ground_truth",
            "model", "blocks", "steps")

# field -> (types, nullable)
_NUM = (int, float)
STEP_FIELDS: dict[str, tuple[tuple, bool]] = {
    "run_id": ((str,), False),
    "block_id": ((int,), False),
    "repeat": ((int,), False),
    "mode": ((str,), False),
    "step": ((int,), False),
    "rank": ((int,), False),
    "timing_source": ((str,), False),
    "step_time_ms": (_NUM, False),
    "wall_step_ms": (_NUM, False),
    "data_wait_ms": (_NUM, False),
    "host_data_wait_ms": (_NUM, False),
    "accumulation_ms": (_NUM, False),
    "forward_ms": (_NUM, False),
    "backward_ms": (_NUM, False),
    "optimizer_ms": (_NUM, False),
    "grads_ready_ms": (_NUM, True),
    "backward_compute_ms": (_NUM, True),
    "compute_time_ms": (_NUM, True),
    "communication_time_ms": (_NUM, True),
    "communication_span_ms": (_NUM, True),
    "exposed_communication_time_ms": (_NUM, True),
    "ddp_finalize_ms": (_NUM, True),
    "communication_bytes": ((int,), True),
    "bucket_count": ((int,), True),
    "comm_causality_violations": ((int,), True),
    "step_start_monotonic_ns": ((int,), False),
    "loss": (_NUM, False),
    "gpu_memory_mb": (_NUM, True),
    "gpu_memory_reserved_mb": (_NUM, True),
    "gpu_utilization_pct": (_NUM, True),
}
MODES = ("ddp", "nosync")


def validate_step(rec: dict) -> list[str]:
    errs = []
    for k, (types, nullable) in STEP_FIELDS.items():
        if k not in rec:
            errs.append(f"missing {k}")
            continue
        v = rec[k]
        if v is None:
            if not nullable:
                errs.append(f"{k} is null")
        elif isinstance(v, bool) or not isinstance(v, types):
            errs.append(f"{k} has type {type(v).__name__}")
    if rec.get("mode") not in MODES:
        errs.append(f"mode {rec.get('mode')!r} invalid")
    if rec.get("mode") == "nosync" and rec.get("communication_bytes"):
        errs.append("nosync step reports communication bytes")
    if rec.get("step_time_ms") is not None and rec["step_time_ms"] <= 0:
        errs.append("non-positive step time")
    return errs


def validate_run(doc: dict) -> list[str]:
    errs = [f"missing run key {k}" for k in RUN_KEYS if k not in doc]
    if doc.get("schema_version") != SCHEMA_VERSION:
        errs.append(f"schema_version {doc.get('schema_version')!r} != {SCHEMA_VERSION}")
    for i, s in enumerate(doc.get("steps", [])):
        errs += [f"steps[{i}]: {e}" for e in validate_step(s)]
        if len(errs) > 20:
            break
    return errs
