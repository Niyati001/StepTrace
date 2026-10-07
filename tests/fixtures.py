"""Deterministic synthetic run documents with known ground-truth statistics."""

from instrument.schema import SCHEMA_VERSION
from instrument.timeline import derive_step

GRAD_BYTES = 1000


def step(run_id, block_id, repeat, mode, i, rank, compute=40.0, comm_tail=0.0, jitter=0.0,
         start_ns=0, data_wait=0.1, timing_source="cuda_event"):
    fwd_end = data_wait + compute * 0.3
    grads_ready = data_wait + compute * 0.9 + jitter
    marks = {"data_ready": data_wait, "accum_end": data_wait, "fwd_end": fwd_end}
    if mode == "ddp":
        comm = [{"ready_ms": fwd_end + 5, "end_ms": fwd_end + 10, "bytes": GRAD_BYTES // 2},
                {"ready_ms": grads_ready, "end_ms": grads_ready + comm_tail, "bytes": GRAD_BYTES // 2}]
        bwd_end = grads_ready + comm_tail
    else:
        comm, bwd_end = [], grads_ready
    marks.update(bwd_end=bwd_end, opt_end=bwd_end + compute * 0.1)
    r = derive_step(marks, comm)
    r.update(run_id=run_id, block_id=block_id, repeat=repeat, mode=mode, step=i, rank=rank,
             timing_source=timing_source, wall_step_ms=r["step_time_ms"] + 0.05, host_data_wait_ms=0.0,
             step_start_monotonic_ns=start_ns + i * 10**8, loss=2.3,
             gpu_memory_mb=100.0, gpu_memory_reserved_mb=200.0, gpu_utilization_pct=90.0)
    return r


def run_doc(run_id, comm_tail=8.0, repeats=3, steps=20, drift=(0.0, 0.3, -0.2), backend="nccl",
            timing_source="cuda_event"):
    recs, blocks, b = [], [], 0
    for rep in range(repeats):
        for mode in (("ddp", "nosync") if rep % 2 == 0 else ("nosync", "ddp")):
            for i in range(steps):
                for rank in (0, 1):
                    recs.append(step(run_id, b, rep, mode, i, rank, compute=40.0 + drift[rep % len(drift)],
                                     comm_tail=comm_tail, jitter=0.01 * ((i * 7 + rank) % 5),
                                     start_ns=rank * 50_000, timing_source=timing_source))
            blocks.append({"block_id": b, "repeat": rep, "mode": mode})
            b += 1
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "experiment_id": "t", "launch_index": 0,
            "timestamp_utc": "x", "environment": {}, "config": {}, "backend": backend,
            "model": {"grad_bytes_fp32": GRAD_BYTES}, "blocks": blocks, "steps": recs,
            "schema_errors": [], "profile": None, "gpu_state": [], "nccl": None}
