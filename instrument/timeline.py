"""Per-step timeline derivation (pure functions; no torch, unit-tested on CPU).

Inputs are offsets in milliseconds from the step-start mark, all on one
device timeline (CUDA events on the GPU, host clock on CPU):

    marks: data_ready, accum_end, fwd_end, bwd_end, opt_end
    comm:  one dict per DDP bucket collective, in issue order:
           {"ready_ms", "end_ms", "bytes"}
           ready_ms = bucket gradients ready on the compute stream (hook entry)
           end_ms   = collective complete (recorded on a stream that waits on it)
           or None when the comm hook was disabled (communication not measured)

Collectives of one process group execute serially on one communication
stream, so collective k actually starts at max(ready_k, end_{k-1}).
A collective that "ends" before its bucket was ready is physically impossible;
such cases are counted in ``comm_causality_violations`` (a measurement-validity
signal checked by analysis.validation), never silently clamped away.
See METHODOLOGY.md for the definitions and their limits.
"""

from __future__ import annotations

COMM_FIELDS = ("grads_ready_ms", "backward_compute_ms", "communication_time_ms",
               "communication_span_ms", "exposed_communication_time_ms", "ddp_finalize_ms",
               "communication_bytes", "bucket_count", "comm_causality_violations",
               "compute_time_ms")


def comm_intervals(comm: list[dict]) -> tuple[list[tuple[float, float]], int]:
    out, prev_end, violations = [], float("-inf"), 0
    for c in comm:
        start = max(c["ready_ms"], prev_end)
        if c["end_ms"] < c["ready_ms"]:
            violations += 1
        end = max(c["end_ms"], start)
        out.append((start, end))
        prev_end = end
    return out, violations


def derive_step(marks: dict[str, float], comm: list[dict] | None) -> dict[str, float | int | None]:
    step = marks["opt_end"]
    data_ready = marks["data_ready"]
    accum_end = marks.get("accum_end", data_ready)
    fwd_end, bwd_end = marks["fwd_end"], marks["bwd_end"]
    out: dict[str, float | int | None] = {
        "step_time_ms": step,
        "data_wait_ms": data_ready,
        "accumulation_ms": accum_end - data_ready,
        "forward_ms": fwd_end - accum_end,
        "backward_ms": bwd_end - fwd_end,
        "optimizer_ms": step - bwd_end,
    }
    if comm is None:  # hook disabled: communication was not measured
        out.update({k: None for k in COMM_FIELDS})
        return out
    if comm:
        iv, viol = comm_intervals(comm)
        grads_ready = comm[-1]["ready_ms"]     # last issued bucket => all gradients produced
        comm_end = iv[-1][1]
        out.update({
            "grads_ready_ms": grads_ready,
            "backward_compute_ms": grads_ready - fwd_end,
            "communication_time_ms": sum(e - s for s, e in iv),
            "communication_span_ms": comm_end - iv[0][0],
            "exposed_communication_time_ms": max(0.0, comm_end - grads_ready),
            "ddp_finalize_ms": max(0.0, bwd_end - max(comm_end, grads_ready)),
            "communication_bytes": sum(int(c["bytes"]) for c in comm),
            "bucket_count": len(comm),
            "comm_causality_violations": viol,
        })
    else:  # no collective issued (nosync mode): communication is known to be zero
        out.update({
            "grads_ready_ms": bwd_end,
            "backward_compute_ms": bwd_end - fwd_end,
            "communication_time_ms": 0.0,
            "communication_span_ms": 0.0,
            "exposed_communication_time_ms": 0.0,
            "ddp_finalize_ms": 0.0,
            "communication_bytes": 0,
            "bucket_count": 0,
            "comm_causality_violations": 0,
        })
    out["compute_time_ms"] = (out["accumulation_ms"] + out["forward_ms"]
                              + out["backward_compute_ms"] + out["optimizer_ms"])
    return out
