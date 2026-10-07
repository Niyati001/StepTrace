import pytest

from instrument.timeline import comm_intervals, derive_step

MARKS = {"data_ready": 1.0, "accum_end": 1.0, "fwd_end": 11.0, "bwd_end": 52.0, "opt_end": 55.0}


def _closes(r):
    return r["data_wait_ms"] + r["compute_time_ms"] + r["exposed_communication_time_ms"] \
        + r["ddp_finalize_ms"] == pytest.approx(r["step_time_ms"])


def test_serial_queueing_start_is_max_of_ready_and_previous_end():
    comm = [{"ready_ms": 10, "end_ms": 18, "bytes": 1},
            {"ready_ms": 20, "end_ms": 35, "bytes": 1},   # starts at its ready time
            {"ready_ms": 30, "end_ms": 50, "bytes": 1}]   # queued behind bucket 1 -> starts at 35
    iv, viol = comm_intervals(comm)
    assert iv == [(10, 18), (20, 35), (35, 50)]
    assert viol == 0


def test_overlapped_and_exposed_parts():
    comm = [{"ready_ms": 20, "end_ms": 30, "bytes": 100}, {"ready_ms": 40, "end_ms": 50, "bytes": 50}]
    r = derive_step(MARKS, comm)
    assert r["communication_time_ms"] == 20          # 10 + 10, serial busy time
    assert r["communication_span_ms"] == 30
    assert r["grads_ready_ms"] == 40
    assert r["exposed_communication_time_ms"] == 10  # 50 - 40: only the tail is on the critical path
    assert r["ddp_finalize_ms"] == 2                 # 52 - 50
    assert r["backward_compute_ms"] == 29            # 40 - 11
    assert r["compute_time_ms"] == 10 + 29 + 3
    assert r["communication_bytes"] == 150 and r["bucket_count"] == 2
    assert _closes(r)


def test_fully_hidden_communication_has_zero_exposure():
    comm = [{"ready_ms": 20, "end_ms": 25, "bytes": 1}, {"ready_ms": 45, "end_ms": 44.5, "bytes": 1}]
    r = derive_step(MARKS, comm)
    assert r["comm_causality_violations"] == 1      # end before ready is reported, not hidden
    comm[1]["end_ms"] = 45
    r = derive_step(MARKS, comm)
    assert r["exposed_communication_time_ms"] == 0
    assert r["comm_causality_violations"] == 0
    assert _closes(r)


def test_nosync_has_known_zero_communication():
    r = derive_step(MARKS, [])
    assert r["communication_time_ms"] == 0 and r["communication_bytes"] == 0
    assert r["compute_time_ms"] == pytest.approx(54.0)
    assert _closes(r)


def test_hook_disabled_leaves_communication_unknown():
    r = derive_step(MARKS, None)
    assert r["communication_time_ms"] is None and r["compute_time_ms"] is None
    assert r["step_time_ms"] == 55.0 and r["data_wait_ms"] == 1.0


def test_gradient_accumulation_counts_as_compute():
    marks = dict(MARKS, accum_end=21.0, fwd_end=31.0)
    r = derive_step(marks, [{"ready_ms": 45, "end_ms": 50, "bytes": 1}])
    assert r["accumulation_ms"] == 20
    assert r["forward_ms"] == 10
    assert _closes(r)
