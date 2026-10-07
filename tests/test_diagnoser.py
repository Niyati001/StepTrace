"""Diagnoser rules on deterministic synthetic runs with known signatures (no GPU)."""

import copy

import pytest

from diagnose.diagnose import build_reference, diagnose, format_report, rules_fingerprint
from diagnose.features import block_features, observable_view
from diagnose.thresholds import MIN_REFERENCE_BLOCKS, reference_stats
from instrument.schema import SCHEMA_VERSION
from instrument.timeline import derive_step

GRAD = 44_695_848


def run(run_id, blocks=3, steps=30, seed=0, compute=30.0, extra_compute_rank1=0.0, data_wait=0.02,
        exposed=7.0, busy=11.0, truth="HEALTHY"):
    """Two ranks; compute/comm shaped like the M1 regime (~39 ms step, ~7 ms exposed)."""
    recs = []
    for b in range(blocks):
        for i in range(steps):
            j = ((i * 7 + b * 3 + seed) % 11 - 5) * 0.02          # deterministic +-0.1 ms jitter
            starts = {}
            for rank in (0, 1):
                c = compute + j + (extra_compute_rank1 if rank == 1 else 0.0)
                fwd_end = data_wait + 0.3 * c
                ready = data_wait + 0.9 * c
                starts[rank] = ready
            last_ready = max(starts.values())
            for rank in (0, 1):
                c = compute + j + (extra_compute_rank1 if rank == 1 else 0.0)
                fwd_end = data_wait + 0.3 * c
                ready = starts[rank]
                end = last_ready + exposed                           # collective ends together
                comm = [{"ready_ms": fwd_end + 2, "end_ms": fwd_end + 2 + (busy - exposed), "bytes": GRAD // 2},
                        {"ready_ms": ready, "end_ms": end, "bytes": GRAD - GRAD // 2}]
                marks = {"data_ready": data_wait, "accum_end": data_wait, "fwd_end": fwd_end,
                         "bwd_end": end + 0.2, "opt_end": end + 0.2 + 0.1 * c}
                r = derive_step(marks, comm)
                r.update(run_id=run_id, block_id=b, repeat=b, mode="ddp", step=i, rank=rank,
                         timing_source="cuda_event", wall_step_ms=r["step_time_ms"] + 0.05,
                         host_data_wait_ms=0.0, step_start_monotonic_ns=0, loss=2.3,
                         gpu_memory_mb=1.0, gpu_memory_reserved_mb=1.0, gpu_utilization_pct=90.0,
                         injected_comm_delay_ms=123.0)           # injector bookkeeping: must be invisible
                recs.append(r)
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "world_size": 2,
            "model": {"grad_bytes_fp32": GRAD}, "steps": recs,
            "ground_truth": {"fault_class": truth}, "config": {"fault": {"mechanism": "secret"}}}


@pytest.fixture(scope="module")
def ref():
    docs = [run(f"ref{i}", seed=i) for i in range(4)]
    return docs, build_reference(docs)


def test_observable_view_hides_ground_truth_and_bookkeeping():
    d = run("x")
    v = observable_view(d)
    assert set(v) == {"run_id", "world_size", "model", "steps"}
    assert "injected_comm_delay_ms" not in v["steps"][0]


def test_diagnosis_independent_of_ground_truth_and_config(ref):
    _, stats = ref
    d = run("x", extra_compute_rank1=8.0, truth="STRAGGLER")
    a = diagnose(d, stats)
    d2 = copy.deepcopy(d)
    d2["ground_truth"] = {"fault_class": "HEALTHY"}
    d2["config"] = {}
    b = diagnose(d2, stats)
    assert a["verdict"] == b["verdict"] and a["agreement"] == b["agreement"]


def test_healthy_run_is_healthy(ref):
    _, stats = ref
    r = diagnose(run("h", seed=9), stats)
    assert r["verdict"] == "HEALTHY" and r["healthy"] and r["agreement"] == "3/3"


@pytest.mark.parametrize("kw,expected", [
    ({"extra_compute_rank1": 8.0}, "STRAGGLER"),
    ({"data_wait": 6.0}, "DATA_STALL"),
    ({"exposed": 14.0, "busy": 18.0}, "COMMUNICATION"),
    ({"compute": 8.0}, "COMMUNICATION"),            # small batch: same comm, less compute -> fraction up
])
def test_fault_signatures(ref, kw, expected):
    _, stats = ref
    r = diagnose(run("f", seed=5, **kw), stats)
    assert r["verdict"] == expected, format_report(r)
    assert not r["healthy"]


def test_straggler_not_blamed_on_communication(ref):
    """Fast rank waits in the collective; exposed_min (slowest rank) must not move."""
    _, stats = ref
    r = diagnose(run("s", extra_compute_rank1=10.0), stats)
    assert r["verdict"] == "STRAGGLER"
    assert not r["blocks"][0]["causes"]["COMMUNICATION"]["elevated"]


def test_tiny_changes_below_practical_floor_stay_healthy(ref):
    _, stats = ref
    # 0.5 ms more data wait: huge robust z (reference ~0.02 ms) but < 3 % of the step
    r = diagnose(run("t", data_wait=0.5), stats)
    assert r["verdict"] == "HEALTHY"
    t = r["blocks"][0]["causes"]["DATA_STALL"]["tests"][0]
    assert t["z"] >= 4 and not t["elevated"]


def test_mixed_fault_primary_is_largest_effect_and_secondary_reported(ref):
    _, stats = ref
    r = diagnose(run("m", data_wait=12.0, extra_compute_rank1=4.0), stats)
    assert r["verdict"] == "DATA_STALL"
    assert "STRAGGLER" in r["blocks"][0]["secondary"]


def test_reference_requires_enough_blocks():
    blocks = block_features(observable_view(run("a", blocks=2)))
    with pytest.raises(ValueError):
        reference_stats(blocks[: MIN_REFERENCE_BLOCKS - 1])


def test_report_is_traceable(ref):
    _, stats = ref
    r = diagnose(run("f", data_wait=6.0), stats)
    txt = format_report(r)
    for s in ("Primary cause: DATA_STALL", "data_wait_ms", "robust z", "Rule agreement", "Rules sha256"):
        assert s in txt
    assert len(rules_fingerprint()["sha256"]) == 64
