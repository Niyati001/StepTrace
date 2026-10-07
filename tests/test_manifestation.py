"""Mechanism-level manifestation criteria (independent of the diagnoser), CPU only."""

import copy

import pytest

from faults import manifestation as MF
from test_diagnoser import run


def doc(mech, params, **kw):
    d = run(f"x_{mech}", **kw)
    d["ground_truth"] = {"fault_class": "X", "fault_mechanism": mech, "fault_parameters": params,
                         "level": "", "emulated": mech == "emulated_bandwidth"}
    d["config"] = {"workload": {"batch_size": 32}}
    d["effective_config"] = {"workload": {"batch_size": 32, "num_workers": 2}}
    for s in d["steps"]:
        s["step_start_monotonic_ns"] = (s["block_id"] * 1000 + s["step"]) * 40_000_000 + s["rank"] * 10_000
    return d


@pytest.fixture(scope="module")
def ref():
    return MF.reference_profile([doc("none", {}, seed=i) for i in range(4)])


def test_reference_profile(ref):
    assert ref["cycle_median_ms"] == pytest.approx(40.0)
    assert ref["step_block_min"] <= ref["step_median"] <= ref["step_block_max"]


def test_straggler_manifests_from_target_rank_compute(ref):
    r = MF.check(doc("sleep", {"rank": 1, "delay_ms": 8.0}, extra_compute_rank1=8.0), ref)
    assert r["status"] == "MANIFESTED" and r["evidence"]["target_rank_compute_excess_ms"] == pytest.approx(8.0)


def test_no_mechanism_and_no_perf_effect_is_did_not_manifest(ref):
    r = MF.check(doc("sleep", {"rank": 1, "delay_ms": 8.0}, seed=7), ref)
    assert r["status"] == "DID_NOT_MANIFEST" and not r["mechanism_check"] and not r["performance_effect"]


def test_slowdown_through_another_pathway_is_never_excused(ref):
    """Mechanism check fails but the step slowed: scored as manifested (a miss stays a miss)."""
    r = MF.check(doc("sleep", {"rank": 1, "delay_ms": 8.0}, data_wait=6.0), ref)
    assert r["status"] == "MANIFESTED_UNEXPECTED_PATHWAY"


def test_fetch_sleep_uses_host_fetch_wait(ref):
    d = doc("fetch_sleep", {"delay_ms": 4.0}, data_wait=4.0)
    for s in d["steps"]:
        s["host_data_wait_ms"] = 4.1
    assert MF.check(d, ref, host_ref_data_wait_ms=0.1)["status"] == "MANIFESTED"
    for s in d["steps"]:
        s["host_data_wait_ms"] = 0.5
    assert not MF.check(d, ref, host_ref_data_wait_ms=0.1)["mechanism_check"]


def test_rate_based_loader_expectation_uses_consumer_cycle(ref):
    # 2 workers x 45 ms per batch -> 22.5 ms per batch produced < 40 ms cycle: nothing to observe
    d = doc("loader_sleep", {"delay_ms": 45.0}, seed=3)
    r = MF.check(d, ref, host_ref_data_wait_ms=0.0)
    assert r["evidence"]["intended_ms"] == 0.0 and r["status"] == "DID_NOT_MANIFEST"


def test_structural_and_transport_checks(ref):
    d = doc("single_bucket", {"bucket_mb": 1024})
    assert not MF.check(d, ref)["mechanism_check"]                 # same bucket structure as healthy
    for s in d["steps"]:
        s["bucket_count"] = 1
    assert MF.check(d, ref)["mechanism_check"]
    t = doc("shm_disable", {})
    t["nccl"] = {"transports": ["NET/Socket/0"]}
    ref2 = dict(ref, transports=["SHM/direct"])
    assert MF.check(t, ref2)["mechanism_check"]
    t["nccl"] = {"transports": ["SHM/direct"]}
    assert not MF.check(t, ref2)["mechanism_check"]


def test_emulated_bandwidth_needs_busy_increase(ref):
    d = doc("emulated_bandwidth", {"throttle_GBps": 8.0}, busy=11.0)
    for s in d["steps"]:
        s["injected_comm_delay_ms"] = 5.0
    assert not MF.check(d, ref)["mechanism_check"]
    d2 = doc("emulated_bandwidth", {"throttle_GBps": 8.0}, busy=16.0, exposed=11.0)
    for s in d2["steps"]:
        s["injected_comm_delay_ms"] = 5.0
    assert MF.check(d2, ref)["mechanism_check"]


def test_criteria_never_read_diagnoser(ref):
    import inspect
    src = inspect.getsource(MF)
    assert "diagnose" not in src.split('"""', 2)[2]                # code below the docstring
    d = doc("sleep", {"rank": 1, "delay_ms": 8.0}, extra_compute_rank1=8.0)
    assert MF.check(copy.deepcopy(d), ref) == MF.check(d, ref)


def test_loader_stall_uses_mean_wait_and_host_loader_cycle(ref):
    """Paired worker arrivals give a bimodal wait: median ~0 but mean ~ the predicted stall."""
    host = MF.reference_profile([doc("none", {}, seed=i) for i in range(2)])
    host = dict(host, cycle_median_ms=44.0, host_data_wait_mean_ms=0.3)
    d = doc("loader_sleep", {"delay_ms": 100.0}, seed=3, data_wait=0.0)    # d/W = 50 -> expect 6 ms
    for i, s in enumerate(d["steps"]):
        s["host_data_wait_ms"] = 12.0 if i % 2 else 0.0                    # median 0/12, mean 6
    r = MF.check(d, ref, host_ref=host)
    assert r["evidence"]["intended_ms"] == pytest.approx(6.0)
    assert r["evidence"]["host_data_wait_increase_ms"] == pytest.approx(5.7, abs=0.4)
    assert r["mechanism_check"] and r["status"] == "MANIFESTED"
    for s in d["steps"]:
        s["host_data_wait_ms"] = 0.3
    assert not MF.check(d, ref, host_ref=host)["mechanism_check"]
