import copy

from analysis.validation import validate
from fixtures import run_doc


def by_name(checks):
    return {c["name"]: c for c in checks}


def test_clean_gpu_doc_passes_gates():
    doc = run_doc("a")
    doc["nccl"] = {"transports": ["SHM/direct"], "runtime_version": "2.31.2"}
    c = by_name(validate([doc]))
    assert all(x["passed"] for x in c.values() if x["kind"] == "gate"), c


def test_byte_mismatch_fails():
    doc = run_doc("a")
    doc["model"]["grad_bytes_fp32"] = 999
    assert not by_name(validate([doc]))["comm_bytes_equal_grad_bytes"]["passed"]


def test_causality_violation_fails():
    doc = run_doc("a")
    doc["steps"][0]["comm_causality_violations"] = 1
    assert not by_name(validate([doc]))["comm_causality"]["passed"]


def test_event_wall_disagreement_fails():
    doc = run_doc("a")
    for s in doc["steps"]:
        s["wall_step_ms"] = s["step_time_ms"] + 10.0
    assert not by_name(validate([doc]))["event_vs_wall_step"]["passed"]


def test_hook_overhead_gate_can_fail_and_pass():
    on = [run_doc(f"on{i}") for i in range(2)]
    off_same = [run_doc(f"off{i}") for i in range(2)]
    assert by_name(validate(on, off_same))["hook_overhead"]["passed"]
    off_fast = copy.deepcopy(off_same)
    for d in off_fast:
        for s in d["steps"]:
            s["step_time_ms"] *= 0.9          # hook adds ~11 %
    c = by_name(validate(on, off_fast))["hook_overhead"]
    assert c["kind"] == "gate" and not c["passed"]


def test_noisy_overhead_comparison_cannot_pass():
    on = [run_doc("on0"), run_doc("on1")]
    off = [run_doc("off0"), copy.deepcopy(run_doc("off1"))]
    for s in off[1]["steps"]:
        s["step_time_ms"] *= 1.3              # launch-to-launch noise >> tolerance
    assert not by_name(validate(on, off))["hook_overhead"]["passed"]


def test_missing_transport_fails_on_nccl():
    doc = run_doc("a")
    doc["nccl"] = {"transports": [], "runtime_version": None}
    assert not by_name(validate([doc]))["nccl_transport_recorded"]["passed"]
