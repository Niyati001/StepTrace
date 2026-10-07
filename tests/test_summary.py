import pytest

from analysis.pilot import select
from analysis.summary import block_table, cluster_steps, describe, summarize
from fixtures import run_doc, step


def test_describe():
    d = describe([1, 2, 3, 4, None])
    assert d["n"] == 4 and d["median"] == 2.5 and d["iqr"] == pytest.approx(1.5)
    assert describe([None])["n"] == 0


def test_cluster_step_uses_slowest_rank_and_skews():
    rows = cluster_steps([step("r", 0, 0, "ddp", 0, 0, compute=40, start_ns=0),
                          step("r", 0, 0, "ddp", 0, 1, compute=44, start_ns=2_000_000)])
    assert len(rows) == 1
    r = rows[0]
    assert r["step_time_ms"] == pytest.approx(max(40, 44) + 0.1 + 0)  # compute + data wait (tail=0)
    assert r["rank_compute_skew_ms"] == pytest.approx(4.0)
    assert r["rank_start_skew_ms"] == pytest.approx(2.0)


def test_outliers_flagged_not_dropped():
    recs = [step("r", 0, 0, "ddp", i, 0) for i in range(20)]
    recs[5] = step("r", 0, 0, "ddp", 5, 0, compute=400)
    b = block_table(cluster_steps(recs))[0]
    assert b["outlier_steps"] == 1 and b["n_steps"] == 20


def test_ablation_recovers_injected_exposed_time():
    s = summarize([run_doc("a", comm_tail=8.0)])
    abl = s["ablation_exposed_ms"]
    assert abl["n"] == 3 and abl["all_positive"]
    assert abl["median"] == pytest.approx(8.0, abs=0.05)
    assert s["modes"]["ddp"]["exposed_communication_time_ms"]["median"] == pytest.approx(8.0, abs=0.05)
    m = s["measurability"]
    assert m["status"] == "ok" and m["resolvable"]
    assert abl["fraction_of_ddp_step"] == pytest.approx(8.0 / 48.1, abs=0.01)


def test_measurability_material_threshold():
    s = summarize([run_doc("a", comm_tail=8.0)])
    assert s["measurability"]["communication_relevant"]          # 8 ms of ~48 ms > 10 %
    s_small = summarize([run_doc("s", comm_tail=2.0)])
    assert s_small["measurability"]["resolvable"]                # clearly above noise...
    assert not s_small["measurability"]["material"]              # ...but only ~5 % of the step
    s0 = summarize([run_doc("b", comm_tail=0.0)])
    assert not s0["measurability"]["communication_relevant"]


def test_measurability_needs_repeats():
    s = summarize([run_doc("a", repeats=1)])
    assert s["measurability"]["status"] == "insufficient_repeats"


def test_selection_rule_prefers_first_model_then_target_fraction():
    rows = [
        {"id": "r18_a", "model": "resnet18", "communication_relevant": True, "ablation_fraction": 0.6},
        {"id": "r18_b", "model": "resnet18", "communication_relevant": True, "ablation_fraction": 0.2},
        {"id": "r18_c", "model": "resnet18", "communication_relevant": False, "ablation_fraction": 0.25},
        {"id": "gpt", "model": "transformer_small", "communication_relevant": True, "ablation_fraction": 0.25},
    ]
    assert select(rows, ["resnet18", "transformer_small"])["selected"] == "r18_b"
    rows = [r for r in rows if r["model"] != "resnet18" or not r["communication_relevant"]]
    sel = select(rows, ["resnet18", "transformer_small"])
    assert sel["selected"] == "gpt" and sel["rejected_models"] == ["resnet18"]
    assert select([], ["resnet18"])["selected"] is None
