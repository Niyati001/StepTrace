"""The eight required plots, exercised ONLY on clearly marked synthetic fixtures in a temp dir.
Nothing here is evidence; the fixtures never leave tmp_path."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analysis import plots as P
from diagnose.evaluate import evaluate
from test_diagnoser import run

ROOT = Path(__file__).resolve().parents[1]
CID = "SYNTHETIC-FIXTURE-campaign"


def _doc(run_id, mech, cls, **kw):
    d = run(run_id, **kw)
    d.update(synthetic_fixture=True, devices=["SYNTHETIC"], launch_index=0, backend="nccl",
             ground_truth={"fault_class": cls, "fault_mechanism": mech, "fault_parameters":
                           {"rank": 1, "delay_ms": 8.0} if mech == "sleep" else
                           {"delay_ms": 6.0} if mech == "fetch_sleep" else {},
                           "level": "", "emulated": False},
             config={"workload": {"batch_size": 32}},
             effective_config={"workload": {"batch_size": 32, "num_workers": 2, "grad_accum_steps": 1}})
    for s in d["steps"]:
        s["step_start_monotonic_ns"] = (s["block_id"] * 1000 + s["step"]) * 40_000_000
        if mech == "fetch_sleep":
            s["host_data_wait_ms"] = 6.0
    return d


@pytest.fixture
def campaign(tmp_path):
    arms = {"healthy_ref": [("none", "HEALTHY", {"seed": i}) for i in range(3)],
            "straggler_sleep": [("sleep", "STRAGGLER", {"extra_compute_rank1": 8.0})] * 2,
            "data_fetch_sleep": [("fetch_sleep", "DATA_STALL", {"data_wait": 6.0})] * 2,
            "comm_slow": [("emulated_bandwidth", "COMMUNICATION", {"exposed": 14.0, "busy": 18.0})] * 2}
    runs = []
    for arm, specs in arms.items():
        for i, (mech, cls, kw) in enumerate(specs):
            rid = f"{arm}-L{i}"
            rf = f"results/raw/campaigns/{CID}/{arm}/{rid}.json"
            p = tmp_path / "raw/campaigns" / CID / arm / f"{rid}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            d = _doc(rid, mech, cls, **kw)
            if mech == "emulated_bandwidth":
                for s in d["steps"]:
                    s["injected_comm_delay_ms"] = 7.0
            p.write_text(json.dumps(d), encoding="utf-8")
            runs.append({"arm": arm, "launch": i, "status": "ok", "run_file": rf,
                         "role": "reference" if arm == "healthy_ref" else "heldout"})
    man = tmp_path / "campaigns" / CID / "manifest.json"
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"campaign_id": CID, "synthetic_fixture": True, "runs": runs,
                               "provenance": {"git_sha": "0" * 40},
                               "spec": {"phases": [{"role": "reference", "arms": []}]}}), encoding="utf-8")
    return man


def test_all_eight_plots_render_from_evidence_schema(campaign, tmp_path, monkeypatch):
    ev = P.Evidence(campaign)
    assert ev.synthetic
    out = tmp_path / "plots"
    files = [P.plot_baseline_distribution(ev, out / "1.png"), P.plot_baseline_vs_fault(ev, out / "2.png"),
             P.plot_breakdown(ev, out / "3.png"), P.plot_signatures(ev, out / "4.png"),
             P.plot_rank_skew(ev, out / "5.png")]
    # evaluation JSON from the real evaluator (frozen check bypassed only for this synthetic fixture)
    import diagnose.freeze as FZ
    monkeypatch.setattr(FZ, "check_frozen", lambda p, h: {"frozen_file_sha256": "f" * 64})
    e = evaluate(str(campaign), ["heldout"], False, "x", "y", marginal_arms=("comm_slow",))
    files += [P.plot_confusion(e, out / "7.png", True), P.plot_heldout(e, out / "8.png", True)]
    sel = {"selection": {"s": {"baseline": "b", "primary": "c", "fault_nature": "injected"}},
           "tuning": {"scenarios": {"s": [{"arm": "b", "samples_per_s_per_gpu": 800.0},
                                         {"arm": "c", "samples_per_s_per_gpu": 1000.0}]}}}
    rep = {"campaign_id": "SYNTHETIC", "scenarios": {"s": {
        "fault_nature": "injected", "baseline": {"samples_per_s_per_gpu": 810.0},
        "candidates": [{"after": {"samples_per_s_per_gpu": 990.0}, "verdict": "ACCEPTED"}]}}}
    files.append(P.plot_optimization(sel, rep, out / "6.png", True))
    assert len(files) == 8 and all(Path(f).stat().st_size > 10_000 for f in files)
    # the evaluator saw the marginal arm and kept it out of the headline score
    assert e["scores"]["n"] == 4 and e["scores_marginal_signal"]["n"] == 2   # 6 held-out runs, 2 marginal
    assert not e["did_not_manifest"]


def test_cli_refuses_to_write_synthetic_plots_under_results(campaign):
    r = subprocess.run([sys.executable, "-m", "analysis.plots", "--campaign", str(campaign),
                        "--out", str(ROOT / "results" / "plots" / "SHOULD_NOT_EXIST")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "SYNTHETIC" in r.stderr + r.stdout
    assert not (ROOT / "results" / "plots" / "SHOULD_NOT_EXIST").exists()
