"""Rule freeze and evaluator integrity guard (CPU only)."""

import hashlib
import json
import zipfile

import pytest

from diagnose import freeze as FZ
from tools import ingest_results as I


def _design_evidence(tmp_path, role="design"):
    rf = "results/raw/campaigns/d-1/arm/r.json"
    man = {"campaign_id": "d-1", "provenance": {"git_sha": "a" * 40, "dirty": False},
           "spec": {"phases": [{"role": role, "arms": [{"id": "arm", "launches": 1}]}]},
           "runs": [{"arm": "arm", "launch": 0, "role": role, "status": "ok", "run_file": rf}]}
    z = tmp_path / "e.zip"
    with zipfile.ZipFile(z, "w") as zz:
        zz.writestr("results/campaigns/d-1/manifest.json", json.dumps(man))
        zz.writestr(rf, "{}")
    dest = tmp_path / "results" / "sessionX"
    assert I.main(["zip", str(z), "--dest", str(dest), "--label", "fixture"]) == 0
    return dest


def test_freeze_writes_immutable_record_and_check_passes(tmp_path):
    ev = _design_evidence(tmp_path)
    out, h = FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                       require_clean_tree=False)
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["design_evidence"][0]["campaign_ids"] == ["d-1"]
    assert h == hashlib.sha256(out.read_bytes()).hexdigest()
    assert FZ.check_frozen(out, h)["rules_sha256"] == rec["rules_sha256"]
    with pytest.raises(FZ.FreezeError, match="immutable"):
        FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                  require_clean_tree=False)


def test_check_refuses_wrong_hash_or_edited_file(tmp_path):
    ev = _design_evidence(tmp_path)
    out, h = FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                       require_clean_tree=False)
    with pytest.raises(FZ.FreezeError, match="hash"):
        FZ.check_frozen(out, "0" * 64)
    rec = json.loads(out.read_text(encoding="utf-8"))
    rec["constants"]["K_ROBUST_Z"] = 3.0          # someone edits the frozen file
    out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(FZ.FreezeError):
        FZ.check_frozen(out, h)


def test_check_refuses_if_rule_files_changed(tmp_path):
    ev = _design_evidence(tmp_path)
    out, h = FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                       require_clean_tree=False)
    rec = json.loads(out.read_text(encoding="utf-8"))
    rec["rules_sha256"] = "f" * 64              # simulate: rules changed after the freeze
    out.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(FZ.FreezeError, match="differ"):
        FZ.check_frozen(out, hashlib.sha256(out.read_bytes()).hexdigest())


def test_freeze_refused_if_heldout_evidence_exists(tmp_path):
    ev = _design_evidence(tmp_path, role="heldout")
    with pytest.raises(FZ.FreezeError, match="held-out evidence already exists"):
        FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                  require_clean_tree=False)


def test_freeze_refused_on_unverified_evidence(tmp_path):
    ev = _design_evidence(tmp_path)
    (ev / "raw/campaigns/d-1/arm/r.json").write_text('{"edited": true}', encoding="utf-8")
    with pytest.raises(FZ.FreezeError, match="failed verification"):
        FZ.freeze("vT", [str(ev)], results_root=tmp_path / "results", out_dir=tmp_path / "frozen",
                  require_clean_tree=False)


def test_evaluator_refuses_heldout_without_frozen_rules(tmp_path):
    from diagnose.evaluate import evaluate
    with pytest.raises(SystemExit, match="requires --frozen-rules"):
        evaluate(str(tmp_path / "m.json"), ["heldout"], False)
    with pytest.raises(SystemExit, match="REFUSED"):
        evaluate(str(tmp_path / "m.json"), ["heldout"], False, str(tmp_path / "nope.json"), "0" * 64)
