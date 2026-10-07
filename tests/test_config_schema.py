from pathlib import Path

import pytest
import yaml

from instrument.schema import validate_step
from workloads import config as C
from fixtures import step

ROOT = Path(__file__).resolve().parents[1]


def test_defaults_valid():
    assert C.validate(C.load()) == []


def test_override_parsing_and_merge():
    cfg = C.load(None, ["workload.batch_size=32", "measurement.modes=[ddp]", "workload.precision=amp"])
    assert cfg["workload"]["batch_size"] == 32 and cfg["measurement"]["modes"] == ["ddp"]
    assert cfg["distributed"]["bucket_cap_mb"] == 25  # untouched default kept


@pytest.mark.parametrize("bad", ["workload.model=resnet9000", "workload.batch_size=0",
                                 "measurement.modes=[allreduce]", "workload.typo=1",
                                 "measurement.comm_hook=maybe"])
def test_invalid_rejected(bad):
    with pytest.raises(ValueError):
        C.load(None, [bad])


def test_repo_configs_load():
    C.load(ROOT / "configs/baseline.yaml")
    spec = yaml.safe_load((ROOT / "configs/pilot.yaml").read_text(encoding="utf-8"))
    ids = [p["id"] for p in spec["grid"]]
    assert len(ids) == len(set(ids))
    for p in spec["grid"]:
        merged = C.deep_merge(C.deep_merge(C.DEFAULTS, spec["base"]), p["override"])
        assert C.validate(merged) == [], p["id"]


def test_schema_accepts_fixture_and_rejects_bad():
    assert validate_step(step("r", 0, 0, "ddp", 0, 0)) == []
    s = step("r", 0, 0, "nosync", 0, 0)
    assert validate_step(s) == []
    s["communication_bytes"] = 5
    assert any("nosync" in e for e in validate_step(s))
    s = step("r", 0, 0, "ddp", 0, 0)
    del s["step_time_ms"]
    assert validate_step(s)
