"""Measurement-validity gate (run before spending GPU time on the pilot).

    python scripts/validate_measurement.py --point r18_b64_fp32              # GPU, ~3 min
    python scripts/validate_measurement.py --point r18_b64_fp32 --cpu --set workload.model=cnn_tiny

Interleaves launches A B A B ... where
  A = timed comm hook + torch.profiler window (ddp mode only)
  B = no hook (DDP's built-in C++ all-reduce), same workload/seed
and evaluates analysis.validation (event/wall agreement, decomposition,
causality, byte accounting, profiler agreement, hook overhead).
Exit code 0 iff every gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.validation import format_checks, validate  # noqa: E402
from run_experiment import format_summary, launch, write_summary  # noqa: E402
from run_pilot import to_sets  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--point", default="r18_b64_fp32", help="grid point id in --pilot-config")
    ap.add_argument("--pilot-config", default="configs/pilot.yaml")
    ap.add_argument("--pairs", type=int, default=2, help="A/B launch pairs")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    grid = {p["id"]: p for p in yaml.safe_load(Path(a.pilot_config).read_text(encoding="utf-8"))["grid"]}
    exp = f"validation_{a.point}"
    base = to_sets(grid[a.point].get("override", {})) + [
        f"experiment.name={exp}", "experiment.seed=7", "workload.dataset=synthetic",
        f"measurement.warmup_steps={a.warmup}", f"measurement.measured_steps={a.steps}",
        "measurement.repeats=1", "measurement.modes=[ddp]"] + a.set
    arm_a = base + ["measurement.comm_hook=true", "measurement.profile.enabled=true"]
    arm_b = base + ["measurement.comm_hook=false", "measurement.profile.enabled=false"]

    tag = time.strftime("%Y%m%d-%H%M%S")
    on, off = [], []
    for i in range(a.pairs):
        on += launch(None, arm_a + [f"experiment.seed={7 + i}"], 1, a.cpu, 2, tag=f"{tag}-hook-p{i}")
        off += launch(None, arm_b + [f"experiment.seed={7 + i}"], 1, a.cpu, 2, tag=f"{tag}-nohook-p{i}")

    s, out = write_summary(on, exp, tag)
    docs_on = [json.loads(p.read_text(encoding="utf-8")) for p in on]
    docs_off = [json.loads(p.read_text(encoding="utf-8")) for p in off]
    checks = validate(docs_on, docs_off)
    report = format_checks(checks)
    (out / "validation.json").write_text(json.dumps({"point": a.point, "checks": checks,
                                                     "hook_off_runs": [str(p) for p in off]},
                                                    indent=2, default=str), encoding="utf-8")
    (out / "validation.md").write_text(format_summary(s) + "\n" + report + "\n", encoding="utf-8")
    print("\n" + format_summary(s))
    print(report)
    print(f"[validate_measurement] wrote {out}")
    return 0 if all(c["passed"] for c in checks if c["kind"] == "gate") else 1


if __name__ == "__main__":
    raise SystemExit(main())
