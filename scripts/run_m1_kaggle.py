"""Milestone 1 GPU pipeline with gates (designed for Kaggle 2x T4; works on any >=2-GPU host).

    python scripts/run_m1_kaggle.py                 # full: ~30-40 min GPU
    python scripts/run_m1_kaggle.py --stages A B    # only some stages

Stages (each later stage requires the earlier gates to pass):
  A  environment detection                                   (seconds)
  B  measurement validation on r18_b64_fp32 (GATE)           (~3 min)
  C  pilot grid, pre-registered selection rule               (~10-15 min)
  D  measurement validation on the selected point (GATE)     (~3 min)
  E  baseline: 5 launches x 2 repeats x {ddp,nosync}, profiler (~10 min)
  F  CIFAR-10 input-pipeline run on the selected point       (~2 min, needs internet)
  G  plots
A JSON log of every stage (command, exit code, wall time, output paths) is
written to results/processed/m1_pipeline_<tag>.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def newest(pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


class Pipeline:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.log: list[dict] = []
        self.path = ROOT / "results/processed" / f"m1_pipeline_{tag}.json"

    def run(self, stage: str, cmd: list[str], **extra) -> int:
        print(f"\n######## stage {stage}: {' '.join(cmd)}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        self.log.append({"stage": stage, "cmd": cmd, "exit_code": rc,
                         "wall_seconds": round(time.time() - t0, 1), **extra})
        self.save()
        print(f"######## stage {stage}: exit {rc} in {time.time() - t0:.0f}s", flush=True)
        return rc

    def note(self, stage: str, **kw) -> None:
        self.log.append({"stage": stage, **kw})
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"tag": self.tag, "stages": self.log}, indent=2, default=str), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="*", default=list("ABCDEFG"))
    ap.add_argument("--validation-point", default="r18_b64_fp32")
    ap.add_argument("--baseline-launches", type=int, default=5)
    ap.add_argument("--point", help="skip pilot selection and use this grid point")
    a = ap.parse_args()
    st = set(a.stages)
    P = Pipeline(time.strftime("%Y%m%d-%H%M%S"))

    if "A" in st:
        P.run("A", [PY, "scripts/detect_environment.py", "--out", f"results/raw/env/env_m1_{P.tag}.json"])
        import torch

        if torch.cuda.device_count() < 2:
            P.note("A", verdict="STOP: fewer than 2 GPUs; multi-GPU DDP/NCCL experiments unavailable")
            print("Fewer than 2 GPUs: stop (spec stop condition B).")
            return 2

    if "B" in st and P.run("B", [PY, "scripts/validate_measurement.py", "--point", a.validation_point]):
        P.note("B", verdict="STOP: measurement validation gate failed; do not run the pilot")
        return 3

    point = a.point
    pilot_dir = None
    if "C" not in st and point is None and st & set("DEF"):
        ap.error("stages D/E/F without C need --point <selected grid point id>")
    if "C" in st:
        P.run("C", [PY, "scripts/run_pilot.py", "--config", "configs/pilot.yaml"])
        pilot_dir = newest("results/processed/pilot/*")
        sel = json.loads((pilot_dir / "pilot_summary.json").read_text(encoding="utf-8"))["selection"]
        P.note("C", pilot_dir=str(pilot_dir), selection=sel)
        point = point or sel["selected"]
    if point is None:
        P.note("C", verdict="STOP: no communication-relevant configuration (spec stop condition A)")
        if "G" in st and pilot_dir:
            P.run("G", [PY, "scripts/plot_m1.py", "--pilot", str(pilot_dir)])
        return 4

    if "D" in st and point != a.validation_point:
        if P.run("D", [PY, "scripts/validate_measurement.py", "--point", point]):
            P.note("D", verdict=f"STOP: validation gate failed on selected point {point}")
            return 5

    baseline_dir = None
    if "E" in st:
        P.run("E", [PY, "scripts/run_baseline.py", "--point", point, "--launches", str(a.baseline_launches)])
        baseline_dir = newest(f"results/processed/baseline_{point}/*")
        P.note("E", baseline_dir=str(baseline_dir))

    if "F" in st:
        # Acquire + verify CIFAR-10 in a single process first; never inside the 2-rank job.
        if P.run("F", [PY, "scripts/prepare_data.py", "--root", "data"], step="prepare_cifar10"):
            P.note("F", verdict="CIFAR-10 unavailable (see prepare_data output); input-pipeline run skipped")
            print(f"\nPipeline log: {P.path}")
            return 0
        rc = P.run("F", [PY, "scripts/run_baseline.py", "--point", point, "--launches", "1",
                         "--set", f"experiment.name=cifar10_{point}", "--set", "workload.dataset=cifar10",
                         "--set", "measurement.repeats=1", "--set", "measurement.modes=[ddp]",
                         "--set", "measurement.profile.enabled=false"])
        if rc:
            P.note("F", verdict="CIFAR-10 run failed (internet disabled?); input-pipeline wait not measured")

    if "G" in st:
        cmd = [PY, "scripts/plot_m1.py"]
        if baseline_dir:
            cmd += ["--baseline", str(baseline_dir)]
        if pilot_dir:
            cmd += ["--pilot", str(pilot_dir)]
        P.run("G", cmd)
    print(f"\nPipeline log: {P.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
