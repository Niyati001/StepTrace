"""Run the baseline on the workload selected by the pilot.

    python scripts/run_baseline.py --pilot results/processed/pilot/<tag>/pilot_summary.json --launches 5
    python scripts/run_baseline.py --point r18_b32_amp --launches 5       # explicit grid point
    python scripts/run_baseline.py --point r18_b32_amp --set workload.dataset=cifar10 --set experiment.name=baseline_cifar10

Baseline settings come from configs/baseline.yaml; the workload override comes
from the chosen pilot grid point in configs/pilot.yaml.
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

from run_experiment import format_summary, launch, write_summary  # noqa: E402
from run_pilot import to_sets  # noqa: E402
from workloads import config as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", help="pilot_summary.json whose selection to use")
    g.add_argument("--point", help="grid point id from configs/pilot.yaml")
    ap.add_argument("--pilot-config", default="configs/pilot.yaml")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--launches", type=int, default=5)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    point = a.point
    if a.pilot:
        sel = json.loads(Path(a.pilot).read_text(encoding="utf-8"))["selection"]
        if not sel["selected"]:
            raise SystemExit(f"pilot selected nothing: {sel['reason']}")
        point = sel["selected"]
    grid = {p["id"]: p for p in yaml.safe_load(Path(a.pilot_config).read_text(encoding="utf-8"))["grid"]}
    sets = to_sets(grid[point].get("override", {})) + [f"experiment.name=baseline_{point}"] + a.set
    print(f"[run_baseline] grid point {point}: {sets}")
    tag = time.strftime("%Y%m%d-%H%M%S")
    paths = launch(a.config, sets, a.launches, a.cpu, 2, tag=tag)
    exp = C.load(a.config, sets)["experiment"]["name"]
    s, out = write_summary(paths, exp, tag)
    print("\n" + format_summary(s))
    print(f"[run_baseline] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
