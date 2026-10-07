"""Milestone 1 pilot: find a configuration where communication is measurable.

    python scripts/run_pilot.py --config configs/pilot.yaml            # all grid points
    python scripts/run_pilot.py --config configs/pilot.yaml --only r18_b32_amp
    python scripts/run_pilot.py --config configs/pilot.yaml --cpu      # software check only

Each grid point = one launch with interleaved ddp/nosync blocks. The selection
rule (analysis/pilot.py) was fixed before any GPU data was collected.
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

from analysis.pilot import pilot_row, select  # noqa: E402
from run_experiment import launch, write_summary  # noqa: E402
from workloads import config as C  # noqa: E402


def to_sets(d: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += to_sets(v, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k}={json.dumps(v)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", action="append", default=[], help="extra override applied to every grid point")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    spec = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    sel = spec.get("selection", {})
    tag = time.strftime("%Y%m%d-%H%M%S")
    rows = []
    for point in spec["grid"]:
        if a.only and point["id"] not in a.only:
            continue
        sets = to_sets(spec["base"]) + to_sets(point.get("override", {})) + a.set
        sets.append(f"experiment.name=pilot_{point['id']}")
        cfg = C.load(None, sets)
        print(f"\n=== pilot point {point['id']} ===", flush=True)
        try:
            paths = launch(None, sets, 1, a.cpu, 2, tag=tag)
        except SystemExit as e:  # e.g. OOM on a large config: record, keep going
            rows.append({"id": point["id"], "model": cfg["workload"]["model"], "error": str(e),
                         "communication_relevant": False})
            continue
        s, _ = write_summary(paths, cfg["experiment"]["name"], tag)
        rows.append({**pilot_row(point["id"], cfg, s), "override": point.get("override", {})})

    choice = select([r for r in rows if "error" not in r], sel.get("preferred_models", ["resnet18"]),
                    sel.get("target_fraction", 0.25))
    out = ROOT / ("results/dev/processed/pilot" if a.cpu else "results/processed/pilot") / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_summary.json").write_text(json.dumps({"rows": rows, "selection": choice,
                                                        "selection_rule": sel}, indent=2), encoding="utf-8")
    md = table(rows) + f"\n\nSelection: {choice}\n"
    (out / "pilot_summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"[run_pilot] wrote {out}")
    return 0


def table(rows: list[dict]) -> str:
    cols = ["id", "model", "batch_size", "precision", "ddp_step_ms", "nosync_step_ms", "ddp_step_sd_ms",
            "comm_busy_ms", "exposed_timeline_ms", "ablation_exposed_ms", "ablation_fraction",
            "compute_interference_ms", "resolvable", "material", "communication_relevant"]
    L = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['id']} | {r['model']} | ERROR: {r['error']} |")
            continue
        L.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
