"""Select optimizations on a TUNING campaign and generate the fresh-session VALIDATION campaign.

    python -m optimize.search --tuning-manifest <root>/campaigns/m3_tuning-<tag>/manifest.json \
                              --write-validation configs/campaigns/m3_validation.yaml

Tuning arms carry ``opt`` metadata (configs/campaigns/m3_tuning.yaml):
    opt: {scenario, role: baseline|candidate, change, semantics_preserving, fault_nature}
Per scenario the PRIMARY selection is the semantics-preserving candidate with the highest
median throughput (samples/s/GPU). A semantics-changing candidate (e.g. gradient
accumulation) is validated as a separate trade-off only if it beats the primary selection;
it is accepted only if the correctness gate passes.

The generated validation campaign uses NEW seeds and runs in a FRESH session:
  reference   : same-session healthy reference (for re-diagnosis)
  validation  : baseline and selected arms, 3 launches each, interleaved
  correctness : identical seed for baseline x2 (noise floor) and each selected candidate,
                parameter fingerprints enabled
Tuning numbers are never reported as the result; only validation numbers are.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instrument.evidence import resolve_run_file  # noqa: E402
from optimize.metrics import throughput  # noqa: E402

VALIDATION_LAUNCHES = 3


def tuning_table(manifest_path: str) -> dict:
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    arms = {a["id"]: a for p in man["spec"]["phases"] for a in p["arms"]}
    per_arm = defaultdict(list)
    for r in man["runs"]:
        if r["status"] == "ok" and "opt" in arms.get(r["arm"], {}):
            doc = json.loads(resolve_run_file(manifest_path, r["run_file"]).read_text(encoding="utf-8"))
            per_arm[r["arm"]].append(throughput(doc))
    scen = defaultdict(list)
    for arm_id, ts in per_arm.items():
        a = arms[arm_id]
        scen[a["opt"]["scenario"]].append({
            "arm": arm_id, **a["opt"], "n_launches": len(ts),
            "samples_per_s_per_gpu": float(np.median([t["samples_per_s_per_gpu"] for t in ts])),
            "step_ms": float(np.median([t["step_ms"] for t in ts])),
            "launch_values": [t["samples_per_s_per_gpu"] for t in ts]})
    return {"manifest": manifest_path, "campaign_id": man["campaign_id"], "arms": arms,
            "resolved": man.get("resolved", {}), "scenarios": dict(scen)}


def select(table: dict) -> dict:
    out = {}
    for sc, rows in table["scenarios"].items():
        base = [r for r in rows if r["role"] == "baseline"]
        if len(base) != 1:
            raise ValueError(f"scenario {sc}: need exactly one baseline arm")
        cands = [r for r in rows if r["role"] == "candidate"]
        pres = [r for r in cands if r["semantics_preserving"]]
        primary = max(pres, key=lambda r: r["samples_per_s_per_gpu"]) if pres else None
        best_all = max(cands, key=lambda r: r["samples_per_s_per_gpu"]) if cands else None
        tradeoff = best_all if (best_all and not best_all["semantics_preserving"] and
                                (primary is None or best_all["samples_per_s_per_gpu"] >
                                 primary["samples_per_s_per_gpu"])) else None
        b = base[0]["samples_per_s_per_gpu"]
        out[sc] = {"baseline": base[0]["arm"], "primary": primary and primary["arm"],
                   "tradeoff": tradeoff and tradeoff["arm"], "fault_nature": base[0]["fault_nature"],
                   "tuning_gain_pct_primary": (100 * (primary["samples_per_s_per_gpu"] - b) / b) if primary else None,
                   "note": "tuning-set numbers select candidates only; results come from fresh validation"}
    return out


def _resolved_arm(table: dict, arm_id: str) -> dict:
    """Arm definition with fault magnitudes frozen to the values resolved during tuning, so
    the validation session injects exactly the same fault (not re-scaled to a new step)."""
    a = copy.deepcopy(table["arms"][arm_id])
    if a.get("fault"):
        a["fault"] = dict(table["resolved"].get(arm_id, a["fault"]))
    a.pop("opt", None)
    return a


def validation_spec(table: dict, sel: dict, tuning_spec: dict, seed: int = 60000) -> dict:
    val_arms, corr_arms = [], []
    corr_seed = seed + 999
    for sc, s in sel.items():
        chosen = [x for x in (s["primary"], s["tradeoff"]) if x]
        for arm_id, tag in [(s["baseline"], "baseline")] + [(c, "selected") for c in chosen]:
            a = _resolved_arm(table, arm_id)
            meta = {"scenario": sc, "role": tag, "tuning_arm": arm_id, "fault_nature": s["fault_nature"],
                    "semantics_preserving": table["arms"][arm_id]["opt"]["semantics_preserving"]}
            val_arms.append({**a, "id": f"val__{arm_id}", "launches": VALIDATION_LAUNCHES, "opt": meta})
            fp = {"measurement": {"fingerprint": True}}
            cset = copy.deepcopy(a.get("set", {}))
            cset.setdefault("measurement", {}).update(fp["measurement"])
            reps = ["a", "b"] if tag == "baseline" else ["c"]
            for rep in reps:
                corr_arms.append({**a, "set": cset, "id": f"corr__{arm_id}__{rep}", "launches": 1,
                                  "seed": corr_seed, "opt": {**meta, "replica": rep}})
    ref = next(p for p in tuning_spec["phases"] if p["role"] == "reference")
    return {"name": "m3_validation", "seed": seed, "workload_point": tuning_spec["workload_point"],
            "base": tuning_spec["base"], "generated_from": table["campaign_id"], "selection": sel,
            "phases": [copy.deepcopy(ref),
                       {"name": "validation", "role": "validation", "arms": val_arms},
                       {"name": "correctness", "role": "correctness", "arms": corr_arms}]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tuning-manifest", required=True)
    ap.add_argument("--write-validation", required=True)
    a = ap.parse_args()
    table = tuning_table(a.tuning_manifest)
    sel = select(table)
    tuning_spec = json.loads(Path(a.tuning_manifest).read_text(encoding="utf-8"))["spec"]
    spec = validation_spec(table, sel, tuning_spec)
    Path(a.write_validation).write_text(
        "# GENERATED by optimize.search from tuning campaign " + table["campaign_id"] +
        ". Run in a FRESH GPU session. Do not edit by hand.\n" + yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    out = Path(a.tuning_manifest).parent / "optimization_selection.json"
    out.write_text(json.dumps({"selection": sel, "tuning": {k: v for k, v in table.items() if k != "arms"}},
                              indent=2, default=str), encoding="utf-8")
    print(json.dumps(sel, indent=2))
    print(f"wrote {a.write_validation} and {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
