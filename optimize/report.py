"""BEFORE -> CHANGE -> AFTER report from a fresh-session VALIDATION campaign.

    python -m optimize.report --validation-manifest <root>/campaigns/m3_validation-<tag>/manifest.json

Per scenario:
  * throughput (samples/s/GPU) and step time for baseline vs selected, per launch; paired
    by launch index (interleaved runs); absolute and % improvement; whether every pair improved;
  * correctness gate (optimize.metrics.correctness) on the identical-seed correctness runs;
    an optimization whose correctness fails is REJECTED regardless of speed;
  * re-diagnosis of baseline and selected runs against the same-session healthy reference
    (current diagnoser; rule sha256 recorded);
  * fault nature: real_config | injected | emulated. Gains on injected/emulated faults are
    reported separately and are never presented as real-hardware improvements.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnose.diagnose import build_reference, diagnose  # noqa: E402
from instrument.evidence import resolve_run_file  # noqa: E402
from optimize.metrics import correctness, throughput  # noqa: E402


def report(manifest_path: str) -> dict:
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    arms = {a["id"]: a for p in man["spec"]["phases"] for a in p["arms"]}
    docs = defaultdict(list)
    for r in sorted(man["runs"], key=lambda r: (r["arm"], r["launch"])):
        if r["status"] == "ok":
            docs[r["arm"]].append(json.loads(resolve_run_file(manifest_path, r["run_file"]).read_text(encoding="utf-8")))
    ref_stats = build_reference(docs["healthy_ref"]) if docs.get("healthy_ref") else None
    scen = defaultdict(dict)
    for arm_id, a in arms.items():
        if "opt" in a and a["id"].startswith("val__"):
            scen[a["opt"]["scenario"]].setdefault(a["opt"]["role"], []).append(arm_id)
    out = {}
    for sc, roles in scen.items():
        base_arm = roles["baseline"][0]
        base_t = [throughput(d) for d in docs[base_arm]]
        res = {"fault_nature": arms[base_arm]["opt"]["fault_nature"], "baseline_arm": base_arm,
               "baseline": _agg(base_t), "candidates": []}
        if ref_stats:
            res["baseline_diagnosis"] = [diagnose(d, ref_stats)["verdict"] for d in docs[base_arm]]
        for sel_arm in roles.get("selected", []):
            t = [throughput(d) for d in docs[sel_arm]]
            n = min(len(t), len(base_t))
            pairs = [t[i]["samples_per_s_per_gpu"] - base_t[i]["samples_per_s_per_gpu"] for i in range(n)]
            b = res["baseline"]["samples_per_s_per_gpu"]
            agg = _agg(t)
            tuning = arms[sel_arm]["opt"]["tuning_arm"]
            corr_c = docs.get(f"corr__{tuning}__c", [])
            corr_a = docs.get(f"corr__{arms[base_arm]['opt']['tuning_arm']}__a", [])
            corr_b = docs.get(f"corr__{arms[base_arm]['opt']['tuning_arm']}__b", [])
            corr = (correctness(corr_a[0], corr_b[0], corr_c[0], arms[sel_arm]["opt"]["semantics_preserving"])
                    if corr_a and corr_b and corr_c else {"passed": None, "reason": "correctness runs missing"})
            c = {"arm": sel_arm, "change": arms[sel_arm].get("set", {}), "after": agg,
                 "semantics_preserving": arms[sel_arm]["opt"]["semantics_preserving"],
                 "abs_improvement_samples_per_s": agg["samples_per_s_per_gpu"] - b,
                 "pct_improvement": 100 * (agg["samples_per_s_per_gpu"] - b) / b,
                 "step_ms_change": agg["step_ms"] - res["baseline"]["step_ms"],
                 "paired_diffs": pairs, "all_pairs_improve": bool(pairs) and all(p > 0 for p in pairs),
                 "correctness": corr}
            c["verdict"] = ("REJECTED (correctness)" if corr.get("passed") is False else
                            "UNVERIFIED (correctness runs missing)" if corr.get("passed") is None else
                            "ACCEPTED" if c["all_pairs_improve"] else "NOT DEMONSTRATED (no consistent gain)")
            if ref_stats:
                c["after_diagnosis"] = [diagnose(d, ref_stats)["verdict"] for d in docs[sel_arm]]
            res["candidates"].append(c)
        out[sc] = res
    return {"manifest": manifest_path, "campaign_id": man["campaign_id"], "scenarios": out}


def _agg(ts):
    return {"n": len(ts), "samples_per_s_per_gpu": float(np.median([t["samples_per_s_per_gpu"] for t in ts])),
            "step_ms": float(np.median([t["step_ms"] for t in ts])),
            "per_launch": [t["samples_per_s_per_gpu"] for t in ts]}


def to_markdown(r: dict) -> str:
    L = [f"# Optimization validation: {r['campaign_id']}", "",
         "Fresh session, new seeds. Gains on `injected`/`emulated` faults are NOT real-hardware gains.", "",
         "| scenario | fault nature | baseline samples/s/GPU | change | after | Δ abs | Δ % | all pairs ↑ | "
         "correctness | verdict | diagnosis before → after |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sc, s in r["scenarios"].items():
        for c in s["candidates"]:
            corr = c["correctness"].get("passed")
            L.append(f"| {sc} | {s['fault_nature']} | {s['baseline']['samples_per_s_per_gpu']:.1f} | "
                     f"`{json.dumps(c['change'])}` | {c['after']['samples_per_s_per_gpu']:.1f} | "
                     f"{c['abs_improvement_samples_per_s']:+.1f} | {c['pct_improvement']:+.1f} | "
                     f"{c['all_pairs_improve']} | {corr} | {c['verdict']} | "
                     f"{s.get('baseline_diagnosis')} → {c.get('after_diagnosis')} |")
    return "\n".join(L) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # reports contain non-ASCII; never crash on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation-manifest", required=True)
    a = ap.parse_args()
    r = report(a.validation_manifest)
    out = Path(a.validation_manifest).parent
    (out / "optimization_report.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    md = to_markdown(r)
    (out / "optimization_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
