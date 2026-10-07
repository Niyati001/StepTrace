"""StepTrace diagnoser: explicit rules over measured evidence.

    python -m diagnose.diagnose --reference results/raw/<campaign>/healthy_ref/*.json \
                                --candidate results/raw/<campaign>/<arm>/<run>.json [--json out.json]

Trace for every verdict: raw per-step measurements -> cluster-step features ->
block medians -> robust z vs same-session healthy reference -> rule -> verdict.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnose import thresholds as T  # noqa: E402
from diagnose.features import block_features, observable_view  # noqa: E402
from diagnose.rules import classify_run  # noqa: E402

RULE_FILES = ("diagnose/features.py", "diagnose/thresholds.py", "diagnose/rules.py")
FROZEN_DIR = ROOT / "diagnose" / "frozen"


def rules_fingerprint() -> dict:
    h = {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in RULE_FILES}
    combined = hashlib.sha256("".join(h[f] for f in RULE_FILES).encode()).hexdigest()
    return {"files": h, "sha256": combined}


def frozen_status() -> dict:
    """Informational: do the current rule files match the newest frozen record, if any?
    (Enforcement happens in diagnose.evaluate via diagnose.freeze.check_frozen.)"""
    fp = rules_fingerprint()
    recs = sorted(FROZEN_DIR.glob("rules_*.json"))
    if not recs:
        return {"frozen": False, "matches": None, "rules_sha256": fp["sha256"]}
    frozen = json.loads(recs[-1].read_text(encoding="utf-8"))
    return {"frozen": True, "matches": frozen["rules_sha256"] == fp["sha256"],
            "rules_sha256": fp["sha256"], "frozen_sha256": frozen["rules_sha256"],
            "frozen_file": recs[-1].relative_to(ROOT).as_posix()}


def build_reference(ref_docs: list[dict], exclude_run: str | None = None) -> dict:
    blocks = [b for d in ref_docs if d["run_id"] != exclude_run
              for b in block_features(observable_view(d))]
    return T.reference_stats(blocks)


def diagnose(doc: dict, ref_stats: dict) -> dict:
    obs = observable_view(doc)  # the diagnoser never sees ground truth or config
    blocks = block_features(obs)
    if not blocks:
        return {"run_id": doc["run_id"], "verdict": "NO_DATA", "healthy": None, "agreement": "0/0",
                "reason": "no ddp-mode blocks with measured communication"}
    res = classify_run(blocks, ref_stats)
    res["run_id"] = doc["run_id"]
    res["reference"] = {"n_blocks": ref_stats["_n_blocks"], "runs": ref_stats["_runs"],
                        "step_ms_center": ref_stats["step_ms"]["center"],
                        "practical_ms": ref_stats["_practical_ms"]}
    res["rules"] = frozen_status()
    res["reason"] = explain(res)
    return res


def explain(res: dict) -> str:
    v = res["verdict"]
    # evidence = median over blocks of the primary test(s) for the verdict
    if v == "HEALTHY":
        return (f"no cause elevated in {res['agreement']} blocks: every primary feature stayed below "
                f"z >= {T.K_ROBUST_Z} or below the practical floor")
    tests = [t for b in res["blocks"] if b["verdict"] == v for t in b["causes"][v]["tests"] if t["elevated"]]
    t = sorted(tests, key=lambda t: t["z"])[len(tests) // 2]
    unit = "" if t["feature"].endswith("_frac") else " ms"
    return (f"{t['feature']} = {t['value']:.4g}{unit} vs healthy {t['reference_center']:.4g}{unit} "
            f"(delta {t['delta']:+.4g}{unit}, robust z {t['z']:.1f} >= {T.K_ROBUST_Z}, "
            f"practical floor {t['practical_threshold']:.3g}{unit}) in {res['agreement']} blocks")


def format_report(res: dict) -> str:
    L = ["StepTrace Diagnosis", "===================", "",
         f"Run: {res['run_id']}",
         f"Primary cause: {res['verdict']}" + ("" if res["healthy"] else "   (unhealthy)"),
         f"Reason: {res['reason']}", ""]
    if res["verdict"] == "NO_DATA":
        return "\n".join(L)
    b0 = res["blocks"]
    L.append("Evidence (median over blocks; delta vs same-session healthy reference):")
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    for cause in ("COMMUNICATION", "STRAGGLER", "DATA_STALL"):
        n_el = sum(b["causes"][cause]["elevated"] for b in b0)
        for i, t in enumerate(b0[0]["causes"][cause]["tests"]):
            per_block = [b["causes"][cause]["tests"][i] for b in b0]
            L.append(f"  {cause:13s} {t['feature']:17s} delta {med(x['delta'] for x in per_block):+9.4g}  "
                     f"z {med(x['z'] for x in per_block):7.1f}  cause elevated in {n_el}/{len(b0)} blocks")
    L.append("Supporting metrics (change vs reference, median over blocks):")
    for k in b0[0]["supporting"]:
        ch = sorted(b["supporting"][k]["change_pct"] for b in b0 if b["supporting"][k]["change_pct"] is not None)
        if ch:
            L.append(f"  {k:13s} {ch[len(ch) // 2]:+7.1f} %")
    L += ["", "Confidence definition:",
          f"  Rule agreement across repeated blocks: {res['agreement']}  (votes: {res['block_votes']})",
          f"Thresholds: robust z >= {T.K_ROBUST_Z}; practical floor {res['reference']['practical_ms']:.3g} ms "
          f"({T.PRACTICAL_STEP_FRACTION:.0%} of healthy step {res['reference']['step_ms_center']:.4g} ms) "
          f"or {T.PRACTICAL_FRACTION_PP:.0%} pp for fractions",
          f"Reference: {res['reference']['n_blocks']} healthy blocks from {len(res['reference']['runs'])} runs",
          f"Rules sha256: {res['rules']['rules_sha256'][:16]}  frozen: {res['rules']['frozen']}"
          + (f"  matches frozen: {res['rules']['matches']}" if res["rules"]["frozen"] else "")]
    return "\n".join(L)


def _load(patterns: list[str]) -> list[dict]:
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit(f"no files match {patterns}")
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # reports contain non-ASCII; never crash on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", nargs="+", required=True, help="healthy reference run JSON files/globs")
    ap.add_argument("--candidate", nargs="+", required=True, help="run JSON file(s)/globs to diagnose")
    ap.add_argument("--json", help="write machine-readable results here")
    a = ap.parse_args()
    refs = _load(a.reference)
    out = []
    for doc in _load(a.candidate):
        stats = build_reference(refs, exclude_run=doc["run_id"])  # never compare a run against itself
        res = diagnose(doc, stats)
        print(format_report(res) + "\n")
        out.append(res)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
