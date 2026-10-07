"""Score the diagnoser against ground truth for a campaign.

    python -m diagnose.evaluate --manifest results/campaigns/<id>/manifest.json --roles design --loo-reference
    python -m diagnose.evaluate --manifest <root>/campaigns/<id>/manifest.json --roles heldout         --frozen-rules diagnose/frozen/rules_v1.json --frozen-rules-sha256 <sha256>         [--marginal-arms A B]

* Diagnosis first, from the observable view only; ground truth is read afterwards,
  only for scoring.
* Reference = the campaign's own healthy reference runs (same session). With
  --loo-reference each reference run is also scored as a HEALTHY sample against
  the other reference runs (leave-one-run-out).
* Held-out roles REQUIRE --frozen-rules and its exact --frozen-rules-sha256; scoring is
  refused if the file hash differs or the current rule/manifestation files changed since
  the freeze (diagnose.freeze.check_frozen).
* Manifestation is decided per run by faults/manifestation.py (mechanism-level, independent
  of the diagnoser). DID_NOT_MANIFEST runs are listed separately and never counted as misses.
* --marginal-arms (DECISIONS.md, fixed before execution): diagnosed and listed, reported as
  a separate category, EXCLUDED from the headline score.
Outputs evaluation_<name>.json / .md next to the manifest.
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

from diagnose.diagnose import build_reference, diagnose, frozen_status  # noqa: E402
from diagnose.features import FEATURES, block_features, observable_view  # noqa: E402
from faults import manifestation as MF  # noqa: E402
from faults.spec import CLASSES  # noqa: E402
from instrument.evidence import resolve_run_file  # noqa: E402


def score(truth: list[str], pred: list[str]) -> dict:
    """Confusion matrix and per-class precision/recall (rows = truth, columns = predicted).

    Plain numpy, identical in definition to sklearn.metrics.confusion_matrix /
    precision_recall_fscore_support(zero_division=0) and cross-checked against them in
    tests/test_evaluate.py wherever scikit-learn is importable.
    """
    labels = list(CLASSES) + (["NO_DATA"] if "NO_DATA" in pred else [])
    idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(truth, pred):
        cm[idx[t], idx[p]] += 1
    per = {}
    for i, lab in enumerate(labels):
        tp, col, row = cm[i, i], cm[:, i].sum(), cm[i, :].sum()
        prec = tp / col if col else 0.0
        rec = tp / row if row else 0.0
        per[lab] = {"precision": float(prec), "recall": float(rec),
                    "f1": float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0,
                    "support": int(row), "predicted": int(col)}
    return {"labels": labels, "confusion_matrix": cm.tolist(),
            "accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0,
            "per_class": per, "n": int(cm.sum())}


HELDOUT_ROLES = {"heldout"}


def evaluate(manifest_path: str, roles: list[str], loo_reference: bool, frozen_rules: str | None = None,
             frozen_rules_sha256: str | None = None, marginal_arms: tuple = ()) -> dict:
    status = frozen_status()
    frozen_check = None
    if set(roles) & HELDOUT_ROLES or frozen_rules or frozen_rules_sha256:
        if not (frozen_rules and frozen_rules_sha256):
            raise SystemExit("held-out scoring requires --frozen-rules and --frozen-rules-sha256")
        from diagnose.freeze import FreezeError, check_frozen
        try:
            frozen_check = check_frozen(frozen_rules, frozen_rules_sha256)
        except FreezeError as e:
            raise SystemExit(f"REFUSED: {e}")
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ok = [r for r in man["runs"] if r["status"] == "ok"]
    load = lambda r: json.loads(resolve_run_file(manifest_path, r["run_file"]).read_text(encoding="utf-8"))  # noqa: E731
    ref_runs = [r for r in ok if r["role"] == "reference"]
    ref_docs = [load(r) for r in ref_runs]
    full_ref = build_reference(ref_docs)
    ref_profile = MF.reference_profile(ref_docs)
    host_ctrl = [load(r) for r in ok if r["arm"] == "healthy_hostloader"]
    host_base = (float(np.median([s["host_data_wait_ms"] for d in host_ctrl for s in d["steps"]
                                  if s["mode"] == "ddp"])) if host_ctrl else None)
    host_profile = MF.reference_profile(host_ctrl) if host_ctrl else None

    samples = []
    if loo_reference:
        for r, d in zip(ref_runs, ref_docs):
            samples.append((r, d, build_reference(ref_docs, exclude_run=d["run_id"])))
    for r in ok:
        if r["role"] in roles:
            samples.append((r, load(r), full_ref))

    rows, details = [], []
    for r, doc, stats in samples:
        res = diagnose(doc, stats)                       # observable data only
        truth = doc["ground_truth"]["fault_class"]       # read only now, for scoring
        feats = block_features(observable_view(doc))
        mres = MF.check(doc, ref_profile, host_base, host_profile)
        category = ("did_not_manifest" if mres["status"] == "DID_NOT_MANIFEST" else
                    "marginal_signal" if r["arm"] in marginal_arms else "headline")
        rows.append({"arm": r["arm"], "role": r["role"], "launch": r["launch"], "run_id": doc["run_id"],
                     "category": category, "manifestation": mres,
                     "mechanism": doc["ground_truth"]["fault_mechanism"], "level": doc["ground_truth"]["level"],
                     "emulated": doc["ground_truth"]["emulated"], "truth": truth,
                     "predicted": res["verdict"], "correct": res["verdict"] == truth,
                     "agreement": res["agreement"], "reason": res["reason"],
                     "feature_medians": {k: float(np.median([b[k] for b in feats])) for k in FEATURES}})
        details.append(res)
    failed = [r for r in man["runs"] if r["status"] != "ok" and r["role"] in roles]
    out = {"manifest": manifest_path, "campaign_id": man["campaign_id"], "roles": roles,
           "loo_reference": loo_reference, "rules": status, "reference": {
               "n_runs": len(ref_docs), "n_blocks": full_ref["_n_blocks"],
               "step_ms_center": full_ref["step_ms"]["center"], "practical_ms": full_ref["_practical_ms"],
               "features": {k: full_ref[k] for k in FEATURES}},
           "frozen_check": frozen_check,
           "scores": score([x["truth"] for x in rows if x["category"] == "headline"],
                           [x["predicted"] for x in rows if x["category"] == "headline"]),
           "scores_marginal_signal": score([x["truth"] for x in rows if x["category"] == "marginal_signal"],
                                           [x["predicted"] for x in rows if x["category"] == "marginal_signal"]),
           "did_not_manifest": [x for x in rows if x["category"] == "did_not_manifest"],
           "rows": rows, "misses": [x for x in rows if not x["correct"] and x["category"] == "headline"],
           "failed_launches": failed, "diagnoses": details}
    out["signatures"] = signatures(rows, full_ref)
    return out


def signatures(rows: list[dict], ref: dict) -> list[dict]:
    """Per arm: median feature delta vs the healthy reference (what each fault looks like)."""
    by = defaultdict(list)
    for x in rows:
        by[x["arm"]].append(x)
    out = []
    for arm, xs in by.items():
        out.append({"arm": arm, "truth": xs[0]["truth"], "mechanism": xs[0]["mechanism"], "n": len(xs),
                    **{f"d_{k}": float(np.median([x["feature_medians"][k] for x in xs])) - ref[k]["center"]
                       for k in FEATURES}})
    return out


def to_markdown(e: dict, title: str) -> str:
    s = e["scores"]
    L = [f"# {title}", "", f"Campaign `{e['campaign_id']}`; roles {e['roles']}; "
         f"rules sha256 `{e['rules']['rules_sha256'][:16]}` (frozen: {e['rules']['frozen']}, "
         f"matches: {e['rules'].get('matches')})", "",
         f"Reference: {e['reference']['n_runs']} healthy runs / {e['reference']['n_blocks']} blocks, "
         f"median step {e['reference']['step_ms_center']:.2f} ms, practical floor "
         f"{e['reference']['practical_ms']:.2f} ms", "",
         f"**Headline accuracy {s['accuracy']:.3f} on n = {s['n']}** "
         f"(marginal-signal and did-not-manifest arms excluded; reported below)", "",
         "Confusion matrix (rows = truth, columns = predicted):", "",
         "| truth \\ predicted | " + " | ".join(s["labels"]) + " |", "|---" * (len(s["labels"]) + 1) + "|"]
    for lab, row in zip(s["labels"], s["confusion_matrix"]):
        L.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")
    L += ["", "| class | precision | recall | support |", "|---|---|---|---|"]
    for lab, m in s["per_class"].items():
        L.append(f"| {lab} | {m['precision']:.2f} | {m['recall']:.2f} | {m['support']} |")
    m = e["scores_marginal_signal"]
    L += ["", f"Marginal-signal category (excluded from headline): n = {m['n']}, accuracy "
              f"{m['accuracy']:.3f}" if m["n"] else "", f"Did not manifest (not counted as misses): "
              f"{sorted({x['arm'] for x in e['did_not_manifest']}) or 'none'}"]
    L += ["", "Per run:", "", "| arm | category | mechanism | level | truth | predicted | agreement | reason |",
          "|---|---|---|---|---|---|---|---|"]
    for x in e["rows"]:
        mark = "" if x["correct"] or x["category"] != "headline" else " **MISS**"
        L.append(f"| {x['arm']} | {x['category']} | {x['mechanism']}{' (emulated)' if x['emulated'] else ''} | "
                 f"{x['level']} | {x['truth']} | {x['predicted']}{mark} | {x['agreement']} | {x['reason']} |")
    L += ["", "Signatures (median feature delta vs healthy reference):", "",
          "| arm | truth | d step ms | d data wait ms | d compute skew ms | d exposed(min) ms | "
          "d exposed frac | d comm busy ms | d GB/s |", "|---|---|---|---|---|---|---|---|---|"]
    for g in e["signatures"]:
        L.append(f"| {g['arm']} | {g['truth']} | {g['d_step_ms']:+.2f} | {g['d_data_wait_ms']:+.2f} | "
                 f"{g['d_compute_skew_ms']:+.2f} | {g['d_exposed_min_ms']:+.2f} | {g['d_exposed_min_frac']:+.3f} | "
                 f"{g['d_comm_busy_ms']:+.2f} | {g['d_comm_GBps']:+.2f} |")
    if e["failed_launches"]:
        L += ["", "Failed launches (not scored):"] + [f"* {r['arm']} L{r['launch']}: {r.get('error')}"
                                                     for r in e["failed_launches"]]
    return "\n".join(L) + "\n"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # reports contain non-ASCII; never crash on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--roles", nargs="+", required=True)
    ap.add_argument("--loo-reference", action="store_true")
    ap.add_argument("--frozen-rules")
    ap.add_argument("--frozen-rules-sha256")
    ap.add_argument("--marginal-arms", nargs="*", default=[])

    ap.add_argument("--name", default=None)
    ap.add_argument("--out", help="output directory (default: next to the manifest)")
    a = ap.parse_args()
    e = evaluate(a.manifest, a.roles, a.loo_reference, a.frozen_rules, a.frozen_rules_sha256,
                 tuple(a.marginal_arms))
    name = a.name or "_".join(a.roles)
    out = Path(a.out) if a.out else Path(a.manifest).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / f"evaluation_{name}.json").write_text(json.dumps(e, indent=2, default=str), encoding="utf-8")
    md = to_markdown(e, f"Diagnosis evaluation: {name}")
    (out / f"evaluation_{name}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
