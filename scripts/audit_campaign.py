"""Evidence audit of a design campaign BEFORE any rule freeze.

    python scripts/audit_campaign.py --manifest results/campaigns/<id>/manifest.json

Reads only the manifest and raw run files. Never modifies rules. Outputs
audit.json / audit.md next to the manifest. Ground truth / injector parameters
are used here ONLY to check whether each fault manifested as intended (the
auditor's job); the diagnoser never sees them.

Decision criteria (fixed before looking at any GPU data):
  healthy range of a feature = [min, max] over healthy reference blocks.
  separated  = every fault block lies outside the healthy range on the intended side.
  realized   = median(fault feature) - median(healthy feature)  (intended observable)
  intended   = sleep, fetch_sleep: delay_ms;  loader_sleep: delay_ms/num_workers - reference step.
  VALID FOR RULE DESIGN  : intended observable separated in all blocks, realized/intended
                           in [0.6, 1.4] where an intended amount exists, mechanism-specific
                           structure check passes, no off-target confound.
  WEAK / NEEDS REDESIGN  : manifests but overlaps healthy, or realized/intended outside
                           [0.6, 1.4], or an off-target class feature also leaves its
                           healthy range by more than the practical floor (confound).
  INVALID FAULT          : intended observable not separated in any block, or the
                           mechanism did not change the observable structure
                           (e.g. identical bucket structure for single_bucket).
  healthy arms           : VALID if every primary feature stays within the healthy
                           range +- practical floor; otherwise WEAK (systematic shift).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.summary import cluster_steps, summarize  # noqa: E402
from analysis.validation import validate  # noqa: E402
from diagnose import thresholds as T  # noqa: E402
from diagnose.diagnose import build_reference, diagnose, rules_fingerprint  # noqa: E402
from diagnose.features import FEATURES, block_features, observable_view  # noqa: E402
from instrument.evidence import resolve_run_file  # noqa: E402

PRIMARY = {"DATA_STALL": "data_wait_ms", "STRAGGLER": "compute_skew_ms", "COMMUNICATION": "exposed_min_ms"}
RATIO_BAND = (0.6, 1.4)


def load(manifest_path):
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    runs = []
    for r in man["runs"]:
        doc = json.loads(resolve_run_file(manifest_path, r["run_file"]).read_text(encoding="utf-8")) if r["status"] == "ok" else None
        runs.append((r, doc))
    return man, runs


def q(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "mad": float(np.median(np.abs(v - np.median(v)))), "n": int(v.size),
            "cv_pct": float(100 * v.std(ddof=1) / v.mean()) if v.size > 1 and v.mean() else None}


# --------------------------------------------------------------------------- 6. integrity
def integrity(man, runs) -> dict:
    expected = sum(a["launches"] for p in man["spec"]["phases"] for a in p["arms"])
    sha = man["provenance"]["git_sha"]
    probs, seed_meta = [], []
    ids, seeds = Counter(), Counter()
    for r, d in runs:
        if r["status"] != "ok":
            probs.append(f"{r['arm']} L{r['launch']}: status {r['status']} ({r.get('error')})")
            continue
        p = d.get("provenance") or {}
        ids[d["run_id"]] += 1
        if p.get("git_sha") != sha:
            probs.append(f"{d['run_id']}: git_sha {p.get('git_sha')} != campaign {sha}")
        if p.get("dirty") or p.get("allow_dirty_override"):
            probs.append(f"{d['run_id']}: dirty={p.get('dirty')} override={p.get('allow_dirty_override')}")
        for k in ("config", "effective_config", "environment", "ground_truth", "timestamp_utc", "started_utc"):
            if not d.get(k):
                probs.append(f"{d['run_id']}: missing {k}")
        if d.get("schema_errors"):
            probs.append(f"{d['run_id']}: schema errors {d['schema_errors'][:3]}")
        want = (r.get("fault") or {}).get("mechanism", "none")
        if d["ground_truth"]["fault_mechanism"] != want:
            probs.append(f"{d['run_id']}: ground truth {d['ground_truth']['fault_mechanism']} != manifest {want}")
        actual_seed = d["config"]["experiment"]["seed"]
        if actual_seed != r["seed"]:
            if actual_seed == r["seed"] + r["launch"]:
                # known run_campaign defect (fixed later): launch() re-applied seed = base + launch index.
                # The run document holds the TRUE seed; the manifest field is wrong.
                seed_meta.append({"run_id": d["run_id"], "manifest_seed": r["seed"], "actual_seed": actual_seed})
            else:
                probs.append(f"{d['run_id']}: seed {actual_seed} != manifest {r['seed']} (unexplained)")
        seeds[actual_seed] += 1
    probs += [f"duplicate run_id {k}" for k, n in ids.items() if n > 1]
    probs += [f"seed {k} used by {n} runs" for k, n in seeds.items() if n > 1]
    return {"expected_launches": expected, "manifest_launches": len(runs),
            "ok": sum(r["status"] == "ok" for r, _ in runs),
            "failed": [r for r, _ in runs if r["status"] != "ok"],
            "campaign_git_sha": sha, "campaign_dirty": man["provenance"]["dirty"],
            "campaign_override": man["provenance"]["allow_dirty_override"],
            "problems": probs, "seed_metadata_defect": seed_meta,
            "pass": not probs and len(runs) == expected}


# --------------------------------------------------------------------------- 1. reference
def reference_audit(ref_docs) -> dict:
    per_run = []
    for d in ref_docs:
        bf = block_features(observable_view(d))
        cs = [c for c in cluster_steps(d["steps"]) if c["mode"] == "ddp"]
        gates = validate([d])
        per_run.append({
            "run_id": d["run_id"], "n_blocks": len(bf),
            **{k: float(np.median([b[k] for b in bf])) for k in FEATURES},
            "exposed_timeline_ms": float(np.median([c["exposed_communication_time_ms"] for c in cs])),
            "rank_start_skew_ms": float(np.median([c["rank_start_skew_ms"] for c in cs])),
            "gates_pass": all(c["passed"] for c in gates if c["kind"] == "gate"),
            "gate_failures": [c["name"] for c in gates if c["kind"] == "gate" and not c["passed"]],
            "gpu_state": d.get("gpu_state", [])[:1],
        })
    blocks = [b for d in ref_docs for b in block_features(observable_view(d))]
    s = summarize(ref_docs)
    # bootstrap over runs: how uncertain is the z-boundary (center + K*scale) itself?
    rng = random.Random(0)
    boundary = defaultdict(list)
    for _ in range(1000):
        pick = [rng.choice(ref_docs) for _ in ref_docs]
        st = T.reference_stats([b for d in pick for b in block_features(observable_view(d))])
        for k in PRIMARY.values():
            boundary[k].append(st[k]["center"] + T.K_ROBUST_Z * st[k]["scale"])
    full = T.reference_stats(blocks)
    suff = {k: {"boundary_p5": float(np.percentile(v, 5)), "boundary_p95": float(np.percentile(v, 95)),
                "width_ms": float(np.percentile(v, 95) - np.percentile(v, 5)),
                "practical_floor_ms": full["_practical_ms"],
                "width_below_floor": float(np.percentile(v, 95) - np.percentile(v, 5)) < full["_practical_ms"]}
            for k, v in boundary.items()}
    return {"per_run": per_run,
            "across_runs": {k: q([r[k] for r in per_run]) for k in
                            list(FEATURES) + ["exposed_timeline_ms", "rank_start_skew_ms"]},
            "blocks": {k: q([b[k] for b in blocks]) for k in FEATURES},
            "ablation_exposed_ms": s.get("ablation_exposed_ms"),
            "compute_interference_ms": s.get("compute_interference_ms"),
            "gpu_state": s.get("gpu_state"), "nccl": s.get("nccl"),
            "all_gates_pass": all(r["gates_pass"] for r in per_run),
            "boundary_bootstrap": suff, "stats": full}


# --------------------------------------------------------------------------- 2/4/5/7 per arm
def arm_audit(arm, docs, man, ref_stats, ref_blocks, ref_docs) -> dict:
    fault = (man["resolved"].get(arm) or {"mechanism": "none"})
    mech = fault["mechanism"]
    blocks = [b for d in docs for b in block_features(observable_view(d))]
    hr = {k: (min(b[k] for b in ref_blocks), max(b[k] for b in ref_blocks)) for k in FEATURES}
    med = {k: float(np.median([b[k] for b in blocks])) for k in FEATURES}
    delta = {k: med[k] - ref_stats[k]["center"] for k in FEATURES}
    floor = ref_stats["_practical_ms"]
    steps = [s for d in docs for s in d["steps"] if s["mode"] == "ddp"]
    ref_steps = [s for d in ref_docs for s in d["steps"] if s["mode"] == "ddp"]
    bucket_obs = sorted({s["bucket_count"] for s in steps})          # distinct counts, not frequencies
    bucket_ref = sorted({s["bucket_count"] for s in ref_steps})
    layout = [[b["bytes"] for b in (d.get("bucket_layout_first_measured_step_rank0") or [])] for d in docs]
    ref_layout = [[b["bytes"] for b in (d.get("bucket_layout_first_measured_step_rank0") or [])] for d in ref_docs]
    bytes_obs = sorted({s["communication_bytes"] for s in steps})
    bytes_ref = sorted({s["communication_bytes"] for s in ref_steps})
    out = {"arm": arm, "mechanism": mech, "class": docs[0]["ground_truth"]["fault_class"], "fault": fault,
           "n_runs": len(docs), "n_blocks": len(blocks), "median": med, "delta": delta,
           "bucket_count_observed": bucket_obs, "bucket_count_healthy": bucket_ref,
           "bucket_bytes_observed": layout, "bucket_bytes_healthy": ref_layout[:1],
           "comm_bytes_observed": bytes_obs, "comm_bytes_healthy": bytes_ref,
           "effective_batch": sorted({d["effective_config"]["workload"]["batch_size"] for d in docs}),
           "effective_bucket_cap_mb": sorted({d["effective_config"]["distributed"]["bucket_cap_mb"] for d in docs})}

    def separation(feat, side=+1):
        vals = [b[feat] for b in blocks]
        lo, hi = hr[feat]
        outside = sum((v > hi) if side > 0 else (v < lo) for v in vals)
        inside = sum(lo <= v <= hi for v in vals)
        return {"feature": feat, "healthy_range": [lo, hi], "fault_range": [min(vals), max(vals)],
                "blocks_outside_on_intended_side": outside, "blocks_inside_healthy_range": inside,
                "n_blocks": len(vals), "separated_all": outside == len(vals), "separated_none": outside == 0}

    # off-target confounds: other classes' primary features leaving healthy range by > floor
    def offtarget(intended_cls):
        res = {}
        for cls, feat in PRIMARY.items():
            if cls == intended_cls:
                continue
            over = med[feat] - hr[feat][1]
            res[cls] = {"feature": feat, "median": med[feat], "healthy_max": hr[feat][1],
                        "excess_over_healthy_max": over, "confound": over > floor}
        return res

    ref_step = ref_stats["step_ms"]["center"]
    notes, decision = [], None
    if mech == "none":
        shifts = {f: med[f] - hr[f][1] for f in PRIMARY.values()}
        bad = {f: v for f, v in shifts.items() if v > floor}
        out["healthy_shift_over_range"] = shifts
        decision = "VALID FOR RULE DESIGN" if not bad else "WEAK / NEEDS REDESIGN"
        if bad:
            notes.append(f"healthy variant sits above the healthy range by more than the practical floor: {bad}")
    else:
        cls = out["class"]
        feat = PRIMARY[cls]
        sep = separation(feat)
        out["separation"] = sep
        out["offtarget"] = offtarget(cls)
        confound = [c for c, v in out["offtarget"].items() if v["confound"]]
        intended = None
        if mech in ("sleep", "fetch_sleep"):
            intended = fault["delay_ms"]
        elif mech == "loader_sleep":
            w = docs[0]["effective_config"]["workload"]["num_workers"]
            intended = fault["delay_ms"] / w - ref_step
        realized = delta[feat]
        out["intended_amount_ms"] = intended
        out["realized_amount_ms"] = realized
        out["realized_over_intended"] = (realized / intended) if intended else None
        structural_ok = True
        if mech == "single_bucket":
            # structure = distinct bucket counts AND per-bucket byte layout of the first measured step
            changed = bucket_obs != bucket_ref or {tuple(x) for x in layout} != {tuple(x) for x in ref_layout}
            out["bucket_structure_changed"] = changed
            structural_ok = changed
            if not changed:
                notes.append("bucket structure identical to healthy: the config change did not alter bucketization")
            if bytes_obs != bytes_ref:
                notes.append(f"communication bytes differ from healthy ({bytes_obs} vs {bytes_ref})")
        if mech == "small_batch":
            same_bytes = bytes_obs == bytes_ref
            ms_sep = sep["separated_all"]
            frac = separation("exposed_min_frac")
            out["separation_fraction"] = frac
            out["small_batch_character"] = (
                "communication more exposed in absolute ms (clean communication-bound signature)" if ms_sep else
                "ONLY the exposed FRACTION rises (less compute, same comm): ratio effect, a confound"
                if frac["separated_all"] else "neither exposed ms nor fraction separated")
            out["compute_delta_ms"] = delta["compute_ms"]
            if not same_bytes:
                notes.append(f"gradient bytes changed ({bytes_obs} vs {bytes_ref})")
            if not ms_sep:
                confound.append("ratio_only")
        if sep["separated_none"] and not (mech == "small_batch" and out.get("separation_fraction", {}).get("separated_all")):
            decision = "INVALID FAULT"
            notes.append(f"{feat} never leaves the healthy range")
        elif not structural_ok:
            decision = "INVALID FAULT"
        elif (not sep["separated_all"] or confound or
              (out["realized_over_intended"] is not None and
               not RATIO_BAND[0] <= out["realized_over_intended"] <= RATIO_BAND[1])):
            decision = "WEAK / NEEDS REDESIGN"
            if not sep["separated_all"]:
                notes.append(f"{sep['blocks_inside_healthy_range']}/{sep['n_blocks']} blocks inside healthy range")
            if confound:
                notes.append(f"confound: {confound}")
            r = out["realized_over_intended"]
            if r is not None and not RATIO_BAND[0] <= r <= RATIO_BAND[1]:
                notes.append(f"realized/intended = {r:.2f} outside {RATIO_BAND}")
        else:
            decision = "VALID FOR RULE DESIGN"
    out["decision"], out["notes"] = decision, notes

    # 4. threshold crossing per block, per rule feature
    cross = {}
    for f in ("data_wait_ms", "compute_skew_ms", "exposed_min_ms", "exposed_min_frac"):
        zs = [T.score_feature(b[f], f, ref_stats) for b in blocks]
        cross[f] = {"z_ge_K": sum(t["z"] >= T.K_ROBUST_Z for t in zs),
                    "delta_ge_floor": sum(t["delta"] >= t["practical_threshold"] for t in zs),
                    "elevated": sum(t["elevated"] for t in zs), "n": len(zs),
                    "z_min": min(t["z"] for t in zs), "z_median": float(np.median([t["z"] for t in zs])),
                    "z_max": max(t["z"] for t in zs)}
    out["threshold_crossing"] = cross
    diag = [diagnose(d, ref_stats) for d in docs]
    out["diagnoses_in_sample"] = [(x["verdict"], x["agreement"]) for x in diag]
    return out


def leakage_audit(docs, ref_stats) -> dict:
    import copy
    import inspect

    from diagnose import features, rules, thresholds
    src = "".join(inspect.getsource(m) for m in (features, rules, thresholds))
    code_lines = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    body = "\n".join(code_lines)
    forbidden = ["ground_truth", "fault_mechanism", "fault_class", "effective_config", "fault_runtime",
                 "injected_comm_delay", "[\"config\"]", "['config']"]
    hits = [f for f in forbidden if f in body]
    changed = []
    for d in docs:
        a = diagnose(d, ref_stats)
        d2 = copy.deepcopy(d)
        for k in ("ground_truth", "config", "effective_config", "fault_runtime", "command", "experiment_id",
                  "fingerprint_by_rank", "environment"):
            d2.pop(k, None)
        d2["run_id"] = "opaque"
        for s in d2["steps"]:
            s["run_id"] = "opaque"
            s.pop("injected_comm_delay_ms", None)
        b = diagnose(d2, ref_stats)
        if (a["verdict"], a["agreement"]) != (b["verdict"], b["agreement"]):
            changed.append(d["run_id"])
    view_keys = sorted(observable_view(docs[0]).keys())
    return {"observable_view_keys": view_keys,
            "run_id_in_view_note": "run_id (contains the arm name) is passed through for labelling only; "
                                   "verdicts are re-checked with an opaque run_id below",
            "forbidden_tokens_in_rule_code": hits,
            "verdict_changes_when_metadata_stripped_and_run_id_opaque": changed,
            "pass": not hits and not changed}


def threshold_sensitivity(arms_docs: dict, ref_docs) -> list[dict]:
    """How design-set outcomes change with K and the practical floor (design data only)."""
    rows = []
    saved = (T.K_ROBUST_Z, T.PRACTICAL_STEP_FRACTION)
    try:
        for K in (3.0, 4.0, 5.0, 6.0):
            for frac in (0.01, 0.03, 0.05):
                T.K_ROBUST_Z, T.PRACTICAL_STEP_FRACTION = K, frac
                correct = total = fp = 0
                for i, d in enumerate(ref_docs):  # leave-one-run-out healthy
                    st = build_reference(ref_docs, exclude_run=d["run_id"])
                    v = diagnose(d, st)["verdict"]
                    total += 1
                    correct += v == "HEALTHY"
                    fp += v != "HEALTHY"
                st = build_reference(ref_docs)
                per = {}
                for arm, docs in arms_docs.items():
                    ok = sum(diagnose(d, st)["verdict"] == d["ground_truth"]["fault_class"] for d in docs)
                    per[arm] = f"{ok}/{len(docs)}"
                    correct += ok
                    total += len(docs)
                rows.append({"K": K, "floor_frac": frac, "accuracy_in_sample": correct / total,
                             "healthy_ref_false_positives": fp, "per_arm": per})
    finally:
        T.K_ROBUST_Z, T.PRACTICAL_STEP_FRACTION = saved
    return rows


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # reports contain non-ASCII; never crash on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", help="output directory (default: next to the manifest); use for ingested "
                                  "evidence, which must never be modified")
    a = ap.parse_args()
    man, runs = load(a.manifest)
    ok = [(r, d) for r, d in runs if d is not None]
    ref_docs = [d for r, d in ok if r["role"] == "reference"]
    ref_stats = build_reference(ref_docs)
    ref_blocks = [b for d in ref_docs for b in block_features(observable_view(d))]
    arms = defaultdict(list)
    for r, d in ok:
        if r["role"] != "reference":
            arms[r["arm"]].append(d)
    report = {
        "manifest": a.manifest, "campaign_id": man["campaign_id"],
        "rules_sha256_at_audit": rules_fingerprint()["sha256"],
        "integrity": integrity(man, runs),
        "reference": reference_audit(ref_docs),
        "arms": [arm_audit(arm, docs, man, ref_stats, ref_blocks, ref_docs) for arm, docs in arms.items()],
        "leakage": leakage_audit([d for _, d in ok], ref_stats),
        "threshold_sensitivity": threshold_sensitivity(arms, ref_docs),
    }
    out = Path(a.out) if a.out else Path(a.manifest).parent
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = to_markdown(report)
    (out / "audit.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def to_markdown(r: dict) -> str:
    f = lambda x, n=2: "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))  # noqa: E731
    I, R = r["integrity"], r["reference"]
    L = [f"# Evidence audit: {r['campaign_id']}", "", "## 6. Campaign integrity", "",
         f"* launches: manifest {I['manifest_launches']} / expected {I['expected_launches']}; ok {I['ok']}; "
         f"failed {len(I['failed'])}",
         f"* campaign git SHA `{I['campaign_git_sha']}`; campaign dirty={I['campaign_dirty']} "
         f"override={I['campaign_override']}",
         f"* problems: {I['problems'] or 'none'}",
         f"* manifest seed metadata defect (run docs hold the true, distinct seeds): "
         f"{len(I['seed_metadata_defect'])} launches" + (" " + str([(x['run_id'], x['manifest_seed'], x['actual_seed'])
                                                               for x in I['seed_metadata_defect']]) if I['seed_metadata_defect'] else ""),
         f"* **integrity pass: {I['pass']}**", "",
         "## 1. Healthy reference", "",
         "| run | blocks | step ms | compute ms | exposed timeline ms | exposed min ms | data wait ms | "
         "compute skew ms | start skew ms | gates |", "|---|---|---|---|---|---|---|---|---|---|"]
    for p in R["per_run"]:
        L.append(f"| {p['run_id']} | {p['n_blocks']} | {f(p['step_ms'])} | {f(p['compute_ms'])} | "
                 f"{f(p['exposed_timeline_ms'])} | {f(p['exposed_min_ms'])} | {f(p['data_wait_ms'], 3)} | "
                 f"{f(p['compute_skew_ms'], 3)} | {f(p['rank_start_skew_ms'], 3)} | "
                 f"{'pass' if p['gates_pass'] else 'FAIL ' + str(p['gate_failures'])} |")
    L += ["", "Run-to-run (medians of per-run medians; CV across runs):", ""]
    for k, v in R["across_runs"].items():
        L.append(f"* {k}: median {f(v['median'], 3)}, range [{f(v['min'], 3)}, {f(v['max'], 3)}], "
                 f"CV {f(v['cv_pct'])} %")
    ab = R.get("ablation_exposed_ms") or {}
    L += ["", f"Paired ablation exposed communication: median {f(ab.get('median'))} ms, n={ab.get('n')}, "
              f"all positive {ab.get('all_positive')}, {f(100 * ab['fraction_of_ddp_step'], 1) if ab else 'n/a'} % "
              f"of step", f"GPU state ranges: {R.get('gpu_state')}", f"NCCL: {R.get('nccl')}",
          f"All reference gates pass: **{R['all_gates_pass']}**", "",
          "Sufficiency: bootstrap (over the reference runs) of the z boundary center + K*scale:", ""]
    for k, v in R["boundary_bootstrap"].items():
        L.append(f"* {k}: 90% interval [{f(v['boundary_p5'], 3)}, {f(v['boundary_p95'], 3)}] ms, width "
                 f"{f(v['width_ms'], 3)} vs practical floor {f(v['practical_floor_ms'], 3)} -> "
                 f"{'stable' if v['width_below_floor'] else 'UNSTABLE relative to floor'}")
    L += ["", "## 2/7. Fault manifestation and decision", ""]
    for x in r["arms"]:
        L.append(f"### {x['arm']} ({x['mechanism']}, class {x['class']}) -> **{x['decision']}**")
        L.append(f"* runs {x['n_runs']}, blocks {x['n_blocks']}; resolved fault {x['fault']}")
        L.append(f"* effective batch {x['effective_batch']}, bucket_cap_mb {x['effective_bucket_cap_mb']}; "
                 f"bucket count observed {x['bucket_count_observed']} vs healthy {x['bucket_count_healthy']}; "
                 f"bucket bytes {x['bucket_bytes_observed'][:1]} vs healthy {x['bucket_bytes_healthy']}; "
                 f"comm bytes {x['comm_bytes_observed']} vs {x['comm_bytes_healthy']}")
        L.append("* deltas vs healthy center: " + ", ".join(f"{k} {x['delta'][k]:+.3f}" for k in x["delta"]))
        if "separation" in x:
            s = x["separation"]
            L.append(f"* intended observable {s['feature']}: healthy [{f(s['healthy_range'][0], 3)}, "
                     f"{f(s['healthy_range'][1], 3)}], fault [{f(s['fault_range'][0], 3)}, "
                     f"{f(s['fault_range'][1], 3)}]; outside {s['blocks_outside_on_intended_side']}/{s['n_blocks']}")
            L.append(f"* intended {f(x['intended_amount_ms'])} ms, realized {f(x['realized_amount_ms'])} ms, "
                     f"ratio {f(x['realized_over_intended'])}")
            L.append("* off-target: " + "; ".join(f"{c} {v['feature']} excess {v['excess_over_healthy_max']:+.3f}"
                                                 f"{' CONFOUND' if v['confound'] else ''}"
                                                 for c, v in x["offtarget"].items()))
        if "small_batch_character" in x:
            L.append(f"* small_batch character: {x['small_batch_character']} (compute delta "
                     f"{f(x['compute_delta_ms'])} ms)")
        if "bucket_structure_changed" in x:
            L.append(f"* bucket structure changed: {x['bucket_structure_changed']}")
        L.append(f"* in-sample diagnoses: {x['diagnoses_in_sample']}")
        for n in x["notes"]:
            L.append(f"* NOTE: {n}")
        L.append("")
    L += ["## 4. Threshold crossing (blocks per arm)", "",
          "| arm | feature | z>=K | delta>=floor | elevated | n | z min/median/max |", "|---|---|---|---|---|---|---|"]
    for x in r["arms"]:
        for feat, c in x["threshold_crossing"].items():
            L.append(f"| {x['arm']} | {feat} | {c['z_ge_K']} | {c['delta_ge_floor']} | {c['elevated']} | {c['n']} | "
                     f"{c['z_min']:.1f} / {c['z_median']:.1f} / {c['z_max']:.1f} |")
    L += ["", "Sensitivity of in-sample outcomes to K and floor (design data only; NOT a tuning result):", "",
          "| K | floor | in-sample acc | healthy-ref FP (LOO) | per arm |", "|---|---|---|---|---|"]
    for s in r["threshold_sensitivity"]:
        L.append(f"| {s['K']} | {s['floor_frac']} | {s['accuracy_in_sample']:.3f} | {s['healthy_ref_false_positives']} | "
                 + ", ".join(f"{k} {v}" for k, v in s["per_arm"].items()) + " |")
    L += ["", "## 5. Class separability", "",
          "| fault | intended mechanism | strongest observable | healthy range | fault range | overlap | verdict |",
          "|---|---|---|---|---|---|---|"]
    for x in r["arms"]:
        if "separation" not in x:
            continue
        s = x["separation"]
        L.append(f"| {x['arm']} | {x['mechanism']} | {s['feature']} | [{f(s['healthy_range'][0], 3)}, "
                 f"{f(s['healthy_range'][1], 3)}] | [{f(s['fault_range'][0], 3)}, {f(s['fault_range'][1], 3)}] | "
                 f"{s['blocks_inside_healthy_range']}/{s['n_blocks']} blocks | {x['decision']} |")
    lk = r["leakage"]
    L += ["", "## 3. Diagnoser leakage", "", f"* observable view keys: {lk['observable_view_keys']}",
          f"* forbidden tokens in rule code: {lk['forbidden_tokens_in_rule_code'] or 'none'}",
          f"* verdict changes with metadata stripped + opaque run_id: "
          f"{lk['verdict_changes_when_metadata_stripped_and_run_id_opaque'] or 'none'}",
          f"* **leakage pass: {lk['pass']}**", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
