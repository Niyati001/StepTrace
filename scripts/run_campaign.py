"""Run a GPU campaign: same-session healthy reference FIRST, then controlled runs.

    python scripts/run_campaign.py --campaign configs/campaigns/m2_design.yaml
    python scripts/run_campaign.py --campaign configs/campaigns/m2_design.yaml --cpu --set workload.model=cnn_tiny  # dev

Order and guarantees:
  1. refuses to start from a dirty tree (instrument/provenance.py);
  2. phase "reference": healthy runs of the selected workload; measurement-validity
     gates (analysis/validation.py) are re-checked on them in THIS session, and the
     campaign stops if they fail;
  3. relative fault magnitudes are resolved from the reference median DDP step:
       {rel_step: x}   -> x * step
       {loader_rel: x} -> num_workers * (1 + x) * step   (per-batch worker delay that
                          leaves an expected wait of ~ x * step per step)
       {loader_rel_cycle: x} -> num_workers * (1 + x) * T_cycle, T_cycle = median interval
                          between consecutive step starts in the healthy reference (the
                          consumer's real inter-batch period): expected wait ~ x * T_cycle
  4. remaining phases; arms interleaved round-robin by launch index to spread drift.
Every launch is appended to results/campaigns/<name>-<tag>/manifest.json as it
finishes (crash-safe). Raw runs: results/raw/campaigns/<name>-<tag>/<arm>/.
CPU (--cpu) campaigns write under results/dev/ and are never evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.summary import cluster_steps, summarize  # noqa: E402
from analysis.validation import format_checks, validate  # noqa: E402
from diagnose.diagnose import build_reference, rules_fingerprint  # noqa: E402
from faults.manifestation import reference_profile  # noqa: E402
from instrument.environment import detect  # noqa: E402
from instrument.provenance import require_clean  # noqa: E402
from run_experiment import format_summary, launch  # noqa: E402
from run_pilot import to_sets  # noqa: E402
from workloads import config as C  # noqa: E402


def resolve(value, ref_step_ms: float, num_workers: int, ref_cycle_ms: float | None = None):
    if isinstance(value, dict) and "rel_step" in value:
        return round(value["rel_step"] * ref_step_ms, 3)
    if isinstance(value, dict) and "loader_rel_cycle" in value:
        if ref_cycle_ms is None:
            raise SystemExit("loader_rel_cycle needs the reference inter-batch period (reference phase first)")
        return round(num_workers * (1 + value["loader_rel_cycle"]) * ref_cycle_ms, 3)
    if isinstance(value, dict) and "loader_rel" in value:
        return round(num_workers * (1 + value["loader_rel"]) * ref_step_ms, 3)
    return value


class Campaign:
    def __init__(self, spec_path: str, cpu: bool, extra_sets: list[str]) -> None:
        self.spec_path = spec_path
        self.spec = yaml.safe_load(Path(spec_path).read_text(encoding="utf-8"))
        self.cpu, self.extra = cpu, extra_sets
        self.tag = time.strftime("%Y%m%d-%H%M%S")
        self.id = f"{self.spec['name']}-{self.tag}"
        dev = "results/dev/" if cpu else "results/"
        self.raw_root = f"{dev}raw/campaigns/{self.id}"
        self.out = ROOT / dev / "campaigns" / self.id
        self.out.mkdir(parents=True, exist_ok=True)
        grid = {p["id"]: p for p in yaml.safe_load((ROOT / "configs/pilot.yaml").read_text(encoding="utf-8"))["grid"]}
        self.point_sets = to_sets(grid[self.spec["workload_point"]]["override"])
        self.manifest = {
            "campaign": self.spec["name"], "campaign_id": self.id, "spec_file": spec_path,
            "spec": self.spec, "cpu_dev_run": cpu, "extra_sets": extra_sets,
            "provenance": require_clean(), "environment": detect(),
            "rules_sha256": rules_fingerprint()["sha256"],
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "resolved": {}, "runs": [], "phases": [],
        }
        self.save()

    def save(self) -> None:
        (self.out / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str), encoding="utf-8")

    def arm_sets(self, arm: dict, fault: dict | None) -> list[str]:
        sets = self.point_sets + to_sets(self.spec.get("base", {})) + to_sets(arm.get("set", {}))
        if fault:
            sets += to_sets({"fault": fault})
        return sets + [f"experiment.name={arm['id']}"] + self.extra

    def run_arm_launch(self, phase: dict, arm: dict, i: int, fault: dict | None) -> None:
        arm_index = [a["id"] for p in self.spec["phases"] for a in p["arms"]].index(arm["id"])
        # an arm may pin its seed (paired correctness checks use identical seeds and data)
        seed = arm["seed"] if "seed" in arm else self.spec["seed"] + 1000 * arm_index + i
        sets = self.arm_sets(arm, fault)
        C.load(None, sets)  # validate before launching
        rec = {"phase": phase["name"], "role": phase["role"], "arm": arm["id"], "launch": i,
               "seed": seed, "fault": fault or {"mechanism": "none"}, "sets": sets}
        t0 = time.time()
        try:
            path = launch(None, sets, 1, self.cpu, 2, out_root=self.raw_root, tag=self.tag, first_index=i,
                          seed=seed)[0]
            rec.update(status="ok", run_file=str(path.relative_to(ROOT).as_posix()))
        except SystemExit as e:
            rec.update(status="failed", error=str(e))
        rec["wall_seconds"] = round(time.time() - t0, 1)
        self.manifest["runs"].append(rec)
        self.save()

    def docs(self, role: str | None = None, arm: str | None = None) -> list[dict]:
        return [json.loads((ROOT / r["run_file"]).read_text(encoding="utf-8")) for r in self.manifest["runs"]
                if r["status"] == "ok" and (role is None or r["role"] == role) and (arm is None or r["arm"] == arm)]

    def reference_report(self) -> float:
        docs = self.docs(role="reference")
        if not docs:
            raise SystemExit("reference phase produced no runs")
        s = summarize(docs)
        s.update(backend=docs[0]["backend"], devices=docs[0]["devices"], config=docs[0]["config"])
        checks = validate(docs)
        gates_ok = all(c["passed"] for c in checks if c["kind"] == "gate")
        ref_stats = build_reference(docs)
        cs = [r for r in cluster_steps([x for d in docs for x in d["steps"]]) if r["mode"] == "ddp"]
        ref_step = float(np.median([r["step_time_ms"] for r in cs]))
        rep = {"summary": s, "validation_checks": checks, "gates_pass": gates_ok,
               "diagnoser_reference": ref_stats, "reference_median_ddp_step_ms": ref_step,
               "n_runs": len(docs)}
        (self.out / "reference_summary.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        (self.out / "reference_summary.md").write_text(
            format_summary(s) + "\n" + format_checks(checks) + "\n\nDiagnoser reference (block features):\n" +
            "\n".join(f"  {k:17s} center {v['center']:.4g}  scale {v['scale']:.3g}  range [{v['min']:.4g}, "
                      f"{v['max']:.4g}]  n={v['n']}" for k, v in ref_stats.items() if not k.startswith("_")) + "\n", encoding="utf-8")
        ref_cycle = reference_profile(docs)["cycle_median_ms"]
        rep["reference_median_cycle_ms"] = ref_cycle
        (self.out / "reference_summary.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        self.manifest["reference"] = {"median_ddp_step_ms": ref_step, "median_cycle_ms": ref_cycle,
                                      "gates_pass": gates_ok,
                                      "n_runs": len(docs), "n_blocks": ref_stats["_n_blocks"]}
        self.save()
        print(format_summary(s))
        print(format_checks(checks))
        if not gates_ok and not self.cpu:
            raise SystemExit("reference measurement-validity gates FAILED in this session: stop")
        return ref_step

    def run(self, phases: list[str] | None) -> None:
        ref_step = None
        for phase in self.spec["phases"]:
            if phases and phase["name"] not in phases:
                continue
            self.manifest["phases"].append({"name": phase["name"], "started_utc":
                                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            self.save()
            print(f"\n================ phase {phase['name']} ({phase['role']})", flush=True)
            if phase["role"] != "reference" and ref_step is None:
                ref_step = self.manifest.get("reference", {}).get("median_ddp_step_ms")
                if ref_step is None:
                    raise SystemExit("non-reference phase needs the reference phase first (same session)")
            arms = phase["arms"]
            n_max = max(a["launches"] for a in arms)
            for i in range(n_max):           # round-robin interleave across arms
                for arm in arms:
                    if i >= arm["launches"]:
                        continue
                    fault = None
                    if arm.get("fault"):
                        workers = C.load(None, self.arm_sets(arm, None))["workload"]["num_workers"]
                        cycle = self.manifest.get("reference", {}).get("median_cycle_ms")
                        fault = {k: resolve(v, ref_step, workers, cycle) for k, v in arm["fault"].items()}
                        self.manifest["resolved"][arm["id"]] = fault
                    self.run_arm_launch(phase, arm, i, fault)
            if phase["role"] == "reference":
                ref_step = self.reference_report()
        self.manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.save()
        failed = [r for r in self.manifest["runs"] if r["status"] != "ok"]
        print(f"\n[campaign] {self.id}: {len(self.manifest['runs'])} launches, {len(failed)} failed")
        print(f"[campaign] manifest: {self.out / 'manifest.json'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--phases", nargs="*", help="run only these phases (reference must be included)")
    ap.add_argument("--cpu", action="store_true", help="CPU/Gloo development run (results/dev, not evidence)")
    ap.add_argument("--set", action="append", default=[], help="extra override for every launch (dev only)")
    a = ap.parse_args()
    if a.set and not a.cpu:
        raise SystemExit("--set overrides are for --cpu development runs only; official campaigns run as specified")
    Campaign(a.campaign, a.cpu, a.set).run(a.phases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
