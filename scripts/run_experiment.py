"""Run one experiment config for N independent launches, then summarize.

    python scripts/run_experiment.py --config configs/baseline.yaml --launches 5
    python scripts/run_experiment.py --config configs/baseline.yaml --cpu --set workload.model=cnn_tiny

Each launch is a fresh process group (torchrun) with seed = experiment.seed +
launch index, so launch-to-launch variance (allocator, cuDNN autotuning,
clocks) is part of the measured spread. Outputs:
    results/raw/<experiment>/<run_id>.json (+ _steps.csv, traces/)
    results/processed/<experiment>/<tag>/summary.json, summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.summary import summarize  # noqa: E402
from instrument.provenance import require_clean  # noqa: E402
from workloads import config as C  # noqa: E402


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def launch(config: str | None, sets: list[str], launches: int, cpu: bool, nproc: int,
           out_root: str | None = None, tag: str | None = None, first_index: int = 0,
           seed: int | None = None) -> list[Path]:
    """seed: exact seed for a single launch (campaigns); default experiment.seed + launch index."""
    if seed is not None and launches != 1:
        raise ValueError("an explicit seed applies to exactly one launch")
    require_clean()  # fail before spending compute (override: COMMSCOPE_ALLOW_DIRTY=1, dev only)
    cfg = C.load(config, sets)  # validate before spending compute
    exp, base_seed = cfg["experiment"]["name"], cfg["experiment"]["seed"]
    tag = tag or time.strftime("%Y%m%d-%H%M%S")
    use_cpu = cpu or gpu_count() < nproc
    if use_cpu and not cpu:
        print(f"[run_experiment] {gpu_count()} GPU(s) < nproc={nproc}: using CPU/Gloo spawn path. "
              "These timings are NOT GPU performance results.")
    # CPU/Gloo development output never mixes with GPU evidence.
    out_root = out_root or ("results/dev/raw" if use_cpu else "results/raw")
    paths = []
    for i in range(first_index, first_index + launches):
        run_id = f"{exp}-{tag}-L{i}"
        args = (["--config", config] if config else []) + sum((["--set", s] for s in sets), [])
        run_seed = seed if seed is not None else base_seed + i
        args += ["--set", f"experiment.seed={run_seed}", "--launch-index", str(i), "--out-root", out_root]
        if use_cpu:
            cmd = [sys.executable, "-m", "workloads.train", "--spawn", str(nproc),
                   "--set", "distributed.backend=gloo"] + args
        else:
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={nproc}", "-m", "workloads.train"] + args
        env = {**os.environ, "COMMSCOPE_RUN_ID": run_id, "PYTHONPATH": str(ROOT)}
        print(f"[run_experiment] launch {run_id}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        path = ROOT / out_root / exp / f"{run_id}.json"
        if rc != 0 or not path.exists():
            raise SystemExit(f"launch {i} failed (exit {rc}); see output above")
        paths.append(path)
    return paths


def write_summary(paths: list[Path], exp: str, tag: str) -> tuple[dict, Path]:
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    s = summarize(docs)
    s["run_files"] = [str(p.relative_to(ROOT)) for p in paths]
    s["config"] = docs[0]["config"]
    s["devices"] = docs[0]["devices"]
    s["backend"] = docs[0]["backend"]
    dev = any("results/dev" in p.as_posix() for p in paths)
    out = ROOT / ("results/dev/processed" if dev else "results/processed") / exp / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    (out / "summary.md").write_text(format_summary(s), encoding="utf-8")
    return s, out


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def format_summary(s: dict) -> str:
    L = [f"backend={s['backend']} devices={s['devices']} runs={s['n_runs']}", ""]
    L.append("| mode | blocks | steps | step ms (median of block medians) | sd | compute | comm busy | "
             "exposed (timeline) | finalize | data wait | rank skew |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for mode, m in s["modes"].items():
        L.append(f"| {mode} | {m['n_blocks']} | {m['n_steps']} | {_f(m['step_time_ms'].get('median'))} | "
                 f"{_f(m['step_time_ms'].get('std'))} | {_f(m['compute_time_ms'].get('median'))} | "
                 f"{_f(m['communication_time_ms'].get('median'))} | "
                 f"{_f(m['exposed_communication_time_ms'].get('median'))} | {_f(m['ddp_finalize_ms'].get('median'))} | "
                 f"{_f(m['data_wait_ms'].get('median'), 3)} | {_f(m['rank_step_skew_ms'].get('median'), 3)} |")
    a = s.get("ablation_exposed_ms")
    if a:
        L += ["", f"Ablation exposed communication (ddp - nosync, paired by repeat): median "
                  f"{_f(a.get('median'))} ms (n={a['n']}, sd {_f(a.get('std'))}, all positive: {a['all_positive']}), "
                  f"{100 * a['fraction_of_ddp_step']:.1f}% of ddp step",
              f"Compute interference (ddp compute - nosync compute): median "
              f"{_f(s['compute_interference_ms'].get('median'))} ms"]
    L += ["", f"Measurability: {s['measurability']}"]
    for r in s.get("profile_cross_check", []):
        L.append(f"Profiler rank{r['rank']}: hook step {_f(r['hook_step_ms'])} / prof step {_f(r['prof_step_ms'])} ms; "
                 f"hook comm busy {_f(r['hook_comm_busy_ms'])} / NCCL kernels {_f(r['prof_nccl_kernel_ms'])} ms; "
                 f"hook exposed {_f(r['hook_exposed_ms'])} / prof exposed NCCL {_f(r['prof_exposed_nccl_ms'])} ms")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--launches", type=int, default=1)
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--cpu", action="store_true", help="force CPU/Gloo spawn path")
    a = ap.parse_args()
    tag = time.strftime("%Y%m%d-%H%M%S")
    paths = launch(a.config, a.set, a.launches, a.cpu, a.nproc, tag=tag)
    exp = C.load(a.config, a.set)["experiment"]["name"]
    s, out = write_summary(paths, exp, tag)
    print("\n" + format_summary(s))
    print(f"[run_experiment] summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
