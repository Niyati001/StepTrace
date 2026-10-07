"""Milestone 1 plots, generated only from saved results.

    python scripts/plot_m1.py --baseline results/processed/<baseline_exp>/<tag> \
                              --pilot results/processed/pilot/<tag>

Writes PNGs to results/plots/m1/. Colors: reference categorical slots in fixed
order (validated: CVD/normal-vision separation pass; two slots < 3:1 contrast on
the light surface, so every chart carries a legend and numbers live in the
summary tables).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.summary import cluster_steps  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_LIGHT = "#86b6ef"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y" if xlabel is None or ylabel else "x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def footer(fig, s):
    cfg = s["config"]["workload"]
    devs = sorted(set(s["devices"]))
    where = f"{len(s['devices'])}x {', '.join(devs)}, single node, backend={s['backend']}"
    if s.get("nccl"):
        where += ", NCCL " + "; ".join(f"{v} via {t}" for t, v in s["nccl"])
    if s["backend"] != "nccl":
        where += " (CPU development data: NOT GPU performance)"
    fig.text(0.01, 0.035, f"Measured on {where}.", fontsize=6.5, color=INK2)
    fig.text(0.01, 0.008, f"model={cfg['model']} batch/GPU={cfg['batch_size']} precision={cfg['precision']}; "
             "numbers are specific to the recorded environment.", fontsize=6.5, color=INK2)


def plot_baseline_distribution(s, docs, out):
    cs = cluster_steps([x for d in docs for x in d["steps"]])
    keys = sorted({(r["run_id"], r["repeat"]) for r in cs})
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(keys) + 2), 4), facecolor=SURFACE)
    for j, (mode, color, dx) in enumerate((("ddp", BLUE, -0.18), ("nosync", ORANGE, 0.18))):
        data = [[r["step_time_ms"] for r in cs if (r["run_id"], r["repeat"]) == k and r["mode"] == mode]
                for k in keys]
        bp = ax.boxplot(data, positions=[i + dx for i in range(len(keys))], widths=0.3, showfliers=True,
                        patch_artist=True, medianprops={"color": INK, "linewidth": 1.2},
                        flierprops={"marker": "o", "markersize": 2.5, "markeredgecolor": color, "alpha": 0.6})
        for b in bp["boxes"]:
            b.set(facecolor=color, edgecolor=color, alpha=0.75)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=color)

    ax.set_xticks(range(len(keys)), [f"L{k[0].rsplit('-L', 1)[-1]}·r{k[1]}" for k in keys])
    style(ax, "Baseline step time per repeat (cluster step = slowest rank)", "launch · repeat", "step time (ms)")
    ax.legend(handles=[Patch(color=BLUE, label="DDP (communication on)"),
                       Patch(color=ORANGE, label="no_sync (communication suppressed)")],
              frameon=False, fontsize=8, loc="upper left")
    footer(fig, s)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_breakdown(s, out):
    d, n = s["modes"]["ddp"], s["modes"].get("nosync")
    med = lambda m, k: (m[k].get("median") or 0.0)  # noqa: E731
    segs = [("data wait", "data_wait_ms", YELLOW), ("compute", "compute_time_ms", BLUE),
            ("exposed communication", "exposed_communication_time_ms", ORANGE),
            ("DDP finalize", "ddp_finalize_ms", AQUA)]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=SURFACE)
    rows = [("DDP step", d)] + ([("no_sync step", n)] if n else [])
    for y, (label, m) in enumerate(rows):
        left = 0.0
        for name, key, color in segs:
            w = med(m, key)
            ax.barh(y, w, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.55,
                    label=name if y == 0 else None)
            left += w
        ax.text(left + 0.5, y, f"{m['step_time_ms']['median']:.1f} ms (median)", va="center", fontsize=8, color=INK)
    busy, exposed = med(d, "communication_time_ms"), med(d, "exposed_communication_time_ms")
    y = len(rows)
    ax.barh(y, busy - exposed, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, height=0.35,
            label="communication hidden behind compute")
    ax.barh(y, exposed, left=busy - exposed, color=ORANGE, edgecolor=SURFACE, linewidth=2, height=0.35)
    ax.text(busy + 0.5, y, f"collective busy {busy:.1f} ms", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(y + 1), [r[0] for r in rows] + ["DDP all-reduce"])
    ax.invert_yaxis()
    style(ax, "Where a training step goes (medians of block medians)", "milliseconds")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    footer(fig, s)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pilot(p, out):
    rows = [r for r in p["rows"] if "error" not in r and r.get("ablation_fraction") is not None]
    sel = p["selection"]["selected"]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(rows) + 1.6), facecolor=SURFACE)
    for i, r in enumerate(rows):
        c = ORANGE if r["communication_relevant"] else BLUE_LIGHT
        ax.barh(i, 100 * r["ablation_fraction"], color=c, height=0.6, edgecolor=SURFACE, linewidth=2)
        note = f"{r['ddp_step_ms']:.1f} ms step, {r['ablation_exposed_ms']:.1f} ms exposed"
        note += "  ← selected" if r["id"] == sel else ""
        ax.text(max(100 * r["ablation_fraction"], 0) + 0.8, i, note, va="center", fontsize=7.5, color=INK)
    thr = 0.10  # analysis.summary.measurability min_fraction (pre-registered)
    ax.axvline(100 * thr, color=INK2, linewidth=1, linestyle="--")
    ax.set_yticks(range(len(rows)), [r["id"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, max(100 * max(r["ablation_fraction"] for r in rows), 100 * thr) * 1.6)
    style(ax, "Pilot: exposed communication share of the DDP step (ablation)", "% of DDP step time")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=ORANGE, label="communication-relevant (resolvable and material)"),
                       Patch(color=BLUE_LIGHT, label="not relevant"),
                       plt.Line2D([], [], color=INK2, linestyle="--", label="10% material threshold")],
              frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_profiler(s, out):
    rows = s.get("profile_cross_check") or []
    rows = [r for r in rows if r["gpu_kernels_found"]]
    if not rows:
        return False
    labels = [f"{r['run_id'].rsplit('-L', 1)[-1] and 'L' + r['run_id'].rsplit('-L', 1)[-1]} rank{r['rank']}" for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(9, 0.35 * len(rows) + 1.8), facecolor=SURFACE, sharey=True)
    for ax, (hk, pk, title) in zip(axs, (("hook_comm_busy_ms", "prof_nccl_kernel_ms", "Collective busy time"),
                                         ("hook_exposed_ms", "prof_exposed_nccl_ms", "Exposed communication"))):
        y = range(len(rows))
        ax.barh([i - 0.18 for i in y], [r[hk] for r in rows], height=0.34, color=BLUE, label="CUDA events (hook)")
        ax.barh([i + 0.18 for i in y], [r[pk] for r in rows], height=0.34, color=ORANGE,
                label="torch.profiler kernels")
        style(ax, title, "ms per step (median over profiled steps)")
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(rows)), labels)
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline")
    ap.add_argument("--pilot")
    ap.add_argument("--out", default="results/plots/m1")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    made = []
    if a.baseline:
        s = json.loads((Path(a.baseline) / "summary.json").read_text(encoding="utf-8"))
        docs = [json.loads((ROOT / f).read_text(encoding="utf-8")) for f in s["run_files"]]
        plot_baseline_distribution(s, docs, out / "01_baseline_step_time.png")
        plot_breakdown(s, out / "03_time_breakdown.png")
        made += ["01_baseline_step_time.png", "03_time_breakdown.png"]
        if plot_profiler(s, out / "m1_profiler_cross_check.png"):
            made.append("m1_profiler_cross_check.png")
    if a.pilot:
        plot_pilot(json.loads((Path(a.pilot) / "pilot_summary.json").read_text(encoding="utf-8")), out / "m1_pilot.png")
        made.append("m1_pilot.png")
    print("[plot_m1] wrote", [str(out / m) for m in made])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
