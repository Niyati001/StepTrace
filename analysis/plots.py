"""The eight required plots (Master Build Spec §25), drawn ONLY from saved evidence.

    python -m analysis.plots --campaign <root>/campaigns/<id>/manifest.json \
        [--evaluation <root>/campaigns/<id>/evaluation_<name>.json] \
        [--selection <tuning campaign>/optimization_selection.json] \
        [--optimization <validation campaign>/optimization_report.json] --out results/plots/<name>

  1 baseline step-time distribution (healthy reference, per launch)
  2 baseline vs fault step time (every arm)
  3 time breakdown: data wait / compute / exposed communication / finalize; hidden vs exposed comm
  4 fault signatures: change of each class's primary observable vs healthy
  5 rank skew (compute skew per block, per arm)
  6 optimization: tuning baseline/selected vs fresh-validation baseline/selected
  7 confusion matrix (headline category only)
  8 held-out: expected vs diagnosed per run (misses, did-not-manifest, marginal-signal marked)

Every figure carries a provenance footer (campaign id, git SHA of the code that produced
the evidence). Inputs marked ``synthetic_fixture`` (tests only) are watermarked
"SYNTHETIC TEST FIXTURE: NOT EVIDENCE", and the CLI refuses to write them under results/.
Colors: reference categorical slots in fixed order (validated: CVD/normal-vision
separation pass); every chart has a legend or direct labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.summary import cluster_steps, summarize  # noqa: E402
from diagnose.features import block_features, observable_view  # noqa: E402
from diagnose.thresholds import reference_stats  # noqa: E402
from instrument.evidence import resolve_run_file  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
BLUE_LIGHT, GREY = "#86b6ef", "#a9a8a2"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CLASS_COLOR = {"HEALTHY": GREY, "COMMUNICATION": BLUE, "STRAGGLER": ORANGE, "DATA_STALL": AQUA}
BLUES = ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


class Evidence:
    """A campaign's raw runs grouped by arm (read-only)."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.path = Path(manifest_path)
        self.man = json.loads(self.path.read_text(encoding="utf-8"))
        self.arms: dict[str, list[dict]] = defaultdict(list)
        self.roles: dict[str, str] = {}
        for r in self.man["runs"]:
            if r["status"] == "ok":
                doc = json.loads(resolve_run_file(self.path, r["run_file"]).read_text(encoding="utf-8"))
                self.arms[r["arm"]].append(doc)
                self.roles[r["arm"]] = r["role"]
        self.synthetic = bool(self.man.get("synthetic_fixture")) or any(
            d.get("synthetic_fixture") for ds in self.arms.values() for d in ds)
        self.ref = [d for a, ds in self.arms.items() if self.roles[a] == "reference" for d in ds]
        self.ref_stats = reference_stats([b for d in self.ref for b in block_features(observable_view(d))])

    def footer(self) -> str:
        devs = sorted({x for d in self.ref for x in d.get("devices", [])})
        return (f"campaign {self.man['campaign_id']} | code {self.man['provenance']['git_sha'][:10]} | "
                f"{', '.join(devs) or 'n/a'} | numbers specific to this recorded environment")


# --------------------------------------------------------------------------- helpers
def _style(ax, title, xlabel=None, ylabel=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _finish(fig, out: Path, footer: str, synthetic: bool) -> Path:
    fig.text(0.01, 0.01, footer, fontsize=6.5, color=INK2)
    if synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC TEST FIXTURE\nNOT EVIDENCE", fontsize=26, color="#d03b3b", alpha=0.35,
                 ha="center", va="center", rotation=20, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def _cluster_ddp(docs):
    return [c for d in docs for c in cluster_steps(d["steps"]) if c["mode"] == "ddp"]


def _arm_class(docs):
    return docs[0]["ground_truth"]["fault_class"]


def _box(ax, data, positions, color):
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1.2},
                    flierprops={"marker": "o", "markersize": 2.2, "markeredgecolor": color, "alpha": 0.6})
    for b in bp["boxes"]:
        b.set(facecolor=color, edgecolor=color, alpha=0.75)
    for w in bp["whiskers"] + bp["caps"]:
        w.set(color=color)


# --------------------------------------------------------------------------- 1
def plot_baseline_distribution(ev: Evidence, out: Path) -> Path:
    runs = sorted(ev.ref, key=lambda d: d["run_id"])
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(runs) + 2), 4), facecolor=SURFACE)
    for j, (mode, color, dx) in enumerate((("ddp", BLUE, -0.2), ("nosync", ORANGE, 0.2))):
        data = [[c["step_time_ms"] for c in cluster_steps(d["steps"]) if c["mode"] == mode] for d in runs]
        if any(data):
            _box(ax, [x or [np.nan] for x in data], [i + dx for i in range(len(runs))], color)
    ax.set_xticks(range(len(runs)), [f"L{d['launch_index']}" if "launch_index" in d else d["run_id"][-4:]
                                     for d in runs])
    _style(ax, "1 · Healthy reference: step time per launch (cluster step = slowest rank)",
           "healthy reference launch", "step time (ms)")
    ax.legend(handles=[Patch(color=BLUE, label="DDP"), Patch(color=ORANGE, label="no_sync (comm suppressed)")],
              frameon=False, fontsize=8, loc="upper left")
    return _finish(fig, out, ev.footer(), ev.synthetic)


# --------------------------------------------------------------------------- 2
def plot_baseline_vs_fault(ev: Evidence, out: Path) -> Path:
    arms = sorted(ev.arms, key=lambda a: (ev.roles[a] != "reference", _arm_class(ev.arms[a]), a))
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(arms) + 2), 4.4), facecolor=SURFACE)
    for i, a in enumerate(arms):
        _box(ax, [[c["step_time_ms"] for c in _cluster_ddp(ev.arms[a])]], [i], CLASS_COLOR[_arm_class(ev.arms[a])])
    ax.axhline(ev.ref_stats["step_ms"]["center"], color=INK2, linestyle="--", linewidth=1)
    ax.set_xticks(range(len(arms)), arms, rotation=35, ha="right")
    _style(ax, "2 · Step time: healthy reference vs each arm (dashed = healthy median)", None, "step time (ms)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLOR.items()], frameon=False, fontsize=8,
              ncol=4, loc="upper left")
    return _finish(fig, out, ev.footer(), ev.synthetic)


# --------------------------------------------------------------------------- 3
def plot_breakdown(ev: Evidence, out: Path) -> Path:
    s = summarize(ev.ref)
    d, n = s["modes"]["ddp"], s["modes"].get("nosync")
    med = lambda m, k: (m[k].get("median") or 0.0)  # noqa: E731
    segs = [("data wait", "data_wait_ms", YELLOW), ("compute", "compute_time_ms", BLUE),
            ("exposed communication", "exposed_communication_time_ms", ORANGE), ("DDP finalize", "ddp_finalize_ms", AQUA)]
    rows = [("DDP step", d)] + ([("no_sync step", n)] if n else [])
    fig, ax = plt.subplots(figsize=(8.5, 3.4), facecolor=SURFACE)
    for y, (label, m) in enumerate(rows):
        left = 0.0
        for name, key, color in segs:
            w = med(m, key)
            ax.barh(y, w, left=left, color=color, edgecolor=SURFACE, linewidth=2, height=0.55,
                    label=name if y == 0 else None)
            left += w
        ax.text(left + 0.4, y, f"{m['step_time_ms']['median']:.1f} ms", va="center", fontsize=8, color=INK)
    busy, exposed = med(d, "communication_time_ms"), med(d, "exposed_communication_time_ms")
    y = len(rows)
    ax.barh(y, busy - exposed, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=2, height=0.35,
            label="communication hidden behind compute")
    ax.barh(y, exposed, left=busy - exposed, color=ORANGE, edgecolor=SURFACE, linewidth=2, height=0.35)
    ax.text(busy + 0.4, y, f"all-reduce busy {busy:.1f} ms", va="center", fontsize=8, color=INK)
    abl = s.get("ablation_exposed_ms")
    title = "3 · Where a healthy step goes (medians of block medians)"
    if abl:
        title += f": paired ablation exposed comm {abl['median']:.2f} ms ({100 * abl['fraction_of_ddp_step']:.1f} %)"
    ax.set_yticks(range(y + 1), [r[0] for r in rows] + ["DDP all-reduce"])
    ax.invert_yaxis()
    _style(ax, title, "milliseconds", grid_axis="x")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    return _finish(fig, out, ev.footer(), ev.synthetic)


# --------------------------------------------------------------------------- 4
PRIMARY = [("data_wait_ms", "data wait", AQUA), ("compute_skew_ms", "compute skew", ORANGE),
           ("exposed_min_ms", "exposed comm (min rank)", BLUE)]


def plot_signatures(ev: Evidence, out: Path) -> Path:
    arms = [a for a in sorted(ev.arms) if ev.roles[a] != "reference"]
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(arms) + 2), 4.4), facecolor=SURFACE)
    w = 0.26
    for j, (k, label, color) in enumerate(PRIMARY):
        vals = [float(np.median([b[k] for d in ev.arms[a] for b in block_features(observable_view(d))]))
                - ev.ref_stats[k]["center"] for a in arms]
        ax.bar([i + (j - 1) * w for i in range(len(arms))], vals, width=w, color=color, label=label)
    ax.axhline(ev.ref_stats["_practical_ms"], color=INK2, linestyle=":", linewidth=1)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(range(len(arms)), arms, rotation=35, ha="right")
    _style(ax, "4 · Fault signatures vs same-session healthy" + chr(10) + "(dotted line = practical floor)",
           None, "Δ vs healthy median (ms)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    return _finish(fig, out, ev.footer(), ev.synthetic)


# --------------------------------------------------------------------------- 5
def plot_rank_skew(ev: Evidence, out: Path) -> Path:
    arms = sorted(ev.arms, key=lambda a: (ev.roles[a] != "reference", _arm_class(ev.arms[a]) != "STRAGGLER", a))
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(arms) + 2), 4.2), facecolor=SURFACE)
    for i, a in enumerate(arms):
        vals = [b["compute_skew_ms"] for d in ev.arms[a] for b in block_features(observable_view(d))]
        _box(ax, [vals], [i], CLASS_COLOR[_arm_class(ev.arms[a])])
    ax.set_xticks(range(len(arms)), arms, rotation=35, ha="right")
    _style(ax, "5 · Rank compute skew per block (max - min compute across ranks)", None, "compute skew (ms)")
    ax.legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLOR.items()], frameon=False, fontsize=8,
              ncol=4, loc="upper left")
    return _finish(fig, out, ev.footer(), ev.synthetic)


# --------------------------------------------------------------------------- 6
def plot_optimization(selection: dict, report: dict, out: Path, synthetic: bool) -> Path:
    scen = sorted(report["scenarios"])
    tune = selection["tuning"]["scenarios"]
    fig, axs = plt.subplots(1, len(scen), figsize=(max(7.5, 3.6 * len(scen) + 1), 4.4), facecolor=SURFACE,
                            squeeze=False)
    for ax, sc in zip(axs[0], scen):
        r, sel = report["scenarios"][sc], selection["selection"][sc]
        rows = {x["arm"]: x for x in tune[sc]}
        c = r["candidates"][0] if r["candidates"] else None
        vals = [rows[sel["baseline"]]["samples_per_s_per_gpu"],
                rows[sel["primary"]]["samples_per_s_per_gpu"] if sel["primary"] else np.nan,
                r["baseline"]["samples_per_s_per_gpu"], c["after"]["samples_per_s_per_gpu"] if c else np.nan]
        colors = [BLUE_LIGHT, ORANGE, BLUE, AQUA]
        ax.bar(range(4), vals, color=colors, edgecolor=SURFACE, linewidth=2)
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(4), ["tuning\nbaseline", "tuning\nselected", "validation\nbaseline",
                                 "validation\nselected"], fontsize=7)
        verdict = c["verdict"] if c else "n/a"
        _style(ax, f"{sc} ({r['fault_nature']})\n{verdict}", None, "samples/s per GPU")
    fig.suptitle("6 · Optimization" + chr(10) + "tuning = selection only; fresh-session validation = the result",
                 x=0.01, ha="left", fontsize=11, color=INK)
    return _finish(fig, out, f"validation campaign {report['campaign_id']}; injected/emulated-fault gains are "
                             f"not real-hardware gains", synthetic)


# --------------------------------------------------------------------------- 7
def plot_confusion(evaluation: dict, out: Path, synthetic: bool) -> Path:
    s = evaluation["scores"]
    cm = np.array(s["confusion_matrix"])
    labels = s["labels"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor=SURFACE)
    vmax = max(1, cm.max())
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i, j]
            color = BLUES[min(len(BLUES) - 1, int(round(v / vmax * (len(BLUES) - 1))))]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, str(v), ha="center", va="center", fontsize=11,
                    color="white" if v / vmax > 0.55 else INK)
    ax.set_xlim(0, len(labels))
    ax.set_ylim(len(labels), 0)
    ax.set_xticks(np.arange(len(labels)) + 0.5, labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)) + 0.5, labels)
    _style(ax, f"7 · Confusion matrix ({', '.join(evaluation['roles'])}; headline only)\n"
               f"accuracy {s['accuracy']:.2f} on n = {s['n']}", "predicted", "ground truth", grid_axis=None)
    return _finish(fig, out, f"campaign {evaluation['campaign_id']} | rules {evaluation['rules']['rules_sha256'][:12]}",
                   synthetic)


# --------------------------------------------------------------------------- 8
def plot_heldout(evaluation: dict, out: Path, synthetic: bool) -> Path:
    rows = sorted(evaluation["rows"], key=lambda x: (x["truth"], x["arm"], x["launch"]))
    labels = ["HEALTHY", "COMMUNICATION", "STRAGGLER", "DATA_STALL"]
    fig, ax = plt.subplots(figsize=(max(7, 0.32 * len(rows) + 3), 4.2), facecolor=SURFACE)
    for i, x in enumerate(rows):
        ax.scatter(i, labels.index(x["truth"]), s=90, facecolors="none", edgecolors=INK2, linewidths=1.2)
        marker, color = (("o", AQUA) if x["correct"] else ("X", ORANGE))
        if x["category"] == "did_not_manifest":
            marker, color = "s", GREY
        elif x["category"] == "marginal_signal":
            marker = "D"
        ax.scatter(i, labels.index(x["predicted"]) if x["predicted"] in labels else -0.6, s=34,
                   marker=marker, color=color)
    ax.set_yticks(range(4), labels)
    ax.set_xticks(range(len(rows)), [x["arm"] for x in rows], rotation=60, ha="right", fontsize=6.5)
    _style(ax, "8 · Held-out runs: expected (ring) vs diagnosed (mark)", None, None)
    ax.legend(handles=[plt.Line2D([], [], marker="o", color=AQUA, linestyle="", label="correct"),
                       plt.Line2D([], [], marker="X", color=ORANGE, linestyle="", label="miss"),
                       plt.Line2D([], [], marker="D", color=INK2, linestyle="", label="marginal-signal (excluded)"),
                       plt.Line2D([], [], marker="s", color=GREY, linestyle="", label="did not manifest (excluded)"),
                       plt.Line2D([], [], marker="o", color=INK2, markerfacecolor="none", linestyle="",
                                  label="ground truth")],
              frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.45))
    return _finish(fig, out, f"campaign {evaluation['campaign_id']} | frozen rules "
                             f"{(evaluation.get('frozen_check') or {}).get('frozen_file_sha256', 'n/a')[:12]}",
                   synthetic)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign")
    ap.add_argument("--evaluation")
    ap.add_argument("--selection")
    ap.add_argument("--optimization")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    made, synthetic = [], False
    if a.campaign:
        ev = Evidence(a.campaign)
        synthetic |= ev.synthetic
        jobs = [(plot_baseline_distribution, "01_baseline_step_time.png"), (plot_baseline_vs_fault, "02_baseline_vs_fault.png"),
                (plot_breakdown, "03_time_breakdown.png"), (plot_signatures, "04_fault_signatures.png"),
                (plot_rank_skew, "05_rank_skew.png")]
    for path in [a.evaluation, a.optimization, a.selection]:
        if path:   # derived inputs inherit the synthetic flag of the campaign they came from
            j = json.loads(Path(path).read_text(encoding="utf-8"))
            src = j.get("manifest") or (j.get("tuning") or {}).get("manifest")
            if j.get("synthetic_fixture") or (src and Path(src).is_file() and
                                              json.loads(Path(src).read_text(encoding="utf-8")).get("synthetic_fixture")):
                synthetic = True
    if synthetic and (ROOT / "results") in out.resolve().parents:
        raise SystemExit("refusing to write plots from SYNTHETIC fixtures under results/")
    if a.campaign:
        made += [str(fn(ev, out / name)) for fn, name in jobs]
    if a.selection and a.optimization:
        sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        rep = json.loads(Path(a.optimization).read_text(encoding="utf-8"))
        made.append(str(plot_optimization(sel, rep, out / "06_optimization.png", synthetic)))
    if a.evaluation:
        e = json.loads(Path(a.evaluation).read_text(encoding="utf-8"))
        made.append(str(plot_confusion(e, out / "07_confusion_matrix.png", synthetic)))
        made.append(str(plot_heldout(e, out / "08_heldout.png", synthetic)))
    print("\n".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
