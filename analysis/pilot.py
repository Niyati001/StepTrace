"""Pilot selection rule (pre-registered in METHODOLOGY.md §Pilot before any GPU data).

1. A configuration is *communication-relevant* if ``analysis.summary.measurability``
   says resolvable AND material.
2. Walk models in the preferred order (ResNet-18 first, per the spec); the first
   model with at least one relevant configuration wins.
3. Within that model pick the configuration whose ablation exposed-communication
   fraction is closest to ``target_fraction`` (default 0.25): clearly
   communication-sensitive, but not so communication-dominated that later
   straggler / data-stall signatures and optimizations have no headroom.
4. If nothing qualifies -> spec stop condition A (change the workload).
"""

from __future__ import annotations


def pilot_row(cfg_id: str, cfg: dict, summary: dict) -> dict:
    m, abl = summary["modes"], summary.get("ablation_exposed_ms", {})
    ddp, ns = m.get("ddp", {}), m.get("nosync", {})
    meas = summary["measurability"]
    return {
        "id": cfg_id,
        "model": cfg["workload"]["model"],
        "batch_size": cfg["workload"]["batch_size"],
        "precision": cfg["workload"]["precision"],
        "bucket_cap_mb": cfg["distributed"]["bucket_cap_mb"],
        "n_repeats": ddp.get("n_blocks", 0),
        "ddp_step_ms": ddp.get("step_time_ms", {}).get("median"),
        "nosync_step_ms": ns.get("step_time_ms", {}).get("median"),
        "ddp_step_sd_ms": ddp.get("step_time_ms", {}).get("std"),
        "comm_busy_ms": ddp.get("communication_time_ms", {}).get("median"),
        "exposed_timeline_ms": ddp.get("exposed_communication_time_ms", {}).get("median"),
        "ddp_finalize_ms": ddp.get("ddp_finalize_ms", {}).get("median"),
        "ablation_exposed_ms": abl.get("median"),
        "ablation_fraction": abl.get("fraction_of_ddp_step"),
        "comm_bytes_per_step": ddp.get("communication_bytes", {}).get("median"),
        "compute_interference_ms": summary.get("compute_interference_ms", {}).get("median"),
        "nccl": summary.get("nccl"),
        "gpu_state": summary.get("gpu_state"),
        "resolvable": meas.get("resolvable"),
        "material": meas.get("material"),
        "communication_relevant": meas.get("communication_relevant", False),
    }


def select(rows: list[dict], preferred_models: list[str], target_fraction: float = 0.25) -> dict:
    for model in preferred_models:
        cands = [r for r in rows if r["model"] == model and r["communication_relevant"]]
        if cands:
            best = min(cands, key=lambda r: abs(r["ablation_fraction"] - target_fraction))
            return {"selected": best["id"], "model": model,
                    "reason": (f"first preferred model with a communication-relevant config; "
                               f"ablation fraction {best['ablation_fraction']:.3f} closest to "
                               f"target {target_fraction}"),
                    "rejected_models": preferred_models[:preferred_models.index(model)]}
    return {"selected": None, "model": None,
            "reason": "no configuration met the criterion: spec stop condition A (change workload)",
            "rejected_models": list(preferred_models)}
