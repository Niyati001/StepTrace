"""Explicit diagnosis rules (spec §17-19). No learning, no LLM.

Per block:
  DATA_STALL     if data_wait_ms is ELEVATED
  STRAGGLER      if compute_skew_ms is ELEVATED
  COMMUNICATION  if exposed_min_ms is ELEVATED or exposed_min_frac is ELEVATED
  HEALTHY        if none of the above
If several causes are elevated, the primary cause is the one with the largest
estimated added critical-path time ("effect_ms"):
  DATA_STALL: delta(data_wait_ms); STRAGGLER: delta(compute_skew_ms);
  COMMUNICATION: max(delta(exposed_min_ms), delta(exposed_min_frac) * reference step).
The others are reported as secondary causes.

Per run: the verdict is the most frequent block verdict (ties broken by summed
effect_ms). Confidence is defined as rule agreement across the run's blocks, e.g.
"3/3 blocks". It is NOT a probability.
"""

from __future__ import annotations

from collections import Counter

from diagnose.thresholds import score_feature

CAUSES = ("COMMUNICATION", "STRAGGLER", "DATA_STALL")
SUPPORTING = ("step_ms", "comm_busy_ms", "comm_GBps", "compute_ms")


def classify_block(block: dict, stats: dict) -> dict:
    tests = {
        "DATA_STALL": [score_feature(block["data_wait_ms"], "data_wait_ms", stats)],
        "STRAGGLER": [score_feature(block["compute_skew_ms"], "compute_skew_ms", stats)],
        "COMMUNICATION": [score_feature(block["exposed_min_ms"], "exposed_min_ms", stats),
                          score_feature(block["exposed_min_frac"], "exposed_min_frac", stats)],
    }
    step_ref = stats["step_ms"]["center"]
    causes = {}
    for cause, ts in tests.items():
        effect = max(t["delta"] * (step_ref if t["feature"].endswith("_frac") else 1.0) for t in ts)
        causes[cause] = {"elevated": any(t["elevated"] for t in ts), "effect_ms": effect, "tests": ts}
    elevated = sorted((c for c in CAUSES if causes[c]["elevated"]),
                      key=lambda c: causes[c]["effect_ms"], reverse=True)
    supporting = {k: {"value": block[k], "reference_center": stats[k]["center"],
                      "change_pct": 100 * (block[k] - stats[k]["center"]) / stats[k]["center"]
                      if stats[k]["center"] else None} for k in SUPPORTING}
    return {"run_id": block["run_id"], "block_id": block["block_id"],
            "verdict": elevated[0] if elevated else "HEALTHY",
            "secondary": elevated[1:], "causes": causes, "supporting": supporting}


def classify_run(blocks: list[dict], stats: dict) -> dict:
    per_block = [classify_block(b, stats) for b in blocks]
    votes = Counter(b["verdict"] for b in per_block)
    effect = Counter()
    for b in per_block:
        if b["verdict"] != "HEALTHY":
            effect[b["verdict"]] += b["causes"][b["verdict"]]["effect_ms"]
    top = max(votes.values())
    tied = [v for v, n in votes.items() if n == top]
    verdict = max(tied, key=lambda v: effect.get(v, 0.0)) if len(tied) > 1 else tied[0]
    return {"verdict": verdict, "healthy": verdict == "HEALTHY",
            "agreement": f"{votes[verdict]}/{len(per_block)}",
            "agreement_fraction": votes[verdict] / len(per_block),
            "block_votes": dict(votes), "blocks": per_block}
