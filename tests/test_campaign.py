"""Integration: a tiny CPU campaign records exactly the seeds and provenance that the runs used."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPEC = """
name: t_campaign
seed: 500
workload_point: r18_b32_fp32
base:
  measurement: {warmup_steps: 1, measured_steps: 3, repeats: 3, modes: [ddp], profile: {enabled: false}}
phases:
  - name: reference
    role: reference
    arms:
      - {id: healthy_ref, launches: 2}
  - name: design
    role: design
    arms:
      - {id: straggler, launches: 2, fault: {mechanism: sleep, rank: 1, delay_ms: {rel_step: 0.5}}}
"""


def test_campaign_manifest_matches_runs(tmp_path):
    spec = tmp_path / "t.yaml"
    spec.write_text(SPEC, encoding="utf-8")
    env = {**os.environ, "COMMSCOPE_ALLOW_DIRTY": "1", "PYTHONWARNINGS": "ignore", "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, "scripts/run_campaign.py", "--campaign", str(spec), "--cpu",
                        "--set", "workload.model=cnn_tiny", "--set", "workload.batch_size=8"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    man_path = sorted((ROOT / "results/dev/campaigns").glob("t_campaign-*/manifest.json"))[-1]
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert len(man["runs"]) == 4 and all(x["status"] == "ok" for x in man["runs"])
    seeds = []
    for x in man["runs"]:
        doc = json.loads((ROOT / x["run_file"]).read_text(encoding="utf-8"))
        assert doc["config"]["experiment"]["seed"] == x["seed"]          # regression: seed metadata
        assert doc["provenance"]["git_sha"] == man["provenance"]["git_sha"]
        seeds.append(x["seed"])
        if x["arm"] == "straggler":
            # magnitude resolved from the measured reference step, recorded in both places
            assert doc["config"]["fault"]["delay_ms"] == man["resolved"]["straggler"]["delay_ms"] > 0
    assert len(set(seeds)) == len(seeds)
    assert man["reference"]["n_blocks"] == 6
