"""Locating evidence files without rewriting them.

Campaign manifests reference runs as written at collection time, relative to the
repository root on the GPU host, e.g. ``results/raw/campaigns/<id>/<arm>/<run>.json``.
After ingestion the same files live under an evidence root such as
``results/session1/`` (the Kaggle zip's leading ``results/`` component is stripped).
Evidence files are never edited; references are resolved here instead.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def evidence_root(manifest_path: str | Path) -> Path:
    """<root>/campaigns/<campaign_id>/manifest.json -> <root>."""
    return Path(manifest_path).resolve().parents[2]


def resolve_run_file(manifest_path: str | Path, run_file: str) -> Path:
    rel = Path(run_file)
    root = evidence_root(manifest_path)
    parts = rel.parts[1:] if rel.parts and rel.parts[0] == "results" else rel.parts
    candidates = [root.joinpath(*parts), REPO_ROOT / rel, root / rel]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"run file {run_file!r} not found under evidence root {root} "
                            f"(tried {[str(c) for c in candidates]})")
