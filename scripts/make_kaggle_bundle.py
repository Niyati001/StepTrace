"""Package the repository for Kaggle as a git bundle (one file, keeps the commit SHA).

    python scripts/make_kaggle_bundle.py          # -> dist/commscope.bundle + dist/BUNDLE_INFO.json

Refuses to run from a dirty tree. Verifies the bundle by cloning it to a temp dir
(LF checkout, as on Kaggle's Linux) and checking HEAD == current commit and a clean tree.
Upload dist/commscope.bundle as a Kaggle Dataset; the notebook clones from it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instrument.provenance import code_state, require_clean  # noqa: E402


def git(*a, cwd=ROOT) -> str:
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    state = require_clean()
    if state["allow_dirty_override"] and state["dirty"]:
        raise SystemExit("refusing to bundle a dirty tree even with the dev override")
    out = ROOT / "dist"
    out.mkdir(exist_ok=True)
    bundle = out / "commscope.bundle"
    git("bundle", "create", str(bundle), "--all")
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "c"
        git("-c", "core.autocrlf=false", "clone", "-q", str(bundle), str(clone), cwd=tmp)
        git("config", "core.autocrlf", "false", cwd=clone)
        head = git("rev-parse", "HEAD", cwd=clone)
        clean = code_state(clone)
        if head != state["git_sha"] or clean["dirty"]:
            raise SystemExit(f"bundle verification failed: head={head} expected={state['git_sha']} state={clean}")
    info = {"git_sha": state["git_sha"], "git_describe": state["git_describe"],
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bundle": str(bundle.relative_to(ROOT)), "bytes": bundle.stat().st_size,
            "verified_clone_head": head, "verified_clone_clean": True}
    (out / "BUNDLE_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
