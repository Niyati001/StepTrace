"""Code provenance for every experiment artifact.

Invariant: an official run records the exact git commit it was produced by,
and refuses to start from a dirty tree.

* dirty = any modified/staged/deleted TRACKED file, or any UNTRACKED file outside
  the output directories (results/, data/). Output files never make the tree dirty;
  a stray source file does.
* No git repository / no commit -> treated as dirty (provenance unknown).
* The only override is the environment variable COMMSCOPE_ALLOW_DIRTY=1, intended
  for development and tests. It is recorded in the run (``allow_dirty_override``)
  and such runs must not be used as evidence.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRS = ("results/", "data/")
OVERRIDE_ENV = "COMMSCOPE_ALLOW_DIRTY"


class DirtyTreeError(RuntimeError):
    pass


def _git(*args: str, root: Path = REPO_ROOT) -> str | None:
    if shutil.which("git") is None:
        return None
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def dirty_paths(porcelain: str) -> list[str]:
    """Paths that make the tree dirty, from `git status --porcelain` output."""
    out = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        code, path = line[:2], line[3:].strip().strip('"')
        if code == "??" and path.startswith(OUTPUT_DIRS):
            continue
        out.append(f"{code.strip()} {path}")
    return out


def code_state(root: Path = REPO_ROOT) -> dict:
    sha = _git("rev-parse", "HEAD", root=root)
    sha = sha.strip() if sha else None
    porcelain = _git("status", "--porcelain", "--untracked-files=all", root=root) if sha else None
    dirty = dirty_paths(porcelain) if porcelain is not None else []
    return {
        "git_sha": sha,
        "git_describe": (_git("describe", "--tags", "--always", root=root) or "").strip() or None,
        "dirty": (sha is None) or bool(dirty),
        "dirty_paths": dirty[:50],
        "allow_dirty_override": os.environ.get(OVERRIDE_ENV) == "1",
    }


def require_clean(state: dict | None = None) -> dict:
    """Fail fast on a dirty/unknown tree unless the documented override is set."""
    state = state or code_state()
    if state["dirty"] and not state["allow_dirty_override"]:
        why = "no git commit found" if state["git_sha"] is None else \
            f"uncommitted changes: {state['dirty_paths'][:10]}"
        raise DirtyTreeError(
            f"refusing to run: working tree is not a clean commit ({why}). Commit first, or set "
            f"{OVERRIDE_ENV}=1 for a development run (recorded; not valid as evidence).")
    return state
