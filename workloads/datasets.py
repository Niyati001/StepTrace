"""Safe dataset acquisition for multi-rank jobs (CIFAR-10).

Design (why): acquiring data inside a distributed job and coordinating with a
collective barrier turns any acquisition problem on one rank (no internet,
stalled download, partial archive) into an unbounded hang on the other rank,
ended only by the NCCL watchdog. Here:

* Acquisition happens in ONE process, ideally before any distributed launch
  (``scripts/prepare_data.py``), with bounded network timeouts.
* "Available" means every expected file exists with the expected MD5. Nothing
  proceeds on mere directory existence.
* Order of sources: already-present copy -> a copy found under ``search_roots``
  (e.g. Kaggle ``/kaggle/input``) -> download (only if allowed). One download, cached.
* Inside a distributed job no collective is used for data. Each rank verifies
  locally; if an in-job preparer is enabled, other ranks poll file markers
  (ready / failed) with a bounded timeout, so failure is explicit and fast.
"""

from __future__ import annotations

import json
import shutil
import socket
import tarfile
import time
from pathlib import Path

BASE = "cifar-10-batches-py"
ARCHIVE = "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEFAULT_SEARCH_ROOTS = ("/kaggle/input",)
NET_TIMEOUT_S = 60          # per socket operation during download
WAIT_TIMEOUT_S = 600        # bound for non-preparer ranks waiting on an in-job preparer


class DatasetUnavailable(RuntimeError):
    pass


def expected_files() -> list[tuple[str, str]]:
    """(relative path, md5) for every CIFAR-10 python batch file, from torchvision's own table."""
    from torchvision.datasets import CIFAR10

    return [(f"{BASE}/{name}", md5) for name, md5 in CIFAR10.train_list + CIFAR10.test_list]


def _md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def missing_or_corrupt(root: str | Path) -> list[str]:
    root = Path(root)
    bad = []
    for rel, md5 in expected_files():
        p = root / rel
        if not p.is_file() or _md5(p) != md5:
            bad.append(rel)
    return bad


def is_available(root: str | Path) -> bool:
    return not missing_or_corrupt(root)


def _find_copy(search_roots) -> Path | None:
    """An extracted batches dir or the original archive somewhere under search_roots."""
    for sr in search_roots:
        sr = Path(sr)
        if not sr.is_dir():
            continue
        for p in sr.rglob(BASE):
            if p.is_dir() and (p / "data_batch_1").is_file():
                return p
        for p in sr.rglob(ARCHIVE):
            if p.is_file():
                return p
    return None


def _extract(archive: Path, root: Path) -> None:
    if _md5(archive) != ARCHIVE_MD5:
        raise DatasetUnavailable(f"archive {archive} has wrong MD5 (corrupt or partial download)")
    with tarfile.open(archive, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.name.startswith(BASE)]
        tf.extractall(root, members=members)


def _download(root: Path) -> Path:
    tmp = root / (ARCHIVE + ".part")
    dst = root / ARCHIVE
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(NET_TIMEOUT_S)
    try:
        import urllib.request

        with urllib.request.urlopen(URL) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise DatasetUnavailable(f"download failed ({type(e).__name__}: {e}); is internet enabled?") from e
    finally:
        socket.setdefaulttimeout(old)
    tmp.replace(dst)  # atomic: a partial file never carries the final name
    return dst


def prepare_cifar10(root: str | Path, download: bool, search_roots=DEFAULT_SEARCH_ROOTS) -> dict:
    """Make CIFAR-10 available under root. Single-process; raises DatasetUnavailable on failure."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if is_available(root):
        return {"source": "cached", "root": str(root)}
    source = None
    cached = root / ARCHIVE
    if cached.is_file():
        if _md5(cached) == ARCHIVE_MD5:
            _extract(cached, root)
            source = "cached_archive"
        else:  # partial/corrupt leftover (e.g. an interrupted earlier download): discard it
            cached.unlink()
    if source is None:
        found = _find_copy(search_roots)
        if found is not None and found.is_dir():
            shutil.copytree(found, root / BASE, dirs_exist_ok=True)
            source = f"copied:{found}"
        elif found is not None:
            _extract(found, root)
            source = f"extracted:{found}"
        elif download:
            _extract(_download(root), root)
            source = "downloaded"
    bad = missing_or_corrupt(root)
    if source is None:
        raise DatasetUnavailable(f"CIFAR-10 not found under {root} or {list(search_roots)} and download disabled")
    if bad:
        raise DatasetUnavailable(f"CIFAR-10 incomplete after {source}: missing/corrupt {bad}")
    return {"source": source, "root": str(root)}


# --------------------------------------------------------------------------- #
# In-job coordination (no collectives)
# --------------------------------------------------------------------------- #
def _markers(root: Path, tag: str) -> tuple[Path, Path]:
    return root / f".cifar10_ready.{tag}", root / f".cifar10_failed.{tag}"


def ensure_cifar10(root: str | Path, is_preparer: bool, tag: str, download: bool = False,
                   wait_timeout_s: float = WAIT_TIMEOUT_S, poll_s: float = 0.5,
                   search_roots=DEFAULT_SEARCH_ROOTS) -> None:
    """Called by every rank of one job before constructing the dataset.

    preparer: prepare (or verify) and publish a ready/failed marker for ``tag``.
    others:   fast path if already verified-available; otherwise wait for the
              preparer's marker, bounded by ``wait_timeout_s``; fail immediately
              when the preparer reports failure.
    ``tag`` must be unique per job (run id) so stale markers are never read.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    ready, failed = _markers(root, tag)
    if is_preparer:
        try:
            info = prepare_cifar10(root, download, search_roots)
        except Exception as e:
            failed.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            raise
        ready.write_text(json.dumps(info), encoding="utf-8")
        return
    if is_available(root):
        return
    deadline = time.monotonic() + wait_timeout_s
    while time.monotonic() < deadline:
        if failed.exists():
            raise DatasetUnavailable(f"preparer rank failed: {failed.read_text(encoding='utf-8')}")
        if ready.exists():
            bad = missing_or_corrupt(root)
            if bad:
                raise DatasetUnavailable(f"preparer reported ready but files missing/corrupt: {bad}")
            return
        time.sleep(poll_s)
    raise DatasetUnavailable(f"timed out after {wait_timeout_s:.0f}s waiting for CIFAR-10 preparation")
