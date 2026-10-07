"""CIFAR-10 acquisition and rank coordination, CPU-only, no network.

Uses tiny fake batch files whose MD5s replace torchvision's expected table, so
the verification and coordination logic is exercised exactly as in production.
"""

import hashlib
import io
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from workloads import datasets as D

NAMES = ["data_batch_1", "data_batch_2", "test_batch", "batches.meta"]
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake(monkeypatch):
    """Patch the expected-file table to tiny deterministic payloads."""
    payload = {n: f"payload-{n}".encode() for n in NAMES}
    table = [(f"{D.BASE}/{n}", hashlib.md5(b).hexdigest()) for n, b in payload.items()]
    monkeypatch.setattr(D, "expected_files", lambda: table)
    return payload


def write_tree(dst: Path, payload, corrupt=None):
    (dst / D.BASE).mkdir(parents=True, exist_ok=True)
    for n, b in payload.items():
        (dst / D.BASE / n).write_bytes(b"x" if n == corrupt else b)


def write_archive(path: Path, payload):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for n, b in payload.items():
            ti = tarfile.TarInfo(f"{D.BASE}/{n}")
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return hashlib.md5(buf.getvalue()).hexdigest()


def no_network(monkeypatch):
    def boom(_):
        raise AssertionError("network must not be used")
    monkeypatch.setattr(D, "_download", boom)


# ----------------------------------------------------------------------------- availability
def test_availability_requires_every_md5(tmp_path, fake):
    assert not D.is_available(tmp_path)
    write_tree(tmp_path, fake, corrupt="data_batch_2")
    assert D.missing_or_corrupt(tmp_path) == [f"{D.BASE}/data_batch_2"]
    write_tree(tmp_path, fake)
    assert D.is_available(tmp_path)


def test_already_cached_is_used_without_network(tmp_path, fake, monkeypatch):
    no_network(monkeypatch)
    write_tree(tmp_path, fake)
    assert D.prepare_cifar10(tmp_path, download=True, search_roots=())["source"] == "cached"


def test_copy_from_search_root_e_g_kaggle_input(tmp_path, fake, monkeypatch):
    no_network(monkeypatch)
    write_tree(tmp_path / "input" / "some-dataset", fake)
    info = D.prepare_cifar10(tmp_path / "data", download=True, search_roots=[tmp_path / "input"])
    assert info["source"].startswith("copied:") and D.is_available(tmp_path / "data")


def test_archive_in_search_root_is_verified_and_extracted(tmp_path, fake, monkeypatch):
    no_network(monkeypatch)
    md5 = write_archive(tmp_path / "input" / D.ARCHIVE, fake)
    monkeypatch.setattr(D, "ARCHIVE_MD5", md5)
    info = D.prepare_cifar10(tmp_path / "data", download=False, search_roots=[tmp_path / "input"])
    assert info["source"].startswith("extracted:") and D.is_available(tmp_path / "data")


def test_partial_cached_archive_is_discarded_then_downloaded_once(tmp_path, fake, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / D.ARCHIVE).write_bytes(b"truncated")          # leftover from an interrupted run
    good = tmp_path / "good.tar.gz"
    monkeypatch.setattr(D, "ARCHIVE_MD5", write_archive(good, fake))
    calls = []

    def fake_download(root):
        calls.append(root)
        dst = Path(root) / D.ARCHIVE
        dst.write_bytes(good.read_bytes())
        return dst

    monkeypatch.setattr(D, "_download", fake_download)
    assert D.prepare_cifar10(data, download=True, search_roots=())["source"] == "downloaded"
    assert len(calls) == 1
    # second call: cached, no further download
    assert D.prepare_cifar10(data, download=True, search_roots=())["source"] == "cached"
    assert len(calls) == 1


def test_unavailable_without_download_is_explicit(tmp_path, fake, monkeypatch):
    no_network(monkeypatch)
    with pytest.raises(D.DatasetUnavailable, match="download disabled"):
        D.prepare_cifar10(tmp_path, download=False, search_roots=())


# ----------------------------------------------------------------------------- coordination
def test_waiter_returns_after_preparer_publishes(tmp_path, fake, monkeypatch):
    src = tmp_path / "input"
    write_tree(src, fake)

    def slow_copy(root, download, search_roots):          # preparer takes a while
        time.sleep(0.5)
        return D.__dict__["_orig_prepare"](root, download, search_roots)

    monkeypatch.setitem(D.__dict__, "_orig_prepare", D.prepare_cifar10)
    monkeypatch.setattr(D, "prepare_cifar10", slow_copy)
    errors = []
    waiter = threading.Thread(target=lambda: _call(errors, D.ensure_cifar10, tmp_path / "data", False,
                                                   "run1", wait_timeout_s=10, poll_s=0.05))
    waiter.start()
    D.ensure_cifar10(tmp_path / "data", True, "run1", search_roots=[src])
    waiter.join(10)
    assert not waiter.is_alive() and errors == []


def test_waiter_fails_fast_when_preparer_fails(tmp_path, fake, monkeypatch):
    no_network(monkeypatch)
    errors = []
    t0 = time.monotonic()
    waiter = threading.Thread(target=lambda: _call(errors, D.ensure_cifar10, tmp_path, False, "run2",
                                                   wait_timeout_s=60, poll_s=0.05))
    waiter.start()
    with pytest.raises(D.DatasetUnavailable):
        D.ensure_cifar10(tmp_path, True, "run2", download=False, search_roots=())
    waiter.join(10)
    assert isinstance(errors[0], D.DatasetUnavailable) and "preparer rank failed" in str(errors[0])
    assert time.monotonic() - t0 < 5                       # not the 60 s bound, let alone NCCL's 10 min


def test_waiter_times_out_boundedly_without_preparer(tmp_path, fake):
    t0 = time.monotonic()
    with pytest.raises(D.DatasetUnavailable, match="timed out"):
        D.ensure_cifar10(tmp_path, False, "run3", wait_timeout_s=0.3, poll_s=0.05)
    assert time.monotonic() - t0 < 3


def test_stale_markers_from_other_runs_are_ignored(tmp_path, fake):
    (tmp_path / ".cifar10_ready.oldrun").write_text("{}", encoding="utf-8")
    with pytest.raises(D.DatasetUnavailable, match="timed out"):
        D.ensure_cifar10(tmp_path, False, "newrun", wait_timeout_s=0.2, poll_s=0.05)


def _call(errors, fn, *args, **kw):
    try:
        fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        errors.append(e)


# ----------------------------------------------------------------------------- 2-rank job
def test_two_rank_job_fails_fast_and_explicitly_when_cifar_missing(tmp_path):
    """Regression for Stage F: both ranks must exit with a clear error, not hang in a collective."""
    args = [sys.executable, "-m", "workloads.train", "--spawn", "2", "--out-root", str(tmp_path / "out"),
            "--set", "experiment.name=cifar_missing", "--set", "workload.model=cnn_tiny",
            "--set", "workload.dataset=cifar10", "--set", f"workload.data_root={(tmp_path / 'nodata').as_posix()}",
            "--set", "workload.num_workers=0", "--set", "measurement.warmup_steps=1",
            "--set", "measurement.measured_steps=1", "--set", "measurement.modes=[ddp]"]
    t0 = time.monotonic()
    r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=180,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "ignore"})
    assert r.returncode != 0
    assert "DatasetUnavailable" in r.stderr and "download disabled" in r.stderr
    assert time.monotonic() - t0 < 120
