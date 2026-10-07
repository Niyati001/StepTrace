import subprocess

import pytest

from instrument import provenance as P


def git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@t")
    git(tmp_path, "config", "user.name", "t")
    git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "code.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_commit(repo):
    s = P.code_state(repo)
    assert s["git_sha"] and len(s["git_sha"]) == 40 and not s["dirty"]


def test_outputs_do_not_dirty_but_code_does(repo):
    (repo / "results" / "raw").mkdir(parents=True)
    (repo / "results" / "raw" / "run.json").write_text("{}", encoding="utf-8")
    (repo / "data").mkdir()
    (repo / "data" / "x.bin").write_text("x", encoding="utf-8")
    assert not P.code_state(repo)["dirty"]
    (repo / "helper.py").write_text("y = 2\n", encoding="utf-8")             # untracked source file
    assert P.code_state(repo)["dirty"]
    (repo / "helper.py").unlink()
    (repo / "code.py").write_text("x = 2\n", encoding="utf-8")               # modified tracked file
    s = P.code_state(repo)
    assert s["dirty"] and any("code.py" in p for p in s["dirty_paths"])


def test_no_repo_is_dirty(tmp_path):
    s = P.code_state(tmp_path)
    assert s["git_sha"] is None and s["dirty"]


def test_require_clean_fails_fast_without_override(monkeypatch):
    monkeypatch.delenv(P.OVERRIDE_ENV, raising=False)
    dirty = {"git_sha": "a" * 40, "dirty": True, "dirty_paths": ["M x.py"], "allow_dirty_override": False}
    with pytest.raises(P.DirtyTreeError, match="uncommitted changes"):
        P.require_clean(dirty)
    unknown = {"git_sha": None, "dirty": True, "dirty_paths": [], "allow_dirty_override": False}
    with pytest.raises(P.DirtyTreeError, match="no git commit"):
        P.require_clean(unknown)


def test_override_is_explicit_and_recorded(monkeypatch):
    monkeypatch.setenv(P.OVERRIDE_ENV, "1")
    s = P.require_clean()
    assert s["allow_dirty_override"] is True


def test_porcelain_parsing():
    out = " M workloads/train.py\n?? results/raw/a.json\n?? data/c.tgz\n?? new.py\nA  faults/spec.py\n"
    assert P.dirty_paths(out) == ["M workloads/train.py", "?? new.py", "A faults/spec.py"]
