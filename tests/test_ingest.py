"""tools/ingest_results: safe extraction, deterministic SHA256 manifest, verification (CPU only)."""

import hashlib
import json
import shutil
import subprocess
import zipfile

import pytest

from tools import ingest_results as I

RUN = {"run_id": "r0", "steps": [], "note": "synthetic test fixture, not evidence"}


def manifest(run_files):
    return {"campaign_id": "c1-20990101", "provenance": {"git_sha": "a" * 40, "dirty": False},
            "spec": {"phases": [{"arms": [{"id": "arm", "launches": len(run_files)}]}]},
            "runs": [{"arm": "arm", "launch": i, "status": "ok", "run_file": f}
                     for i, f in enumerate(run_files)]}


def make_zip(path, entries, stored=False):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED) as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return path


def good_entries():
    rf = "results/raw/campaigns/c1-20990101/arm/r0.json"
    return {
        "results/campaigns/c1-20990101/manifest.json": json.dumps(manifest([rf])),
        "results/campaigns/c1-20990101/audit.md": "# audit (fixture)\n",
        rf: json.dumps(RUN),
        "results/raw/campaigns/c1-20990101/arm/r0_steps.csv": "a,b\n1,2\n",
    }


def ingest(zip_path, dest):
    return I.main(["zip", str(zip_path), "--dest", str(dest), "--label", "fixture"])


def test_zip_ingest_manifest_and_verify(tmp_path):
    z = make_zip(tmp_path / "r.zip", good_entries())
    dest = tmp_path / "ev"
    assert ingest(z, dest) == 0
    # leading results/ stripped; bytes identical to the archive
    assert (dest / "campaigns/c1-20990101/manifest.json").is_file()
    with zipfile.ZipFile(z) as zz:
        for name in zz.namelist():
            assert (dest / name[len("results/"):]).read_bytes() == zz.read(name)
    sums = (dest / I.SUMS).read_text(encoding="utf-8").splitlines()
    assert [ln.split("  ")[1] for ln in sums] == sorted(ln.split("  ")[1] for ln in sums)
    for ln in sums:
        h, rel = ln.split("  ")
        assert hashlib.sha256((dest / rel).read_bytes()).hexdigest() == h
    info = json.loads((dest / I.INFO).read_text(encoding="utf-8"))
    assert info["input"]["zip_sha256"] == hashlib.sha256(z.read_bytes()).hexdigest()
    c = info["campaigns"][0]
    assert c["run_files_resolved"] == 1 and not c["run_files_missing"] and c["audit_files"] == ["audit.md"]
    assert I.main(["verify", str(dest)]) == 0


def test_manifest_is_deterministic(tmp_path):
    z = make_zip(tmp_path / "r.zip", good_entries())
    ingest(z, tmp_path / "a")
    ingest(z, tmp_path / "b")
    assert (tmp_path / "a" / I.SUMS).read_bytes() == (tmp_path / "b" / I.SUMS).read_bytes()


@pytest.mark.parametrize("tamper", ["modify", "delete", "extra", "sums"])
def test_verify_detects_tampering(tmp_path, tamper):
    dest = tmp_path / "ev"
    ingest(make_zip(tmp_path / "r.zip", good_entries()), dest)
    target = dest / "campaigns/c1-20990101/audit.md"
    if tamper == "modify":
        target.write_text("# edited\n", encoding="utf-8")
    elif tamper == "delete":
        target.unlink()
    elif tamper == "extra":
        (dest / "raw/new.json").write_text("{}", encoding="utf-8")
    else:
        sums = dest / I.SUMS
        sums.write_text(sums.read_text(encoding="utf-8").replace("audit.md", "audit.MD"), encoding="utf-8")
    assert I.main(["verify", str(dest)]) == 1


def test_never_overwrites_existing_evidence(tmp_path):
    dest = tmp_path / "ev"
    dest.mkdir()
    (dest / "keep.txt").write_text("existing evidence", encoding="utf-8")
    assert ingest(make_zip(tmp_path / "r.zip", good_entries()), dest) == 1
    assert (dest / "keep.txt").read_text(encoding="utf-8") == "existing evidence"
    assert sorted(p.name for p in dest.iterdir()) == ["keep.txt"]


def test_zip_slip_rejected(tmp_path):
    e = good_entries()
    e["results/../../evil.txt"] = "x"
    assert ingest(make_zip(tmp_path / "r.zip", e), tmp_path / "ev") == 1
    assert not (tmp_path / "ev").exists() and not (tmp_path / "evil.txt").exists()


def test_corrupt_archive_rejected(tmp_path):
    z = make_zip(tmp_path / "r.zip", good_entries(), stored=True)
    data = bytearray(z.read_bytes())
    i = data.find(b"fixture")                      # inside a stored file body
    data[i] ^= 0xFF
    z.write_bytes(bytes(data))
    assert ingest(z, tmp_path / "ev") == 1
    assert not (tmp_path / "ev").exists()


def test_incomplete_campaign_rejected_before_anything_lands(tmp_path):
    e = good_entries()
    del e["results/raw/campaigns/c1-20990101/arm/r0.json"]   # manifest references it
    assert ingest(make_zip(tmp_path / "r.zip", e), tmp_path / "ev") == 1
    assert not (tmp_path / "ev").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".ingest-")]  # temp cleaned up


def test_dir_mode_and_no_restamping(tmp_path):
    src = tmp_path / "unz"
    with zipfile.ZipFile(make_zip(tmp_path / "r.zip", good_entries())) as z:
        z.extractall(src)
    root = src / "results"
    assert I.main(["dir", str(root), "--label", "placed by hand"]) == 0
    assert I.main(["verify", str(root)]) == 0
    assert I.main(["dir", str(root), "--label", "again"]) == 1     # never re-stamped


def test_sha256sum_compatible(tmp_path):
    if shutil.which("sha256sum") is None:
        pytest.skip("sha256sum not available")
    dest = tmp_path / "ev"
    ingest(make_zip(tmp_path / "r.zip", good_entries()), dest)
    r = subprocess.run(["sha256sum", "-c", "--quiet", I.SUMS], cwd=dest, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
