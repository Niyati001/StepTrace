"""Ingest GPU evidence (e.g. a Kaggle results zip) with SHA256 provenance.

    python -m tools.ingest_results zip <results.zip> --dest results/session2 --label "M2 Session 2"
    python -m tools.ingest_results dir results/session1 --label "M2 Session 1"   # already-extracted folder
    python -m tools.ingest_results verify results/session1

zip mode:  rejects unsafe entries (absolute paths, '..', symlinks); checks every CRC
           before extracting; extracts into a temporary sibling directory; checks file
           count and sizes against the archive; strips a common leading ``results/``
           component; then moves the tree into --dest. --dest must not exist (evidence is
           never overwritten).
dir mode:  for evidence that was placed by hand; refuses if a manifest already exists.
Both:      every run file referenced by any campaigns/*/manifest.json must be present
           (incomplete extraction fails); writes SHA256SUMS (sha256sum format, sorted
           POSIX paths) and INGEST.json (provenance); then immediately re-verifies.
verify:    recomputes every hash; fails on any mismatch, missing or unexpected file, or
           if SHA256SUMS itself no longer matches the hash recorded in INGEST.json.
Exit status 0 only on success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from instrument.evidence import resolve_run_file  # noqa: E402
from instrument.provenance import code_state  # noqa: E402

SUMS, INFO = "SHA256SUMS", "INGEST.json"
META = {SUMS, INFO}


class IngestError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_files(root: Path) -> list[str]:
    out = []
    for p in root.rglob("*"):
        if p.is_symlink():
            raise IngestError(f"symlink in evidence tree: {p}")
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel not in META:
                out.append(rel)
    return sorted(out)


def build_sums(root: Path) -> str:
    return "".join(f"{sha256_file(root / rel)}  {rel}\n" for rel in evidence_files(root))


# --------------------------------------------------------------------------- zip handling
def _safe_members(z: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    for info in z.infolist():
        name = info.filename
        p = PurePosixPath(name)
        mode = (info.external_attr >> 16) & 0o170000
        if p.is_absolute() or ".." in p.parts or ":" in p.parts[0] or name.startswith(("/", "\\")):
            raise IngestError(f"unsafe path in archive: {name!r}")
        if mode == stat.S_IFLNK:
            raise IngestError(f"symlink in archive: {name!r}")
        if not info.is_dir():
            members.append(info)
    if not members:
        raise IngestError("archive contains no files")
    bad = z.testzip()
    if bad is not None:
        raise IngestError(f"CRC check failed for {bad!r} (corrupt or truncated archive)")
    return members


def _strip_prefix(members) -> str:
    firsts = {PurePosixPath(m.filename).parts[0] for m in members}
    return "results/" if firsts == {"results"} else ""


def extract_zip(zip_path: Path, dest: Path) -> dict:
    if dest.exists():
        raise IngestError(f"destination {dest} already exists; refusing to overwrite evidence")
    with zipfile.ZipFile(zip_path) as z:
        members = _safe_members(z)
        prefix = _strip_prefix(members)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".ingest-", dir=dest.parent))
        try:
            for m in members:
                rel = m.filename[len(prefix):]
                target = tmp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                if target.stat().st_size != m.file_size:
                    raise IngestError(f"size mismatch after extraction: {rel}")
            got = evidence_files(tmp)
            want = sorted(m.filename[len(prefix):] for m in members)
            if got != want:
                raise IngestError(f"incomplete extraction: {len(got)} files extracted, {len(want)} in archive")
            missing = [c for c in campaign_check(tmp) if c["run_files_missing"]]
            if missing:  # checked before anything appears at --dest
                raise IngestError("incomplete evidence: campaign manifests reference missing run files: " +
                                  "; ".join(f"{c['campaign_id']}: {c['run_files_missing'][:5]}" for c in missing))
            os.replace(tmp, dest)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return {"source": "zip", "zip_name": zip_path.name, "zip_sha256": sha256_file(zip_path),
            "zip_bytes": zip_path.stat().st_size, "zip_members": len(members), "stripped_prefix": prefix}


# --------------------------------------------------------------------------- campaign completeness
def campaign_check(root: Path) -> list[dict]:
    out = []
    for man_path in sorted(root.glob("campaigns/*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        missing, resolved = [], 0
        for r in man.get("runs", []):
            if r.get("status") != "ok":
                continue
            try:
                p = resolve_run_file(man_path, r["run_file"])
                if root.resolve() not in p.resolve().parents:
                    raise FileNotFoundError("resolved outside the evidence root")
                resolved += 1
            except FileNotFoundError:
                missing.append(r["run_file"])
        expected = sum(a["launches"] for p in man["spec"]["phases"] for a in p["arms"]) \
            if "spec" in man else None
        cdir = man_path.parent
        out.append({"manifest": man_path.relative_to(root).as_posix(), "campaign_id": man.get("campaign_id"),
                    "campaign_git_sha": (man.get("provenance") or {}).get("git_sha"),
                    "campaign_dirty": (man.get("provenance") or {}).get("dirty"),
                    "launches_in_manifest": len(man.get("runs", [])), "launches_expected": expected,
                    "ok_runs": sum(r.get("status") == "ok" for r in man.get("runs", [])),
                    "run_files_resolved": resolved, "run_files_missing": missing,
                    "audit_files": sorted(p.name for p in cdir.glob("audit*")),
                    "evaluation_files": sorted(p.name for p in cdir.glob("evaluation*"))})
    return out


# --------------------------------------------------------------------------- manifest + verify
def write_manifest(root: Path, provenance: dict, label: str) -> dict:
    camps = campaign_check(root)
    incomplete = [c for c in camps if c["run_files_missing"]]
    if incomplete:
        raise IngestError("incomplete evidence: campaign manifests reference missing run files: " +
                          "; ".join(f"{c['campaign_id']}: {c['run_files_missing'][:5]}" for c in incomplete))
    sums = build_sums(root)
    (root / SUMS).write_text(sums, newline="\n", encoding="utf-8")
    files = evidence_files(root)
    info = {
        "label": label, "ingested_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_root": root.relative_to(ROOT).as_posix() if ROOT in root.resolve().parents else str(root),
        "input": provenance, "file_count": len(files),
        "total_bytes": sum((root / f).stat().st_size for f in files),
        "sha256sums_sha256": hashlib.sha256(sums.encode()).hexdigest(),
        "campaigns": camps,
        "ingest_tool_code": {k: v for k, v in code_state().items() if k in ("git_sha", "dirty")},
        "note": "Evidence files are byte-identical to the input; nothing was edited. "
                "Campaign run_file paths are resolved relative to this evidence root "
                "(instrument/evidence.py).",
    }
    (root / INFO).write_text(json.dumps(info, indent=2) + "\n", newline="\n", encoding="utf-8")
    return info


def verify(root: Path) -> dict:
    sums_path, info_path = root / SUMS, root / INFO
    if not sums_path.is_file() or not info_path.is_file():
        raise IngestError(f"{root} has no {SUMS}/{INFO}; not an ingested evidence directory")
    text = sums_path.read_bytes().decode()
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if hashlib.sha256(text.encode()).hexdigest() != info["sha256sums_sha256"]:
        raise IngestError(f"{SUMS} was modified after ingestion (hash differs from {INFO})")
    listed = {}
    for line in text.splitlines():
        h, rel = line.split("  ", 1)
        listed[rel] = h
    present = set(evidence_files(root))
    missing = sorted(set(listed) - present)
    extra = sorted(present - set(listed))
    mismatched = sorted(rel for rel in set(listed) & present if sha256_file(root / rel) != listed[rel])
    if missing or extra or mismatched:
        raise IngestError(f"verification FAILED: missing {missing[:10]}, unexpected {extra[:10]}, "
                          f"checksum mismatch {mismatched[:10]}")
    return {"verified_files": len(listed), "sha256sums_sha256": info["sha256sums_sha256"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    z = sub.add_parser("zip")
    z.add_argument("zip")
    z.add_argument("--dest", required=True)
    z.add_argument("--label", required=True)
    d = sub.add_parser("dir")
    d.add_argument("path")
    d.add_argument("--label", required=True)
    v = sub.add_parser("verify")
    v.add_argument("path")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "zip":
            dest = (ROOT / a.dest) if not Path(a.dest).is_absolute() else Path(a.dest)
            prov = extract_zip(Path(a.zip), dest)
            info = write_manifest(dest, prov, a.label)
            root = dest
        elif a.cmd == "dir":
            root = (ROOT / a.path) if not Path(a.path).is_absolute() else Path(a.path)
            if not root.is_dir():
                raise IngestError(f"{root} is not a directory")
            if (root / SUMS).exists() or (root / INFO).exists():
                raise IngestError(f"{root} already has a manifest; use 'verify' (evidence is never re-stamped)")
            info = write_manifest(root, {"source": "directory placed by hand"}, a.label)
        else:
            root = (ROOT / a.path) if not Path(a.path).is_absolute() else Path(a.path)
            print(json.dumps(verify(root), indent=2))
            print("VERIFIED")
            return 0
        res = verify(root)  # immediate re-verification
        print(json.dumps({k: info[k] for k in ("label", "evidence_root", "file_count", "total_bytes",
                                                "sha256sums_sha256", "campaigns")}, indent=2))
        print(f"INGESTED + VERIFIED: {res['verified_files']} files")
        return 0
    except IngestError as e:
        print(f"INGEST FAILED: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
