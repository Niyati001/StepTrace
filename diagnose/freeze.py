"""Freeze the diagnostic rules before any held-out execution (DECISIONS.md §Freeze).

    python -m diagnose.freeze --version v1 --design-evidence results/session1 [--design-evidence ...]

Refuses unless ALL of the following hold:
  * clean committed tree (the dirty-tree override is NOT honoured here);
  * diagnose/frozen/rules_<version>.json does not exist yet (frozen files are immutable;
    a later change needs a new version and a new tag, reported as a new experiment);
  * every design-evidence directory is ingested and passes SHA256 verification;
  * no held-out campaign evidence exists anywhere under results/ (held-out data must
    not exist before the freeze, so it cannot have influenced the rules).
Writes the frozen record and prints its SHA256 plus the exact git commands to commit
it and create the annotated tag ``evaluation-rules-frozen``. The evaluator refuses to
score held-out data unless given this file and its exact SHA256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnose import thresholds as T  # noqa: E402
from diagnose.diagnose import rules_fingerprint  # noqa: E402
from instrument.provenance import code_state  # noqa: E402

FROZEN_DIR = ROOT / "diagnose" / "frozen"
# Code that defines fault-manifestation criteria is frozen alongside the rules, so the
# "did not manifest" decision cannot be adjusted after seeing held-out results.
MANIFESTATION_FILES = ("faults/manifestation.py", "scripts/audit_campaign.py")
HELDOUT_ROLES = {"heldout"}


class FreezeError(RuntimeError):
    pass


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def heldout_evidence_present(results_root: Path) -> list[str]:
    hits = []
    for man in results_root.rglob("campaigns/*/manifest.json"):
        try:
            m = json.loads(man.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        roles = {p.get("role") for p in m.get("spec", {}).get("phases", [])} | \
                {r.get("role") for r in m.get("runs", [])}
        if roles & HELDOUT_ROLES:
            hits.append(str(man.relative_to(ROOT) if ROOT in man.resolve().parents else man))
    return hits


def freeze(version: str, design_evidence: list[str], results_root: Path = ROOT / "results",
           out_dir: Path = FROZEN_DIR, require_clean_tree: bool = True) -> tuple[Path, str]:
    from tools.ingest_results import IngestError, verify

    state = code_state()
    if require_clean_tree and state["dirty"]:
        raise FreezeError(f"freeze requires a clean committed tree: {state['dirty_paths'][:10]}")
    out = out_dir / f"rules_{version}.json"
    if out.exists():
        raise FreezeError(f"{out} already exists; frozen rules are immutable (use a new version)")
    held = heldout_evidence_present(results_root)
    if held:
        raise FreezeError(f"held-out evidence already exists before the freeze: {held}")
    evidence = []
    for d in design_evidence:
        root = (ROOT / d) if not Path(d).is_absolute() else Path(d)
        try:
            v = verify(root)
        except IngestError as e:
            raise FreezeError(f"design evidence {d} failed verification: {e}") from e
        info = json.loads((root / "INGEST.json").read_text(encoding="utf-8"))
        evidence.append({"path": d, "sha256sums_sha256": v["sha256sums_sha256"],
                         "campaign_ids": [c["campaign_id"] for c in info["campaigns"]]})
    fp = rules_fingerprint()
    record = {
        "version": version,
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_git_sha": state["git_sha"],
        "rule_files": fp["files"], "rules_sha256": fp["sha256"],
        "manifestation_files": {f: file_sha256(ROOT / f) for f in MANIFESTATION_FILES},
        "constants": {"K_ROBUST_Z": T.K_ROBUST_Z, "PRACTICAL_STEP_FRACTION": T.PRACTICAL_STEP_FRACTION,
                      "PRACTICAL_FRACTION_PP": T.PRACTICAL_FRACTION_PP, "SCALE_FLOOR_REL": T.SCALE_FLOOR_REL,
                      "SCALE_FLOOR_ABS": T.SCALE_FLOOR_ABS, "MIN_REFERENCE_BLOCKS": T.MIN_REFERENCE_BLOCKS},
        "design_evidence": evidence,
        "statement": "Thresholds and rules derived from design evidence only; frozen before any "
                     "held-out execution. Held-out data must not influence thresholds, rule logic or "
                     "mechanism selection after this point.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", newline="\n", encoding="utf-8")
    return out, hashlib.sha256(out.read_bytes()).hexdigest()


def check_frozen(path: str | Path, expected_sha256: str) -> dict:
    """Used by the evaluator: the frozen file must hash to the supplied value and the
    current rule + manifestation files must match what was frozen."""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        raise FreezeError(f"frozen rules file {p} not found")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise FreezeError(f"frozen rules file hash {actual} != supplied {expected_sha256}")
    rec = json.loads(p.read_text(encoding="utf-8"))
    fp = rules_fingerprint()
    if fp["sha256"] != rec["rules_sha256"]:
        raise FreezeError("current rule files differ from the frozen rules "
                          f"({fp['sha256'][:16]} != {rec['rules_sha256'][:16]})")
    for f, h in rec["manifestation_files"].items():
        if file_sha256(ROOT / f) != h:
            raise FreezeError(f"manifestation criteria file {f} changed since the freeze")
    return {"frozen_file": str(p), "frozen_file_sha256": actual, "version": rec["version"],
            "rules_sha256": rec["rules_sha256"], "frozen_utc": rec["frozen_utc"]}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # reports contain non-ASCII; never crash on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--design-evidence", action="append", required=True)
    a = ap.parse_args()
    try:
        out, h = freeze(a.version, a.design_evidence)
    except FreezeError as e:
        print(f"FREEZE REFUSED: {e}", file=sys.stderr)
        return 1
    rel = out.relative_to(ROOT).as_posix()
    print(f"frozen: {rel}\nsha256: {h}\n\nNext (exactly):")
    print(f"  git add {rel}")
    print(f'  git commit -m "Freeze diagnostic rules {a.version} (sha256 {h})"')
    print(f'  git tag -a evaluation-rules-frozen -m "{rel} sha256 {h}"')
    print(f"  python -m diagnose.evaluate ... --roles heldout --frozen-rules {rel} --frozen-rules-sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
