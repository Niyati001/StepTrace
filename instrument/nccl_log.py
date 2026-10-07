"""Capture and parse NCCL's own INFO logs (runtime version, transport per channel).

The transport is whatever NCCL reports at communicator init, never inferred
from topology. Capture is enabled by environment variables that NCCL reads at
communicator creation, so ``enable`` must run before the first collective.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def enable(log_dir: Path, subsys: str = "INIT,GRAPH,ENV") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", subsys)
    os.environ.setdefault("NCCL_DEBUG_FILE", str((log_dir / "nccl_%h_%p.log").resolve()))


def parse_text(text: str) -> dict:
    via, version, p2p_type = set(), None, None
    for line in text.splitlines():
        msg = re.sub(r"^.*?NCCL INFO ", "", line).strip()
        if " via " in msg:
            via.add(msg)
        m = re.match(r"NCCL version (\S+)", msg)
        if m and version is None:
            version = m.group(1)
        if msg.startswith("Check P2P Type"):
            p2p_type = msg
    transports = sorted({m.group(1) for v in via for m in [re.search(r" via (\S+)", v)] if m})
    return {"runtime_version": version, "transports": transports,
            "via_lines": sorted(via)[:64], "p2p_check": p2p_type}


def parse_dir(log_dir: Path) -> dict:
    files = sorted(Path(log_dir).glob("nccl_*.log"))
    merged = parse_text("\n".join(f.read_text(errors="ignore", encoding="utf-8") for f in files))
    merged["log_files"] = [str(f) for f in files]
    return merged
