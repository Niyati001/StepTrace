"""Prepare CIFAR-10 in ONE process before any distributed launch.

    python scripts/prepare_data.py                      # cached -> /kaggle/input copy -> download
    python scripts/prepare_data.py --no-download        # never touch the network

Exit 0 only when every batch file is present with the expected MD5.
Exit 1 with an explicit reason otherwise (no distributed job is started).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from workloads.datasets import DEFAULT_SEARCH_ROOTS, DatasetUnavailable, prepare_cifar10  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--search-root", action="append", default=None,
                    help=f"where to look for an existing copy (default {list(DEFAULT_SEARCH_ROOTS)})")
    a = ap.parse_args()
    t0 = time.time()
    try:
        info = prepare_cifar10(a.root, download=not a.no_download,
                               search_roots=a.search_root or DEFAULT_SEARCH_ROOTS)
    except DatasetUnavailable as e:
        print(f"[prepare_data] CIFAR-10 UNAVAILABLE: {e}")
        return 1
    print(f"[prepare_data] CIFAR-10 ready ({json.dumps(info)}) in {time.time() - t0:.1f}s; all MD5s verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
