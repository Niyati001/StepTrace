"""Print and save the detected execution environment.

    python scripts/detect_environment.py [--out results/raw/env/env_<ts>.json]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instrument.environment import detect, format_report, save  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="JSON output path")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    env = detect()
    print(format_report(env))
    if not args.no_save:
        out = args.out or f"results/raw/env/env_{time.strftime('%Y%m%d-%H%M%S')}.json"
        print(f"\nSaved: {save(env, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
