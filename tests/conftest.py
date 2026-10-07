import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Tests run from development trees; the documented override is required and recorded.
os.environ.setdefault("COMMSCOPE_ALLOW_DIRTY", "1")
