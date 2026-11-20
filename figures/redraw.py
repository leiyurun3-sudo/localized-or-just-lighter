"""Compatibility entry for the figure command used in the original release."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "redraw.py"), run_name="__main__")
