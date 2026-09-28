"""Course-named evaluation command; delegates to the existing evaluate CLI."""

import sys
from pathlib import Path

# Support both direct execution and python -m scripts.eval_runner.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.evaluate import main

if __name__ == "__main__":
    main()
