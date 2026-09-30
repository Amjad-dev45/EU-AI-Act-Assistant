"""Entry point for the prepare container (thin wrapper: no live-app imports)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.rag.prepare import main  # noqa: E402

main()
