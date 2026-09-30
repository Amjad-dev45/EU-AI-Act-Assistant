"""Entry point for the agent container (thin wrapper: no live-app imports)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from services.agent.server import main  # noqa: E402

main()
