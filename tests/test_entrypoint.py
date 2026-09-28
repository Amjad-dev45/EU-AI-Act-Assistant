"""Check imports with the same search-path ordering as the direct entry point."""

from pathlib import Path
import subprocess
import sys
import unittest


class EntrypointTests(unittest.TestCase):
    def test_direct_script_path_resolves_application_package(self):
        repo = Path(__file__).resolve().parents[1]
        script = """
import sys
from pathlib import Path
repo = Path(sys.argv[1])
sys.path.insert(0, str(repo / 'app'))
sys.path.insert(0, str(repo))
import app
assert Path(app.__file__).resolve() == repo / 'app' / '__init__.py'
from app.rag.retrieval import Retriever
from app.rag.answering import Answerer
"""
        result = subprocess.run([sys.executable, "-c", script, str(repo)],
                                cwd=repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
