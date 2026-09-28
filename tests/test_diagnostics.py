"""Verify local error details retain useful traces while redacting credentials."""

import os
import unittest
from unittest.mock import patch

from app.ui.diagnostics import error_details


class DiagnosticsTests(unittest.TestCase):
    def test_chained_errors_are_visible_with_secrets_redacted_and_html_escaped(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "private-test-credential"}):
            try:
                try:
                    raise ConnectionRefusedError("Cannot connect to 127.0.0.1:9; private-test-credential")
                except ConnectionRefusedError as cause:
                    raise RuntimeError("<script>bad</script> Bearer hidden-token sk-example-secret") from cause
            except RuntimeError as exc:
                exc.rag_request_id = "request-test"
                with self.assertLogs("app.ui.diagnostics", level="ERROR") as logs:
                    details = error_details(exc)
        rendered = details["html"] + details["summary"] + "".join(logs.output)
        self.assertIn("ConnectionRefusedError", details["html"])
        self.assertIn("RuntimeError", details["html"])
        self.assertIn("127.0.0.1:9", details["html"])
        self.assertIn("Traceback", details["html"])
        self.assertEqual(details["error_id"], "request-test")
        self.assertNotIn("<script>", details["html"])
        for secret in ("private-test-credential", "hidden-token", "sk-example-secret"):
            self.assertNotIn(secret, rendered)


if __name__ == "__main__":
    unittest.main()
