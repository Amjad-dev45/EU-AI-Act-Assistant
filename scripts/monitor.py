"""Report persisted RAG usage, costs, latency, quality and errors without API calls."""

import argparse
import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))
from app.monitoring import Monitor



# Command-line inputs, validation, execution, and output.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--conversation-id")
    parser.add_argument("--kind", choices=["interactive", "evaluation", "retrieval_check"])
    parser.add_argument("--recent", type=int, default=0, help="Include the last N metadata-only records")
    # Parse options before opening files or making API requests.
    args = parser.parse_args()
    if args.recent < 0:
        parser.error("--recent must be nonnegative")
    from dotenv import load_dotenv
    # Load credentials only for the requested runtime operation.
    load_dotenv(REPO_DIR / ".env")
    monitor = Monitor(args.db)
    output = {"summary": monitor.summary(args.conversation_id, args.kind)}
    if args.recent:
        output["recent_requests"] = monitor.records(args.conversation_id, args.kind)[-args.recent:]
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


# Running this file explicitly invokes the command.
if __name__ == "__main__":
    main()
