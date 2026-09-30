"""Load saved RAG answers without regenerating questions, labels, or evidence."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def passages(hits):
    if not isinstance(hits, list) or any(not isinstance(h, dict) or not isinstance(h.get("text"), str)
                                         or not h["text"].strip() for h in hits):
        raise ValueError("Saved context must contain full nonempty passage text.")
    return [hit["text"] for hit in hits]


def load_samples(data_dir, numbers):
    """Newest matching saved run per question; never substitute current retrieval."""
    data_dir = Path(data_dir)
    questions = read_json(data_dir / "questions.json")
    if any(type(n) is not int or n < 1 or n > len(questions) for n in numbers):
        raise ValueError(f"Question numbers must be between 1 and {len(questions)}.")
    wanted = {questions[n - 1]["question"] for n in numbers}
    latest = {}
    for path in sorted((data_dir / "answer_runs").glob("*.json"), reverse=True):
        run = read_json(path)
        question = run.get("question")
        if question in wanted and question not in latest:
            latest[question] = (path, run)
    samples = []
    for n in numbers:
        label = questions[n - 1]
        item = {"question_number": n, "user_input": label["question"], "reference": label.get("expected_answer"),
                "category": label.get("category"), "skip_reason": None}
        saved = latest.get(label["question"])
        if not saved:
            item["skip_reason"] = "missing_saved_answer"
        else:
            path, run = saved
            item.update(answer_file=str(path), answer_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        answer_status=run.get("status"), response=run.get("answer"),
                        generator_model=run.get("model"), prompt_version=run.get("prompt_version"),
                        generator_code_sha256=run.get("code_sha256"))
            if run.get("question_number") not in (None, n):
                raise ValueError(f"Saved answer's question number does not match question {n}.")
            if (label.get("category") in {"ambiguous", "out-of-scope"}
                    or label.get("sub_type") == "on_topic_unanswerable"):
                item["skip_reason"] = "refusal_or_ambiguity_case_requires_manual_review"
            elif run.get("status") != "answered":
                item["skip_reason"] = "answer_not_completed_requires_manual_review"
            elif not all(isinstance(item[key], str) and item[key].strip() for key in ("response", "reference")):
                item["skip_reason"] = "missing_response_or_reference"
            else:
                # Faithfulness sees all actual generation evidence, including section expansion.
                # Retriever metrics see only the original ranked hits, in original order.
                item["generation_contexts"] = passages(run.get("answer_context", {}).get("hits"))
                item["retrieved_contexts"] = passages(run.get("retrieval", {}).get("hits"))
                if not item["generation_contexts"] or not item["retrieved_contexts"]:
                    item["skip_reason"] = "missing_saved_context"
        samples.append(item)
    return samples
