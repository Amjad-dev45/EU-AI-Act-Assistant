"""User-facing warnings from recorded guard decisions, never text guesses."""
from html import escape


def _object(value):
    return value if isinstance(value, dict) else {}


def _origins(blocked):
    """Combine repeated chunks/pages without inventing missing provenance."""
    documents = {}
    for row in blocked:
        name = row.get("source") or row.get("document_id")
        name = ' '.join(name.split())[:240] if isinstance(name, str) and name.strip() else "Unknown document"
        pages = row.get("pages", [])
        pages = pages if isinstance(pages, list) else []
        documents.setdefault(name, set()).update(p for p in pages if type(p) is int and p > 0)
    labels = []
    for name, pages in list(documents.items())[:3]:
        ordered = sorted(pages)
        location = ("page " if len(ordered) == 1 else "pages ") + ', '.join(map(str, ordered[:8])) if ordered else "page unavailable"
        if len(ordered) > 8:
            location += f" (+{len(ordered) - 8} more)"
        labels.append(f"{name} — {location}")
    if len(documents) > 3:
        labels.append(f"{len(documents) - 3} more documents (see tool details)")
    return '; '.join(labels) or "Unknown document — location not recorded"


# Names describe the actual guards, not the retrieval algorithms that found a hit.
DETECTION_STAGES = {
    "retrieval_screening": "Source screening after retrieval",
    "context_screening": "Answer-context screening",
    "generation_request": "Query check before answer generation",
    "generation_screening": "Final source check before answer generation",
    "agent_evidence": "Agent evidence validation",
    "unknown": "Source screening (stage not recorded)",
    "unknown_request": "RAG security check (stage not recorded)",
}
RETRIEVAL_STAGES = {"faiss": "FAISS", "bm25": "BM25", "rrf": "RRF",
                    "reranker": "Reranker", "selected": "Top-k selection"}


def tool_security_events(entry):
    """Associate a detection with its tool and the recorded candidate snapshots."""
    entry = _object(entry)
    data = _object(entry.get("diagnostics"))
    output = _object(entry.get("output"))
    if output.get("cached"):
        output = _object(output.get("result"))
    retrieval = _object(_object(data.get("retrieval")).get("trace"))
    snapshots = retrieval.get("stages", [])
    events = {}

    def event(stage):
        stage = stage if isinstance(stage, str) and stage in DETECTION_STAGES else "unknown"
        return events.setdefault(stage, {"stage": stage, "label": DETECTION_STAGES[stage],
                                         "blocked": [], "appeared_in": []})

    for container in (_object(data.get("context")), _object(data.get("answer")), output):
        security = _object(container.get("security"))
        blocked = security.get("blocked", [])
        for row in blocked if isinstance(blocked, list) else []:
            if not isinstance(row, dict) or not row.get("reasons"):
                continue
            current = event(row.get("stage", "unknown"))
            location = {k: row[k] for k in ("source", "document_id", "pages", "chunk_id", "reasons") if k in row}
            seen = []
            # Presence in a FAISS/BM25 candidate list is provenance, not proof
            # that those algorithms performed a prompt-injection check.
            for snapshot in snapshots if isinstance(snapshots, list) else []:
                snapshot = _object(snapshot)
                name = snapshot.get("name")
                results = snapshot.get("results", [])
                if (name in RETRIEVAL_STAGES and snapshot.get("enabled") and row.get("chunk_id")
                        and isinstance(results, list) and any(_object(hit).get("chunk_id") == row["chunk_id"] for hit in results)):
                    seen.append(RETRIEVAL_STAGES[name])
            location["appeared_in"] = list(dict.fromkeys(seen))
            if location not in current["blocked"]:
                current["blocked"].append(location)
            current["appeared_in"] = list(dict.fromkeys(current["appeared_in"] + seen))
        if security.get("request_risks"):
            event(security.get("request_stage", "unknown_request"))["request_risks"] = security["request_risks"]
    if output.get("error") == "SourceInjectionError" and not events:
        event("agent_evidence")
    if _object(data.get("answer")).get("refusal_reason") == "prompt_injection" and not events:
        event("unknown_request")
    return list(events.values())


def security_event_description(event):
    """Plain-text explanation shared by the warning and expandable tool substep."""
    if event.get("stage") in {"generation_request", "unknown_request"}:
        return "The corpus answer was blocked by the prompt-injection guard."
    text = "Source: " + _origins(event.get("blocked", [])) + ". Blocked from use as evidence."
    if event.get("appeared_in"):
        text += " Flagged passages appeared in: " + ", ".join(event["appeared_in"]) + ". These are retrieval stages; detection occurred at the guard step."
    return text


def injection_notice(response):
    """Describe the origin and exact guard step, using recorded metadata only."""
    trace = _object(response).get("trace", [])
    if not isinstance(trace, list):
        return ""
    blocked_sources, locations = [], []
    source_blocked = request_blocked = False
    call_number = 0
    for entry in trace:
        entry = _object(entry)
        if entry.get("type") == "security" and entry.get("action") == "blocked_user_override":
            return "Step: input check. Source: your message. Your message was blocked before any model or tool call."
        if entry.get("type") != "tool":
            continue
        call_number += 1
        for event in tool_security_events(entry):
            blocked_sources.extend(event["blocked"])
            request_blocked |= event["stage"] in {"generation_request", "unknown_request"}
            source_blocked |= event["stage"] not in {"generation_request", "unknown_request"}
            location = f"Tool call {call_number} ({entry.get('name', 'tool')}) → {event['label']}"
            if entry.get("cached") or _object(entry.get("output")).get("cached"):
                location += " (reused recorded result)"
            if event["appeared_in"]:
                location += "; flagged passages appeared in " + ", ".join(event["appeared_in"])
            locations.append(location + ".")
    if source_blocked:
        notice = "Source: " + _origins(blocked_sources) + ". Suspicious instructions in retrieved material were blocked from use as evidence."
    elif request_blocked:
        notice = "Source: corpus tool request or evidence. The corpus answer was blocked by the prompt-injection guard."
    else:
        return ""
    return notice + " " + " ".join(locations)


def notify_injection(response):
    """Gradio popup; keep framework imports out of pure formatting/callback tests."""
    notice = injection_notice(response)
    if notice:
        import gradio as gr
        gr.Warning(f"Prompt injection detected: {escape(notice)}", title="Prompt injection detected", duration=10)
