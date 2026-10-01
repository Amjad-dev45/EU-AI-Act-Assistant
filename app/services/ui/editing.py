"""Replace a user turn and discard its old answer and all later context."""

from .chat import MAX_MESSAGE_CHARS, _state, _failure, bounded_history
from .sharing import _with_sharing
from .source_viewer import _with_sources, submit_with_sources


def history_before_edit(session, index, previous_value, message):
    """Validate the selected turn before changing any conversation state."""
    state = _state(session)
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Enter a question before saving your edit.")
    if len(message) > MAX_MESSAGE_CHARS:
        raise ValueError("Keep your question within 2,000 characters.")
    messages = state.get("messages", [])
    if (type(index) is not int or not 0 <= index < len(messages)
            or len(messages) != len(state["display"])):
        raise ValueError("This message is no longer available to edit. Reload or start a new chat.")
    selected = messages[index]
    if (selected.get("role") != "user" or selected.get("content") != previous_value
            or state["display"][index] != selected):
        raise ValueError("The conversation changed. Select the message again to edit it.")
    # Rebuild context from this prefix, not the latest five messages: those may
    # belong to the discarded future. Raw answers exclude display-only sources.
    prefix = messages[:index]
    return {"history": bounded_history(prefix), "display": state["display"][:index],
            "messages": prefix}


def edit_with_sources(session, injection_test, index, previous_value, message):
    """Clear the old branch immediately, then generate through the normal path."""
    try:
        branch = history_before_edit(session, index, previous_value, message)
    except ValueError as error:
        # Restore the unchanged display if the browser optimistically edited it.
        # Leave its source preview, sharing state and diagnostics intact.
        unchanged = [{"__type__": "update"} for _ in range(15)]
        unchanged[1] = _state(session)["display"]
        unchanged[3] = str(error)
        yield tuple(unchanged)
        return

    pending = list(_with_sources(_with_sharing(_failure(
        "", branch, "Message edited. Later messages removed; generating a new answer…"))))
    pending[1] = branch["display"] + [{"role": "user", "content": message.strip()}]
    yield tuple(pending)
    # Failure retains the truncated branch and puts the edited text in the input
    # for retry; removed messages must never reappear or reach a later request.
    yield submit_with_sources(message, branch, injection_test)
