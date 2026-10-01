"""Explicit click-to-share callbacks. The UI never receives the webhook URL."""

from contracts.httpkit import call_json
from contracts.schemas import new_request_id
from .chat import AGENT_URL, new_chat, submit_chat

READY_LABEL = "Send to Teams"
DEFAULT_MESSAGE = "Get an answer, then click Send to Teams to share it with the configured channel."


def _button(enabled=False, label=READY_LABEL):
    # Gradio component updates without importing Gradio into callback unit tests.
    return {"__type__": "update", "interactive": enabled, "value": label}


def submit_with_sharing(message, session, injection_test=False):
    outputs = list(submit_chat(message, session, injection_test))
    # Keep the approval token only in server-side session state, not the JSON
    # inspector, conversation history, prompts, or subsequent model requests.
    details = dict(outputs[6])
    share = details.pop("sharing", {})
    outputs[6] = details
    ready = bool(details.get("status") == "answered" and share.get("available") and share.get("token"))
    state = {"token": share["token"], "request_id": details.get("request_id")} if ready else {}
    return *outputs, state, _button(ready), share.get("message", DEFAULT_MESSAGE)


def clear_with_sharing():
    return *new_chat(), {}, _button(), DEFAULT_MESSAGE, False


def send_to_teams(state, displayed_response):
    """Only the button invokes this; generation and reset never perform a POST."""
    if not isinstance(state, dict) or not state.get("token") or state.get("attempted"):
        return state or {}, _button(), "Get a new answer before sharing."
    # Gradio captures this visible response when the user clicks. If a queued
    # question finishes before this handler runs, do not send its unseen answer.
    if not isinstance(displayed_response, dict) or displayed_response.get("request_id") != state.get("request_id"):
        return state, _button(True), "The answer changed while you clicked. Review the latest answer, then click again."
    attempted = {**state, "attempted": True}
    try:
        status, body = call_json("POST", f"{AGENT_URL}/v1/share/teams",
                                 {"token": state["token"], "approved": True,
                                  "request_id": new_request_id("teams")}, 25, "agent")
    except Exception:
        # The POST may have succeeded before the UI connection failed.
        return attempted, _button(label="Check Teams"), "Delivery could not be confirmed. Check the channel before sharing another answer."
    if status == 200 and body.get("status") == "accepted":
        return attempted, _button(label="Submitted to Teams"), "Submitted to Teams. Check the channel for delivery."
    if status == 200 and body.get("status") == "rejected":
        return attempted, _button(), "Teams rejected the message. Check the webhook/workflow configuration before sharing a new answer."
    if status == 503 and body.get("error", {}).get("code") == "teams_not_configured":
        return state, _button(True), "Set web_hook in .env and restart the Agent, then click again."
    return attempted, _button(label="Check Teams"), "Delivery could not be confirmed. Check the Teams channel; this answer will not be sent again automatically."
