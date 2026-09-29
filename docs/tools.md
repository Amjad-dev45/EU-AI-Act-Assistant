# Tool contribution

This contribution contains only the four tools and their function-calling schemas.
The existing RAG UI, answer pipeline, evaluation scripts and monitoring are unchanged.
Your teammates can connect these tools to their agent loop and approval UI.

Install from the repository root:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

## Integration

```python
from app.tools import CorpusTools, TOOL_SCHEMAS

# Reuse the team's existing Retriever. Keep one toolbox per question/session.
toolbox = CorpusTools(retriever, k=5, trace=request_trace)  # trace is optional

# Teammate passes TOOL_SCHEMAS as the model call's tools parameter.
# For each returned function call, send this JSON result back to the model:
result_json = toolbox.run_tool(call.function.name, call.function.arguments)
```

- `search_corpus(query)` reuses retrieval and returns passages with source, full
  page lists and stable citation IDs across that toolbox's searches. The union
  of passages is available in `toolbox.passages`. Without a supplied request trace,
  the existing retriever logs each search as a standalone retrieval check.
- `calculator(expression)` evaluates bounded numeric arithmetic through `simpleeval`.
  Code execution, function calls, powers, strings and bitwise operations are excluded.
- `list_documents()` returns sorted, distinct source names from the loaded chunks.
- `send_message(text)` returns an exact Slack message proposal. Include any source
  references in `text`; it is sent unchanged after approval.

`run_tool` returns recoverable JSON errors, without exposing provider exception text.
The caller owns the agent loop, model calls, step limits, trajectory logging and UI.

## Send-message approval and setup

The host must display the proposal's exact `text`, `channel` and `transport` to the
person. Only after their yes/no response, the host calls:

```python
delivery = toolbox.approve_message(proposal_id, approved=human_confirmed)
```

Never include `approve_message` in the model's tool list or derive `human_confirmed`
from model output. Proposals belong to the toolbox instance; do not share an instance
across users. A rejection consumes the proposal without sending. An approval consumes
it before attempting delivery to prevent duplicate sends. On `delivery_unknown`,
check Slack before preparing a new proposal. Pending proposals are in memory only.

The default transport uses the MCP client SDK with Slack's official MCP server.
Create an internal Slack app enabled for MCP, obtain a user OAuth token with
`chat:write`, and set `SLACK_TRANSPORT=mcp`, `SLACK_CHANNEL_ID` and `SLACK_MCP_TOKEN`
in `.env`. Load `.env` in the host before creating the toolbox. OAuth setup and token
renewal are external to this tool contribution. See
[Slack's official setup guide](https://docs.slack.dev/ai/slack-mcp-server/developing/).

The course-approved bot fallback uses `SLACK_TRANSPORT=bot`, `SLACK_CHANNEL_ID` and
`SLACK_BOT_TOKEN`. Give the bot `chat:write` and invite it to the target channel.
Credentials stay in `.env`; `.env.example` contains only empty placeholders.
Live sending requires credentials and has not been tested against Slack.

Offline tool tests:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_tools.py -v
```
