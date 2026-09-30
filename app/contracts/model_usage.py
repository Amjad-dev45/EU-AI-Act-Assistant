"""Per-request context usage, separate from cumulative billing counters."""
# https://developers.openai.com/api/docs/models/gpt-4o-mini
CONTEXT_WINDOWS = {"gpt-4o-mini": 128_000}


def context_snapshot(usage, model="gpt-4o-mini"):
    """Unknown token counts stay unknown instead of looking like an empty context."""
    def count(key):
        value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
        return value if type(value) is int and value >= 0 else None
    prompt, completion = count("prompt_tokens"), count("completion_tokens")
    return {"model": model, "limit_tokens": CONTEXT_WINDOWS.get(model),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "used_tokens": prompt + completion if prompt is not None and completion is not None else None}
