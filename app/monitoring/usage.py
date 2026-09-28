"""Read provider token usage and estimate costs using the configured price table.

Unknown usage/prices remain unknown; known subtotals are reported separately.
"""

from decimal import Decimal
import json
import math
from pathlib import Path

from app.settings import configured_path

# Read SDK objects and dictionaries without substituting estimated token counts.
def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def provider_usage(response, operation):
    """Unknown usage stays unknown; token estimates are never substituted."""
    usage = field(response, "usage")
    if usage is None:
        return None
    input_tokens = field(usage, "prompt_tokens")
    total = field(usage, "total_tokens")
    output = 0 if operation == "embedding" else field(usage, "completion_tokens")
    if input_tokens is None and operation == "embedding":
        input_tokens = total
    details = field(usage, "prompt_tokens_details")
    cached = field(details, "cached_tokens", 0) or 0
    if any(type(n) is not int or n < 0 for n in (input_tokens, output, total, cached)):
        return None
    if cached > input_tokens or total != input_tokens + output:
        return None
    return {"input_tokens": input_tokens, "output_tokens": output,
            "total_tokens": total, "cached_input_tokens": cached,
            "source": "provider"}


# Apply versioned rates, including cached-input discounts.
class Pricing:
    def __init__(self, path=None):
        path = path or configured_path("RAG_PRICING_FILE", "config/pricing.json")
        self.config = json.loads(Path(path).read_text(encoding="utf-8"))
        if self.config["currency"] != "USD":
            raise ValueError("Pricing currency must be USD")
        for rates in self.config["models"].values():
            for name in ("input_per_million", "output_per_million"):
                if name not in rates:
                    raise ValueError(f"Missing price: {name}")
            for rate in rates.values():
                if isinstance(rate, bool) or not math.isfinite(float(rate)) or float(rate) < 0:
                    raise ValueError("Prices must be finite, nonnegative numbers")

    def estimate(self, model, usage):
        key = self.config.get("aliases", {}).get(model, model)
        rates = self.config["models"].get(key)
        if rates is None or usage is None:
            return None
        cached = usage["cached_input_tokens"]
        # If a model reports cache usage without a cache price, don't invent a price.
        if cached and "cached_input_per_million" not in rates:
            return None
        divisor = Decimal(1_000_000)
        ordinary = usage["input_tokens"] - cached
        input_cost = (ordinary * Decimal(str(rates["input_per_million"]))
                      + cached * Decimal(str(rates.get("cached_input_per_million", 0)))) / divisor
        output_cost = usage["output_tokens"] * Decimal(str(rates["output_per_million"])) / divisor
        return {"input_usd": float(input_cost), "output_usd": float(output_cost),
                "total_usd": float(input_cost + output_cost), "rates": rates,
                "pricing_model": key, "pricing_version": self.config["version"]}


# Aggregate calls without double-counting operation breakdowns.
def usage_summary(calls):
    def group(selected):
        complete = all(c["usage"] is not None for c in selected)
        known = {key: sum(c["usage"][key] for c in selected if c["usage"] is not None)
                 for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens")}
        return {**{k: v if complete else None for k, v in known.items()},
                "known_tokens": known, "complete": complete, "calls": len(selected)}

    additional = sorted({c["operation"] for c in calls} - {"main_llm", "embedding"})
    return {**group(calls),
            "main_llm": group([c for c in calls if c["operation"] == "main_llm"]),
            "embedding": group([c for c in calls if c["operation"] == "embedding"]),
            "additional_llm": group([c for c in calls if c["operation"] in additional]),
            "by_operation": {op: group([c for c in calls if c["operation"] == op])
                             for op in sorted({c["operation"] for c in calls})}}


def cost_summary(calls):
    def subtotal(selected, part="total_usd"):
        if any(c["cost"] is None for c in selected):
            return None
        return float(sum((Decimal(str(c["cost"][part])) for c in selected), Decimal(0)))

    main = [c for c in calls if c["operation"] == "main_llm"]
    extras = [c for c in calls if c["operation"] not in {"main_llm", "embedding"}]
    return {"currency": "USD", "estimated": True,
            "llm_input_usd": subtotal(main, "input_usd"),
            "llm_output_usd": subtotal(main, "output_usd"),
            "embedding_usd": subtotal([c for c in calls if c["operation"] == "embedding"]),
            "evaluation_usd": subtotal([c for c in calls if c["operation"] == "evaluation"]),
            "reranking_usd": subtotal([c for c in calls if c["operation"] == "reranking"]),
            "additional_llm_usd": subtotal(extras),
            "by_operation_usd": {op: subtotal([c for c in calls if c["operation"] == op])
                                 for op in sorted({c["operation"] for c in calls})},
            "total_usd": subtotal(calls),
            "known_total_usd": subtotal([c for c in calls if c["cost"] is not None]),
            "complete": all(c["cost"] is not None for c in calls)}
