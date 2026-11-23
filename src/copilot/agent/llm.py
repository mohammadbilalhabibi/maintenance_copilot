"""Chat model factory (any OpenAI-compatible endpoint), token usage and cost helpers."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from copilot.config import Settings

# USD per 1M tokens (input, cached input, output), OpenAI standard tier.
# Source: https://developers.openai.com/api/docs/pricing (checked 2026-11-23).
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
}


def make_llm(settings: Settings, model: str | None = None) -> BaseChatModel:
    """Create the chat model from settings; `model` overrides LLM_MODEL."""
    kwargs: dict[str, Any] = {
        "model": model or settings.llm_model,
        "base_url": settings.llm_base_url,
        "api_key": settings.llm_api_key.get_secret_value() or "missing",
        "timeout": settings.llm_timeout,
        "max_retries": settings.llm_max_retries,
        "max_completion_tokens": settings.llm_max_output_tokens,
        "stream_usage": True,
    }
    if settings.llm_temperature is not None:
        kwargs["temperature"] = settings.llm_temperature
    return ChatOpenAI(**kwargs)


def usage_from_message(msg: Any) -> dict[str, int]:
    """Token usage of one AIMessage: totals plus per-model keys 'm|<model>|<kind>'."""
    meta = getattr(msg, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    usage = {
        "input": int(meta.get("input_tokens", 0)),
        "cached_input": int(details.get("cache_read", 0) or 0),
        "output": int(meta.get("output_tokens", 0)),
        "llm_calls": 1,
    }
    model = (getattr(msg, "response_metadata", None) or {}).get("model_name")
    if model:
        for kind in ("input", "cached_input", "output"):
            usage[f"m|{model}|{kind}"] = usage[kind]
    return usage


def price_for(model: str, settings: Settings) -> tuple[float, float, float]:
    """Prices for a model name (dated names like gpt-4o-mini-2024-07-18 match by prefix)."""
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            return PRICES[name]
    return (settings.llm_price_input_per_m, settings.llm_price_cached_input_per_m,
            settings.llm_price_output_per_m)


def cost_usd(usage: dict[str, int], settings: Settings) -> float:
    """Cost of a usage dict, per model where known, else at the configured prices."""
    per_model: dict[str, dict[str, int]] = {}
    for key, value in usage.items():
        if key.startswith("m|"):
            _, model, kind = key.split("|")
            per_model.setdefault(model, {})[kind] = value
    if not per_model:
        per_model = {settings.llm_model: usage}
    total = 0.0
    for model, u in per_model.items():
        p_in, p_cached, p_out = price_for(model, settings)
        cached = u.get("cached_input", 0)
        total += ((u.get("input", 0) - cached) * p_in + cached * p_cached
                  + u.get("output", 0) * p_out)
    return total / 1e6
