"""Token usage and cost accounting."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from copilot.agent.graph import merge_usage
from copilot.agent.llm import cost_usd, price_for, usage_from_message
from copilot.config import Settings

S = Settings(_env_file=None)


def test_usage_has_per_model_keys() -> None:
    msg = AIMessage("x", usage_metadata={"input_tokens": 1000, "output_tokens": 100,
                                         "total_tokens": 1100,
                                         "input_token_details": {"cache_read": 400}},
                    response_metadata={"model_name": "gpt-4o-mini-2024-07-18"})
    u = usage_from_message(msg)
    assert u["input"] == 1000 and u["cached_input"] == 400 and u["llm_calls"] == 1
    assert u["m|gpt-4o-mini-2024-07-18|output"] == 100


def test_cost_mixes_models_with_cached_prices() -> None:
    usage = merge_usage(
        {"input": 1_000_000, "cached_input": 0, "output": 0, "m|gpt-4o-mini|input": 1_000_000,
         "m|gpt-4o-mini|cached_input": 0, "m|gpt-4o-mini|output": 0},
        {"input": 1_000_000, "cached_input": 500_000, "output": 1_000_000,
         "m|gpt-4.1-2025-04-14|input": 1_000_000, "m|gpt-4.1-2025-04-14|cached_input": 500_000,
         "m|gpt-4.1-2025-04-14|output": 1_000_000},
    )
    # mini: 1M fresh input * 0.15; 4.1: 0.5M fresh * 2 + 0.5M cached * 0.5 + 1M out * 8
    assert cost_usd(usage, S) == pytest.approx(0.15 + 1.0 + 0.25 + 8.0)


def test_unknown_model_uses_configured_prices() -> None:
    assert price_for("mock", S) == (S.llm_price_input_per_m, S.llm_price_cached_input_per_m,
                                    S.llm_price_output_per_m)
    assert price_for("gpt-4.1-mini-2025-04-14", S) == (0.40, 0.10, 1.60)
