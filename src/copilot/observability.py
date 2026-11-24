"""OpenTelemetry tracing to Arize Phoenix (LangChain/LangGraph auto-instrumentation)."""

from __future__ import annotations

import logging

from copilot.config import Settings

log = logging.getLogger(__name__)
_done = False


def exporter_headers(settings: Settings) -> dict[str, str]:
    """Auth header for a Phoenix with login enabled (system API key as Bearer token)."""
    key = settings.phoenix_api_key.get_secret_value() if settings.phoenix_api_key else ""
    return {"authorization": f"Bearer {key}"} if key else {}


def setup_tracing(settings: Settings) -> bool:
    """Send LLM, tool and graph spans to Phoenix if PHOENIX_COLLECTOR_ENDPOINT is set."""
    global _done
    if _done or not settings.phoenix_collector_endpoint:
        return _done
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": settings.phoenix_project_name})
    )
    exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint,
                                headers=exporter_headers(settings))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    log.info("tracing to %s", settings.phoenix_collector_endpoint)
    _done = True
    return True
