"""
agent_core/observability/tracing.py  —  Phoenix (OTLP) tracing initializer
===========================================================================
Call `init_tracing(service_name, project_name)` once at startup, then use
`get_tracer()` everywhere else.

Usage:
    from agent_core.observability.tracing import init_tracing, get_tracer

    init_tracing(project_name="my-project")
    tracer = get_tracer("my-module")
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def init_tracing(
    project_name: str = "default",
    phoenix_endpoint: str = "http://localhost:6006",
    auto_instrument_openai: bool = True,
) -> None:
    """
    Initialize Phoenix (OTLP HTTP) tracing.

    Parameters
    ----------
    project_name : str
        Project name shown in the Phoenix UI (default: "default")
    phoenix_endpoint : str
        Phoenix server address (default: "http://localhost:6006")
    auto_instrument_openai : bool
        Whether to auto-instrument OpenAI client (default: True)
    """
    import os
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    os.environ["PHOENIX_PROJECT_NAME"] = project_name

    resource = Resource.create({
        "service.name": project_name,
        "project.name": project_name,
        "openinference.project.name": project_name,
    })

    provider = TracerProvider(resource=resource)

    otlp_endpoint = f"{phoenix_endpoint}/v1/traces"
    headers = {"X-Phoenix-Project-Name": project_name}
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, headers=headers)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)

    logger.info(
        f"[Tracing] Phoenix tracing initialized "
        f"(project='{project_name}', endpoint='{otlp_endpoint}')"
    )

    if auto_instrument_openai:
        _try_instrument_openai(provider)


def _try_instrument_openai(tracer_provider) -> None:
    """Attempt to auto-instrument the OpenAI client."""
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("[Tracing] OpenAI auto-instrumentation enabled")
    except ImportError:
        logger.warning(
            "[Tracing] openinference-instrumentation-openai not found; "
            "skipping OpenAI auto-instrumentation. "
            "(pip install openinference-instrumentation-openai)"
        )


def get_tracer(name: str = "agent-core"):
    """
    Return a tracer from the currently configured TracerProvider.
    Call init_tracing() first.

    Usage:
        tracer = get_tracer("my-module")
        with tracer.start_as_current_span("my-op") as span:
            span.set_attribute("key", "value")
    """
    from opentelemetry import trace
    return trace.get_tracer(name)
