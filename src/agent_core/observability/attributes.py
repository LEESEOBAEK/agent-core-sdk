"""
agent_core/observability/attributes.py  —  OpenInference semantic convention constants
=======================================================================================
Import these constants instead of hard-coding attribute key strings.

Reference: https://github.com/Arize-ai/openinference/tree/main/spec
"""

# Span kind key
SPAN_KIND = "openinference.span.kind"


class SpanKind:
    """Literal values for openinference.span.kind."""
    LLM       = "LLM"
    CHAIN     = "CHAIN"
    TOOL      = "TOOL"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    AGENT     = "AGENT"


# LLM attributes
LLM_MODEL    = "llm.model_name"
LLM_PROVIDER = "llm.provider"
LLM_SYSTEM   = "llm.system"

# Token counts (shown in Phoenix dashboard Token tab)
LLM_TOKEN_COUNT_PROMPT     = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
LLM_TOKEN_COUNT_TOTAL      = "llm.token_count.total"

# Input / output
INPUT_VALUE  = "input.value"
INPUT_MIME   = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME  = "output.mime_type"

# Tool attributes
TOOL_NAME        = "tool.name"
TOOL_DESCRIPTION = "tool.description"
TOOL_PARAMETERS  = "tool.parameters"

# Retrieval attributes
# Value is a JSON-encoded list; each element is a dict with keys:
#   document.id, document.content, document.score, document.metadata
RETRIEVAL_DOCS = "retrieval.documents"

# MIME type constants
MIME_TEXT = "text/plain"
MIME_JSON = "application/json"
