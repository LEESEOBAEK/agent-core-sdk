"""
agent_core/llm/client.py  —  Unified multi-provider LLM client
===============================================================
Supports OpenAI, Gemini, Claude, and Ollama through a single interface.
Every chat() call automatically creates an 'llm-call' span with full
input/output/token attributes for Phoenix observability.

Usage:
    from agent_core.llm.client import LLMClient

    client = LLMClient(provider="ollama")   # or "openai", "gemini", "claude"
    response = client.chat("Hello!")
    print(response)
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

Provider = Literal["openai", "gemini", "claude", "ollama"]


class LLMClient:
    """
    Unified LLM client supporting multiple providers.

    Parameters
    ----------
    provider : str
        LLM provider: "openai" | "gemini" | "claude" | "ollama"
    model : str, optional
        Model name. None selects the provider default.
    """

    DEFAULT_MODELS = {
        "openai": "gpt-5.2",
        "gemini": "gemini-3-flash-preview",
        "claude": "claude-sonnet-4-6",
        "ollama": "qwen3:14b",
    }

    def __init__(self, provider: Provider = "ollama", model: str = None):
        # Normalize provider name to lowercase
        self.provider = provider.lower() if provider else "ollama"
        self.model = model or self.DEFAULT_MODELS.get(self.provider)
        
        if not self.model:
            valid = ", ".join(self.DEFAULT_MODELS.keys())
            raise ValueError(f"Unknown provider '{provider}'. Valid options: {valid}")
            
        logger.info(f"[LLMClient] provider={self.provider}, model={self.model}")

    def chat(self, prompt: str, system: str = None) -> str:
        """
        Send a message to the LLM and return the response text.
        Every call is recorded as an 'llm-call' span with full prompt/response.
        """
        from agent_core.observability.tracing import get_tracer
        from agent_core.observability.attributes import (
            SPAN_KIND, SpanKind,
            LLM_MODEL, LLM_PROVIDER,
            INPUT_VALUE, OUTPUT_VALUE,
            LLM_TOKEN_COUNT_PROMPT, LLM_TOKEN_COUNT_COMPLETION, LLM_TOKEN_COUNT_TOTAL,
        )
        from opentelemetry.trace import StatusCode

        tracer = get_tracer("llm_client")

        with tracer.start_as_current_span("llm-call") as span:
            span.set_attribute(SPAN_KIND, SpanKind.LLM)
            span.set_attribute(LLM_PROVIDER, self.provider)
            span.set_attribute(LLM_MODEL, self.model)
            span.set_attribute(INPUT_VALUE, prompt)
            if system:
                span.set_attribute("llm.system_prompt", system)

            dispatch = {
                "openai": self._call_openai,
                "gemini": self._call_gemini,
                "claude": self._call_claude,
                "ollama": self._call_ollama,
            }

            try:
                response = dispatch[self.provider](prompt, system)
                span.set_attribute(OUTPUT_VALUE, response)

                prompt_tokens     = max(1, len(prompt) // 4)
                completion_tokens = max(1, len(response) // 4)
                span.set_attribute(LLM_TOKEN_COUNT_PROMPT,     prompt_tokens)
                span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
                span.set_attribute(LLM_TOKEN_COUNT_TOTAL,      prompt_tokens + completion_tokens)

                span.set_status(StatusCode.OK)
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(StatusCode.ERROR, str(e))
                logger.error(f"[{self.provider}] call failed: {e}")
                raise

    def _call_openai(self, prompt: str, system: str = None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY in .env")

        client = OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt: str, system: str = None) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY in .env")

        genai.configure(api_key=api_key)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(full_prompt)
        return response.text

    def _call_claude(self, prompt: str, system: str = None) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY in .env")

        client = anthropic.Anthropic(api_key=api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _call_ollama(self, prompt: str, system: str = None) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = OpenAI(base_url=ollama_url, api_key="ollama")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content
