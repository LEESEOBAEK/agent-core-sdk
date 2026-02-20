# Agent-Core-SDK v0.2.0

**Shared observability, LLM client, state management, and RAG infrastructure for production AI agents.**

This SDK provides a professional, rock-solid foundation for building AI agents with built-in Phoenix tracing, multi-provider LLM support, graceful fallback resilience, and modular RAG components.

---

## 🌟 Key Features

### 1. Unified LLM Client with Graceful Fallback

- **Seamless Switching**: Switch between OpenAI, Gemini, Claude, and Ollama with a single interface and zero application-level changes.
- **Unified Hyperparameter API**: Pass `temperature`, `max_tokens`, and `top_p` once at construction time — the client normalizes and forwards them correctly to every provider's native API (including `GenerationConfig` for Gemini).
- **Auto-Normalization**: Handles case-sensitivity and provider-specific model defaults automatically.
- **Graceful Fallback**: If the `google-generativeai` SDK is missing or the environment is misconfigured, the client transparently falls back to `ollama/qwen3:14b` with a clean `model=None` reset, preventing crashes. A descriptive warning is emitted so the operator always knows what happened.
- **Full Observability**: Every `chat()` call is recorded as a Phoenix `llm-call` span capturing the *exact* model used (`llm.model`), all three sampling hyperparameters, token counts, and the full prompt/response payload.
- **Robust Error Handling**: Exceptions are captured via `span.record_exception()` and `StatusCode.ERROR`, making failures visible in the Phoenix trace UI rather than silently swallowed.

### 2. Robust LangGraph State Reducers

- **`prune_messages` Reducer**: A production-grade drop-in replacement for LangGraph's built-in `add_messages` reducer. It enforces a hard cap of 100 messages while guaranteeing survival of the two most critical anchors in any long-running agent:
  - **All `SystemMessage` objects** — the agent must never lose its persona or instruction set.
  - **The very first `HumanMessage`** — the original task query is the root context for every downstream node and is always preserved.
- **Context-Window Amnesia Prevention**: Oldest non-essential messages are evicted first, keeping the working context lean without sacrificing agent identity or task coherence.
- **Fully Compatible**: Works as a standard LangGraph `Annotated` reducer — no changes to your `StateGraph` wiring required.

### 3. Professional Observability (OpenInference / Phoenix)

- **Deep Tracing**: Every LLM call, RAG retrieval, and tool execution is recorded as a structured span.
- **Phoenix Ready**: Fully compatible with [Arize Phoenix](https://phoenix.arize.com/) for visualization.
- **Metric Rich**: Automatically captures token counts (prompt, completion, total), latency, provider, model, and sampling hyperparameters.

### 4. Modular RAG Infrastructure

- **VectorStore Abstraction**: Clean interface for different backends. **ChromaDB** is built-in, and the architecture is ready for production-grade engines like **Vespa**.
- **Embedding Utilities**: Easy-to-use embedding functions for local and cloud models.

---

## 🛠️ Installation

### Basic (Tracing + LLM Client)

```bash
pip install -e agent-core-sdk
```

### With LangGraph State Utilities

```bash
pip install -e "agent-core-sdk[langgraph]"
# or manually:
pip install langgraph langchain-core
```

### Full (including RAG, Server, and all LLMs)

```bash
pip install -e "agent-core-sdk[all]"
```

### Advanced RAG (Vespa)

```bash
# 1. Install Vespa dependencies
pip install -e "agent-core-sdk[vespa]"

# 2. Start Vespa engine
cd docker/vespa
docker compose up -d

# 3. Use in code
from agent_core.rag.infrastructure import create_store
store = create_store("vespa")
```

---

## 🚀 Quick Start

### Basic LLM Call with Hyperparameter Control

```python
from agent_core.llm.client import LLMClient
from agent_core.observability.tracing import init_tracing

# 1. Initialize Phoenix tracing
init_tracing(project_name="my-first-agent")

# 2. Instantiate the client — hyperparameters are set once and applied universally
client = LLMClient(
    provider="ollama",
    model="qwen3:14b",
    temperature=0.3,    # more deterministic outputs
    max_tokens=1024,
    top_p=0.95,
)

response = client.chat("What is modular architecture?")
print(response)
```

### Automatic Fallback (Gemini → Ollama)

```python
# If google-generativeai is not installed, this silently falls back to
# ollama/qwen3:14b rather than crashing. Check your logs for the warning.
client = LLMClient(provider="gemini", temperature=0.5)
response = client.chat("Summarize the concept of graceful degradation.")
print(response)
```

### Robust State Management with `prune_messages`

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from agent_core.state.reducers import prune_messages

# Replace add_messages with prune_messages in your state definition.
# No other changes to your StateGraph are needed.
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], prune_messages]

# LangGraph will automatically invoke prune_messages on every state update.
# SystemMessages and the first HumanMessage are always preserved,
# even after hundreds of turns.
```

---

## 📁 Directory Structure

```
agent-core-sdk/
├── src/
│   └── agent_core/
│       ├── llm/            # Unified multi-provider client with fallback
│       ├── observability/  # OpenTelemetry & Phoenix tracing
│       ├── state/          # LangGraph reducers (prune_messages)
│       ├── rag/            # VectorStore & Embedding infrastructure
│       └── tools/          # Shared tool implementations
├── pyproject.toml          # Package metadata & dependencies
└── README.md
```

---

## 🔧 Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | Active provider: `ollama` / `openai` / `gemini` / `claude` |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Local Ollama endpoint |
| `PHOENIX_ENDPOINT` | `http://localhost:6006` | Phoenix OTLP collector |
| `OPENAI_API_KEY` | — | OpenAI credentials |
| `GEMINI_API_KEY` | — | Google Gemini credentials |
| `ANTHROPIC_API_KEY` | — | Anthropic credentials |

---

## 📄 License

MIT License
