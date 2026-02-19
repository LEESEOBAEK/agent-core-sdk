# Agent-Core-SDK 🚀

**Shared observability, LLM client, RAG infrastructure, and base tools for LLM agents.**

This SDK provides a professional, production-ready foundation for building AI agents with built-in tracing, multi-provider support, and modular RAG components.

---

## 🌟 Key Features

### 1. Unified LLM Client

- **Seamless Switching**: Switch between OpenAI, Gemini, Claude, and Ollama with a single interface.
- **Auto-Normalization**: Handles case-sensitivity and provider-specific model defaults.
- **Robustness**: Built-in error handling and logging.

### 2. Professional Observability (OpenInference)

- **Deep Tracing**: Every LLM call, RAG retrieval, and tool execution is recorded as a structured span.
- **Phoenix Ready**: Fully compatible with [Arize Phoenix](https://phoenix.arize.com/) for visualization.
- **Metric Rich**: Automatically captures token counts (prompt, completion, total) and latency.

### 3. Modular RAG Infrastructure

- **VectorStore Abstraction**: Clean interface for different backends. **ChromaDB** is built-in, and the architecture is ready for production-grade engines like **Vespa**.
- **Embedding Utilities**: Easy-to-use embedding functions for local and cloud models.

---

## 🛠️ Installation

### Basic (Tracing + LLM Client)

```bash
pip install -e agent-core-sdk
```

### Full (including RAG, Server, and all LLMs)

```bash
pip install -e agent-core-sdk[all]
```

### Advanced RAG (Vespa)

```bash
# 1. Install Vespa dependencies
pip install -e agent-core-sdk[vespa]

# 2. Start Vespa engine
cd docker/vespa
docker compose up -d

# 3. Use in code
from agent_core.rag.infrastructure import create_store
store = create_store("vespa")
```

---

## 🚀 Quick Start

```python
from agent_core.llm.client import LLMClient
from agent_core.observability.tracing import init_tracing

# 1. Initialize Tracing
init_tracing(project_name="my-first-agent")

# 2. Start Chatting
client = LLMClient(provider="ollama", model="qwen3:14b")
response = client.chat("What is modular architecture?")
print(response)
```

---

## 📁 Directory Structure

```
agent-core-sdk/
├── src/
│   └── agent_core/
│       ├── llm/            # Multi-provider client
│       ├── observability/  # OpenTelemetry & Phoenix tracing
│       └── rag/            # VectorStore & Embedding infra
├── pyproject.toml          # Package metadata & dependencies
└── README.md
```

---

## 📄 License

MIT License
