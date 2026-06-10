# Memory Agent

Memory Agent is a small long-term memory chatbot built with LangGraph,
Chainlit, SQLite, and a local embedding model.

The app keeps user memories in a SQLite database, retrieves relevant memories
with semantic search before each response, and extracts durable memory updates
after each turn.

## Features

- LangGraph conversation flow with a per-thread short-term checkpoint.
- SQLite-backed long-term memory store.
- Local `sentence-transformers` embedding model for semantic memory search.
- OpenAI-compatible chat model configuration.
- Chainlit UI with streaming responses.
- Memory viewing, searching, deletion confirmation, and current-context display.
- CLI demo that exercises memory creation, update, and retrieval.

## Project Layout

```text
.
├── chainlit_app.py              # Chainlit web UI entrypoint
├── main.py                      # CLI demo entrypoint
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI/session helpers
│   ├── config.py                # Environment-backed settings
│   ├── graph.py                 # LangGraph nodes and graph builder
│   ├── llm.py                   # OpenAI-compatible chat model loader
│   ├── memory_extractor.py      # Durable memory extraction logic
│   ├── persistent_store.py      # SQLite + embedding memory store
│   ├── prompts.py               # Prompt templates
│   └── state.py                 # LangGraph state schema
├── pyproject.toml               # Python dependency metadata
└── .env.example                 # Safe environment template
```

## Requirements

- Python 3.12 or newer.
- An OpenAI-compatible chat completion endpoint.
- Enough disk space for the local embedding model.

The project was developed against the dependency versions pinned in
`pyproject.toml`.

## Setup

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
pip install -e .
```

Create your local environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

Download the embedding model into the default local path:

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.save("./models/bge-m3")
PY
```

The default `.env.example` points `EMBEDDING_MODEL` to `./models/bge-m3`.

## Configuration

The application reads configuration from `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `mimo-v2.5-pro` | Chat model name sent to the OpenAI-compatible endpoint. |
| `LLM_API_KEY` | none | API key for the LLM endpoint. |
| `LLM_BASE_URL` | none | OpenAI-compatible base URL, usually ending with `/v1`. |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature. |
| `LLM_MAX_TOKENS` | `2048` | Maximum output tokens. |
| `LLM_TIMEOUT` | `30` | LLM request timeout in seconds. |
| `LLM_TRUST_ENV` | `false` | Whether HTTP clients should use proxy variables from the environment. |
| `LLM_STREAMING` | `true` | Whether normal chat responses should stream. |
| `MEMORY_DB_PATH` | `./memory.db` | SQLite database path for long-term memories. |
| `EMBEDDING_MODEL` | `./models/bge-m3` | Local or Hugging Face embedding model path. |
| `DEFAULT_USER_ID` | `user_001` | Default user namespace for memories. |
| `APP_DEBUG` | `false` | Enables extra backend debug output. |

If your shell exports `DEBUG=release`, Chainlit may parse it as its own boolean
debug flag. Start Chainlit with `env -u DEBUG` as shown below.

## Run The CLI Demo

```bash
source .venv/bin/activate
python main.py
```

The demo runs three turns:

1. Store a Python Agent coding preference.
2. Update it to Rust-first.
3. Start a new thread and retrieve the remembered preference.

The demo writes to the configured `MEMORY_DB_PATH`.

## Run The Chainlit App

```bash
source .venv/bin/activate
env -u DEBUG chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The UI provides actions for:

- Viewing all long-term memories.
- Searching memories by query.
- Inspecting the current user, thread, model, and recent memory hits.
- Starting a new short-term thread while keeping long-term memory.
- Deleting individual memories after confirmation.

## Data And Ignored Files

The repository ignores local runtime artifacts:

- `.env`
- `*.db`
- `models/`
- `.chainlit/`
- `chainlit.md`
- Python cache directories

Do not commit API keys, local databases, or downloaded model weights.

## Notes

- Memory storage is implemented in `SQLiteVectorMemoryStore`.
- SQLite and embedding work are executed behind async-friendly background
  threads so Chainlit is less likely to block on local CPU or disk work.
- Memory extraction failures are logged and skipped instead of breaking the
  user-facing chat turn.
- The LLM client defaults to `LLM_TRUST_ENV=false` to avoid broken system proxy
  settings. Set it to `true` only if your endpoint requires environment proxy
  variables.
