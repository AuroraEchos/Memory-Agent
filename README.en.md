# Memory Agent

[中文](./README.md) | [English](./README.en.md)

Memory Agent is a long-term memory chatbot built with LangGraph, Chainlit,
Qdrant, and service-ready embedding.

The app classifies user memories with a long-term memory taxonomy, stores them
in Qdrant, retrieves relevant memories with semantic search before each
response, and extracts durable memory updates after each turn.

## Features

- LangGraph conversation flow with a per-thread short-term checkpoint.
- Short-term thread checkpoints persist to PostgreSQL and share the same
  database with Chainlit UI history.
- Long-term memories are stored in a Qdrant server with vector search and
  remote-service access.
- Long-term memories are classified as `persona`, `entity`, `project`, `task`,
  `episodic`, `procedural`, or `knowledge`, and written to typed namespaces.
- Embeddings are generated only through the standalone embedding service, so
  the main app process does not load the embedding model.
- OpenAI-compatible chat model configuration.
- Chainlit UI with streaming and non-streaming responses.
- Chainlit login, chat history list, chat resume, and a streamlined chat UI.
- Long-term memory retrieval, extraction, and updates run automatically in the
  backend without manual UI triggers.

## Project Layout

```text
.
├── chainlit_app.py              # Chainlit web UI entrypoint
├── docker-compose.yml           # Docker Compose runtime for all app services
├── Dockerfile                   # Application image
├── embedding_server.py          # FastAPI embedding service entrypoint
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI/session helpers
│   ├── config.py                # Environment-backed settings
│   ├── context_builder.py       # Short-term conversation context builder
│   ├── graph.py                 # LangGraph nodes and graph builder
│   ├── llm.py                   # OpenAI-compatible chat model loader
│   ├── long_term_memory/        # Isolated long-term memory subsystem
│   │   ├── consolidator.py      # Extraction, reconciliation, and persistence
│   │   ├── retrieval.py         # Cross-namespace retrieval and prompt formatting
│   │   ├── taxonomy.py          # Memory types, namespaces, and retrieval budgets
│   │   ├── prompts.py           # Long-term memory extraction prompt
│   │   ├── embedding/
│   │   │   ├── base.py          # Embedding provider protocol
│   │   │   ├── factory.py       # Embedding provider factory
│   │   │   └── remote.py        # HTTP embedding service provider
│   │   └── store/
│   │       ├── base.py          # Memory store protocol
│   │       ├── factory.py       # Qdrant store factory
│   │       └── qdrant.py        # Qdrant vector memory store
│   ├── prompts.py               # Chat system prompt
│   └── state.py                 # LangGraph state schema
├── scripts/download_embedding_model.py # One-shot Docker model downloader
├── scripts/sql/init_chainlit_schema.sql # Chainlit UI history schema
├── pyproject.toml               # Python dependency metadata
└── .env.example                 # Safe environment template
```

`memory_agent/long_term_memory/` encapsulates taxonomy, retrieval, memory
consolidation, the embedding client, and Qdrant persistence. `graph.py` now
orchestrates these public capabilities, while short-term conversation context
remains independently managed by the root-level `context_builder.py`.

## Storage Domains

The project has three logical storage domains:

| Domain | Implementation | Stored Data | Primary Index |
| --- | --- | --- | --- |
| UI session storage | Chainlit SQLAlchemy DataLayer, backed by PostgreSQL | Users, thread list, chat message steps, elements, feedback | Chainlit authenticated user and Chainlit thread id |
| Graph state storage | LangGraph `AsyncPostgresSaver`, backed by PostgreSQL | Per-thread checkpoints, short-term conversation state, graph runtime state | `configurable.thread_id` |
| Long-term memory storage | `QdrantMemoryStore`, backed by Qdrant server | Cross-thread durable memories, taxonomy metadata, vector index | `user_id` and `memory_type` namespace |

UI session storage and Graph state storage currently share the same PostgreSQL database from `CHAINLIT_DATABASE_URL`, but they have different responsibilities. UI session storage powers the history list and message replay. Graph state storage restores the LangGraph thread execution context. Long-term memory storage lives separately in Qdrant and isolates users plus memory types with the `("memories", user_id, memory_type)` namespace.

Identity and linking rules:

- `CHAINLIT_AUTH_USER_ID` is the only user identity source, and it is used in the Chainlit session, LangGraph `Context.user_id`, and long-term memory namespaces.
- The Chainlit thread id is passed to LangGraph as `configurable.thread_id`, linking UI sessions to Graph checkpoints.
- `source.thread_id` in long-term memory payloads records the source thread for traceability, but it is not the primary long-term memory index.
- Deleting a UI session does not automatically delete Graph checkpoints or long-term memories; deleting a long-term memory does not rewrite historical chat messages.

## Requirements

- Docker Engine with the Docker Compose plugin.
- An OpenAI-compatible chat completion endpoint.
- Enough disk space for the embedding model.

The default workflow is Docker-only: `memory-agent`, `embedding-service`,
`qdrant`, and `chainlit-postgres` are started and managed through
`docker compose`. Running `chainlit_app.py` or `embedding_server.py` directly
on the host is no longer the recommended setup.

The images are built and validated against the dependency versions pinned in
`pyproject.toml`.

## Quick Start With Docker

Docker Compose starts Memory Agent, the embedding service, Qdrant, and
PostgreSQL, and is the recommended quick start path.

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
CHAINLIT_AUTH_SECRET=replace-with-a-random-secret
CHAINLIT_AUTH_USERNAME=your-login-name
CHAINLIT_AUTH_PASSWORD=your-login-password
CHAINLIT_AUTH_USER_ID=wenhao
```

On Linux, keep `APP_UID=1000` and `APP_GID=1000` for the common default user,
or change them to the output of `id -u` and `id -g` before building the image.
The container runs as this non-root user.

Model downloading uses a Hugging Face mirror by default. To override it, set
this in `.env`:

```env
HF_ENDPOINT=https://hf-mirror.com
```

Start the app:

```bash
docker compose up --build
```

On first startup, Compose runs the one-shot `model-downloader` service before
the embedding service. It downloads `EMBEDDING_MODEL_SOURCE` into the Docker
named volume `embedding_models`, using
`HF_ENDPOINT=https://hf-mirror.com` by default. Later startups reuse that
cached model unless the requested source changes.

Open:

```text
http://127.0.0.1:8000
```

Docker setup notes:

- The default Compose file starts dedicated `model-downloader`,
  `embedding-service`, `qdrant`, and `chainlit-postgres` services.
- `docker-compose.yml` mounts `scripts/sql/init_chainlit_schema.sql` into the
  Postgres init directory. The script creates the Chainlit UI history tables
  and runs automatically only when a new `chainlit_postgres_data` volume is
  initialized for the first time. LangGraph checkpoint tables are created or
  migrated by the app on startup.
- `memory-agent` requests vectors through
  `EMBEDDING_SERVICE_URL=http://embedding-service:8001` and does not load the
  embedding model in the main process.
- `model-downloader` fetches `EMBEDDING_MODEL_SOURCE` into the Docker named
  volume `embedding_models`, and `embedding-service` mounts that volume
  read-only and loads the model from `/app/models/current`.
- Embedding model data is persisted in the Docker named volume
  `embedding_models`.
- Qdrant data is persisted in Docker named volume `qdrant_storage`.
- Chainlit chat history, the chat list, and LangGraph short-term checkpoints
  are persisted in Docker named volume `chainlit_postgres_data`.
- The app connects to Qdrant through `QDRANT_URL=http://qdrant:6333`.
- The app connects to PostgreSQL through
  `CHAINLIT_DATABASE_URL=postgresql+asyncpg://memory_agent:memory_agent@chainlit-postgres:5432/chainlit`.
- `embedding-service` waits for `model-downloader` to complete successfully
  before startup.
- `memory-agent` waits for the embedding service and PostgreSQL health checks
  before startup. Qdrant readiness is handled by application-level retries,
  which avoids transient `Connection refused` errors while Qdrant is still
  booting.
- The default Compose file pins the embedding service to
  `EMBEDDING_DEVICE=cpu`, so container inference always runs on CPU without
  accelerator detection.
- `.env` is meant for user-facing app settings; internal container service
  addresses usually do not need to be written there.
- `.env` is read at runtime and is never copied into the image.

If you want to use a different embedding model, update
`EMBEDDING_MODEL_SOURCE` in `.env` and also update `EMBEDDING_DIMENSION`.
Vectors from different models must not be mixed, even when their dimensions
match. Use a new `QDRANT_COLLECTION` when switching models, or migrate
existing memories and regenerate their vectors. To force-clear the old model
cache, run `docker compose down -v`.

To stop the app:

```bash
docker compose down
```

To reset Docker embedding, Qdrant, and PostgreSQL data:

```bash
docker compose down -v
```

If you already have an older `chainlit_postgres_data` volume, the new Chainlit
init script will not run retroactively. Apply the SQL manually or reset the
volume with the command above. LangGraph checkpoint tables are created or
migrated by the app on startup.

## Configuration

The application reads user-facing settings from `.env`. Docker Compose and the
app defaults wire the internal Qdrant, embedding-service, and PostgreSQL
addresses automatically, so you usually do not need to write those URLs into
`.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_MODEL` | `mimo-v2.5-pro` | Chat model name sent to the OpenAI-compatible endpoint. |
| `LLM_API_KEY` | none | API key for the LLM endpoint. |
| `LLM_BASE_URL` | none | OpenAI-compatible base URL, usually ending with `/v1`. |
| `LLM_TEMPERATURE` | `0.7` | Sampling temperature. |
| `LLM_MAX_COMPLETION_TOKENS` | `2048` | Maximum completion tokens passed as `max_completion_tokens` to the OpenAI-compatible API. |
| `LLM_MAX_TOKENS` | none | Legacy compatibility variable; used only when `LLM_MAX_COMPLETION_TOKENS` is not set. |
| `LLM_TIMEOUT` | `30` | LLM request timeout in seconds. |
| `LLM_TRUST_ENV` | `false` | Whether HTTP clients should use proxy variables from the environment. |
| `LLM_STREAMING` | `true` | Whether normal chat responses should stream. When disabled, Chainlit uses the model's final output. |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Default Hugging Face mirror URL used by `model-downloader`. |
| `QDRANT_COLLECTION` | `agent_memories` | Qdrant collection name for long-term memories. |
| `EMBEDDING_MODEL_SOURCE` | `BAAI/bge-m3` | Hugging Face model id downloaded by `model-downloader` during first startup. |
| `EMBEDDING_DIMENSION` | `1024` | Current embedding model output dimension. It must match the Qdrant collection dimension, and existing collection dimensions are checked at startup. |
| `EMBEDDING_TIMEOUT` | `30` | Remote embedding request timeout in seconds. |
| `EMBEDDING_TRUST_ENV` | `false` | Whether the remote embedding HTTP client should use proxy variables from the environment. |
| `CHAINLIT_AUTH_SECRET` | none | Secret used by Chainlit login/session cookies. Required when chat history is enabled. |
| `CHAINLIT_AUTH_USERNAME` | none | Chainlit password login username. |
| `CHAINLIT_AUTH_PASSWORD` | none | Chainlit password login password. |
| `CHAINLIT_AUTH_USER_ID` | none | Required. Chainlit authenticated user identifier. Long-term memory, chat history, and LangGraph context all use this identity. |
| `LANGGRAPH_STRICT_MSGPACK` | `true` | LangGraph checkpoint serialization safety switch. Keep it enabled unless you have a compatibility reason to change it. |
| `CONVERSATION_MESSAGE_WINDOW` | `20` | Maximum message budget for short-term conversation context. The context builder prioritizes the current user message and recent complete turns; long-term memories are still retrieved from Qdrant and injected separately. |
| `APP_UID` | `1000` | Docker image user id used by Compose build args. |
| `APP_GID` | `1000` | Docker image group id used by Compose build args. |
| `APP_DEBUG` | `false` | Enables extra backend debug output. |

If you intentionally need a different topology, you can still add advanced
variables manually, such as `QDRANT_URL`, `QDRANT_API_KEY`,
`QDRANT_PREFER_GRPC`, `EMBEDDING_SERVICE_URL`, `CHAINLIT_DATABASE_URL`,
`EMBEDDING_CONCURRENCY`, `EMBEDDING_BATCH_SIZE`, or `EMBEDDING_MODEL`.

The application has exactly one user identity source: the Chainlit
authenticated user. A typical setup can use `CHAINLIT_AUTH_USERNAME=admin` for
login and `CHAINLIT_AUTH_USER_ID=wenhao`; Qdrant namespaces are split by
memory type, such as `memories/wenhao/persona` and
`memories/wenhao/project`, and Chainlit history plus LangGraph
`Context.user_id` also use `wenhao`.

## UI Notes

The current UI provides:

- Viewing the Chainlit chat history list after login.
- Resuming a historical chat with the same LangGraph thread checkpoint.
- Displaying assistant responses in streaming or non-streaming mode.

The current version intentionally removes per-message copy, feedback, memory
inspection, memory search, current-status, and delete-confirmation buttons.
Long-term memory retrieval, extraction, and updates still happen automatically
in the backend.

## Data And Ignored Files

The repository ignores local runtime artifacts:

- `.env`
- `qdrant_data/` (legacy local Qdrant data directory)
- `models/` (legacy host-side model directory)
- `.chainlit/*` (while keeping `.chainlit/config.toml` versioned)
- `chainlit.md`
- Python cache directories
- Local ZIP archives

Do not commit API keys, legacy local Qdrant data, or manually downloaded model
weights.

## Notes

- Memory storage is implemented in `QdrantMemoryStore`.
- `QdrantMemoryStore` only handles vector database reads and writes; embeddings
  are supplied by the remote `EmbeddingProvider`.
- Short-term conversation context is built by `context_builder`; it filters
  empty messages, ignores orphan assistant messages, and prioritizes the
  current user message plus recent complete turns.
- Long-term memory capabilities live under `memory_agent.long_term_memory`;
  the fixed taxonomy is defined in `long_term_memory.taxonomy`, and each
  `memory_type` is written to its own Qdrant namespace and retrieved with a
  fixed per-type budget.
- Docker uses the standalone `embedding-service`, reducing model loading and
  inference pressure in the Chainlit main process.
- Chainlit chat history uses the SQLAlchemy data layer; LangGraph short-term
  context uses `AsyncPostgresSaver`. Both use the PostgreSQL database pointed
  to by `CHAINLIT_DATABASE_URL`.
- Memory consolidation failures are logged and skipped instead of breaking the
  user-facing chat turn.
- The LLM client defaults to `LLM_TRUST_ENV=false` to avoid broken system proxy
  settings. Set it to `true` only if your endpoint requires environment proxy
  variables.
