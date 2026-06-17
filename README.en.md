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
- Chainlit login, chat history list, chat resume, memory viewing, searching,
  deletion confirmation, and current-context display.
- Tests covering configuration loading, the embedding provider factory,
  long-term memory taxonomy, memory extraction schema, and basic Qdrant store
  constraints.

## Project Layout

```text
.
├── chainlit_app.py              # Chainlit web UI entrypoint
├── docker-compose.chainlit-db.yml # Optional PostgreSQL persistence service
├── docker-compose.yml           # Docker Compose runtime with Qdrant and embedding
├── Dockerfile                   # Application image
├── embedding_server.py          # FastAPI embedding service entrypoint
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI/session helpers
│   ├── config.py                # Environment-backed settings
│   ├── embedding/
│   │   ├── base.py              # Embedding provider protocol
│   │   ├── factory.py           # Embedding provider factory
│   │   └── remote.py            # HTTP embedding service provider
│   ├── graph.py                 # LangGraph nodes and graph builder
│   ├── llm.py                   # OpenAI-compatible chat model loader
│   ├── memory_extractor.py      # Durable memory extraction logic
│   ├── memory_taxonomy.py       # Long-term memory types, namespaces, and budgets
│   ├── store/
│   │   ├── base.py              # Memory store protocol
│   │   ├── factory.py           # Qdrant store factory
│   │   └── qdrant_store.py      # Qdrant vector memory store
│   ├── prompts.py               # Prompt templates
│   └── state.py                 # LangGraph state schema
├── scripts/sql/init_chainlit_schema.sql # Chainlit UI history schema
├── tests/                       # Unit tests
├── pyproject.toml               # Python dependency metadata
└── .env.example                 # Safe environment template
```

## Requirements

- Python 3.12 or newer.
- An OpenAI-compatible chat completion endpoint.
- PostgreSQL for Chainlit UI history and LangGraph short-term checkpoints.
- Qdrant server, either started locally with Docker Compose or provided by a
  remote service.
- Enough disk space for the embedding model.

The project was developed against the dependency versions pinned in
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

Download the embedding model on the host machine.

If downloads are slow, set a Hugging Face mirror first:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
mkdir -p models
python3 - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.save("./models/bge-m3")
PY
```

Start the app:

```bash
docker compose -f docker-compose.yml -f docker-compose.chainlit-db.yml up --build
```

Open:

```text
http://127.0.0.1:8000
```

Docker setup notes:

- The default Compose file starts dedicated `embedding-service` and `qdrant`
  services.
- `docker-compose.chainlit-db.yml` starts `chainlit-postgres` and overrides
  `CHAINLIT_DATABASE_URL` inside the app container.
- `docker-compose.chainlit-db.yml` mounts
  `scripts/sql/init_chainlit_schema.sql` into the Postgres init directory. The
  script creates the Chainlit UI history tables and runs automatically only
  when a new `chainlit_postgres_data` volume is initialized for the first time.
  LangGraph checkpoint tables are created or migrated by the app on startup.
- `memory-agent` requests vectors through
  `EMBEDDING_SERVICE_URL=http://embedding-service:8001` and does not load the
  embedding model in the main process.
- Qdrant data is persisted in Docker named volume `qdrant_storage`.
- Chainlit chat history, the chat list, and LangGraph short-term checkpoints
  are persisted in Docker named volume `chainlit_postgres_data`.
- The app connects to Qdrant through `QDRANT_URL=http://qdrant:6333`.
- The default Compose file installs the CPU PyTorch wheel.
- The downloaded embedding model is mounted from `./models` into
  `embedding-service`.
- `.env` is read at runtime and is never copied into the image.

If you want to use a different embedding model, save it under `./models` and
update the `EMBEDDING_MODEL` container path in `docker-compose.yml` and
`EMBEDDING_DIMENSION` in `.env`.

To stop the app:

```bash
docker compose down
```

To reset Docker Qdrant and PostgreSQL data:

```bash
docker compose -f docker-compose.yml -f docker-compose.chainlit-db.yml down -v
```

If you already have an older `chainlit_postgres_data` volume, the new Chainlit
init script will not run retroactively. Apply the SQL manually or reset the
volume with the command above. LangGraph checkpoint tables are created or
migrated by the app on startup.

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
CHAINLIT_AUTH_SECRET=replace-with-a-random-secret
CHAINLIT_AUTH_USERNAME=your-login-name
CHAINLIT_AUTH_PASSWORD=your-login-password
CHAINLIT_AUTH_USER_ID=wenhao
```

Download the embedding model into the default local path.

If downloads are slow, set a Hugging Face mirror first:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
python - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")
model.save("./models/bge-m3")
PY
```

Local runs must connect to a Qdrant server; the app no longer uses
`QdrantClient(path=...)` embedded/local mode. You can start Qdrant with Docker:

```bash
docker compose up -d qdrant
```

For local Python runs, set `QDRANT_URL=http://127.0.0.1:6333`. Docker Compose
sets the app container to `http://qdrant:6333`. To connect to a remote Qdrant
server, set `QDRANT_URL` to that server URL.

The app only supports the remote embedding provider. Start the standalone
embedding service before running the Chainlit app:

```bash
source .venv/bin/activate
uvicorn embedding_server:app --host 127.0.0.1 --port 8001
```

`.env.example` defaults to `EMBEDDING_SERVICE_URL=http://127.0.0.1:8001`.

## Configuration

The application reads configuration from `.env`.

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
| `QDRANT_URL` | `http://127.0.0.1:6333` | Required Qdrant server URL. For local development, start it with `docker compose up -d qdrant`; the Docker Compose app container overrides this to `http://qdrant:6333`. |
| `QDRANT_API_KEY` | none | Remote Qdrant API key. |
| `QDRANT_COLLECTION` | `agent_memories` | Qdrant collection name for long-term memories. |
| `QDRANT_PREFER_GRPC` | `false` | Whether the Qdrant client should prefer gRPC. |
| `EMBEDDING_MODEL` | `./models/bge-m3` | Model path loaded by the embedding service. Docker Compose overrides this to `/app/models/bge-m3`. |
| `EMBEDDING_DEVICE` | `cpu` | Device used by the embedding service. CPU is the default and only supported runtime for now; legacy `auto` values are treated as CPU. |
| `EMBEDDING_DIMENSION` | `1024` | Current embedding model output dimension. It must match the Qdrant collection dimension, and existing collection dimensions are checked at startup. |
| `EMBEDDING_CONCURRENCY` | `1` | Embedding service inference concurrency. |
| `EMBEDDING_BATCH_SIZE` | `32` | Embedding service inference batch size. |
| `EMBEDDING_SERVICE_URL` | `http://127.0.0.1:8001` | Embedding service URL called by the main app; the Docker Compose app container overrides this to `http://embedding-service:8001`. |
| `EMBEDDING_TIMEOUT` | `30` | Remote embedding request timeout in seconds. |
| `EMBEDDING_TRUST_ENV` | `false` | Whether the remote embedding HTTP client should use proxy variables from the environment. |
| `CHAINLIT_AUTH_SECRET` | none | Secret used by Chainlit login/session cookies. Required when chat history is enabled. |
| `CHAINLIT_AUTH_USERNAME` | none | Chainlit password login username. |
| `CHAINLIT_AUTH_PASSWORD` | none | Chainlit password login password. |
| `CHAINLIT_AUTH_USER_ID` | none | Required. Chainlit authenticated user identifier. Long-term memory, chat history, and LangGraph context all use this identity. |
| `CHAINLIT_DATABASE_URL` | none | PostgreSQL database URL for Chainlit chat history, the chat list, and LangGraph short-term checkpoints. The Docker Postgres override replaces it with the in-network service URL. |
| `LANGGRAPH_STRICT_MSGPACK` | `true` | LangGraph checkpoint serialization safety switch. Keep it enabled unless you have a compatibility reason to change it. |
| `CONVERSATION_MESSAGE_WINDOW` | `20` | Number of recent conversation messages sent to the LLM on each turn. Long-term memories are still retrieved from Qdrant and injected separately. |
| `APP_UID` | `1000` | Docker image user id used by Compose build args. |
| `APP_GID` | `1000` | Docker image group id used by Compose build args. |
| `APP_DEBUG` | `false` | Enables extra backend debug output. |

The application has exactly one user identity source: the Chainlit
authenticated user. A typical setup can use `CHAINLIT_AUTH_USERNAME=admin` for
login and `CHAINLIT_AUTH_USER_ID=wenhao`; Qdrant namespaces are split by
memory type, such as `memories/wenhao/persona` and
`memories/wenhao/project`, and Chainlit history plus LangGraph
`Context.user_id` also use `wenhao`.

If your shell exports `DEBUG=release`, Chainlit may parse it as its own boolean
debug flag. Start Chainlit with `env -u DEBUG` as shown below.

## Run Tests

```bash
source .venv/bin/activate
python -m unittest discover -s tests
```

The tests cover configuration loading, PostgreSQL connection URL conversion,
the remote-only embedding provider factory, long-term memory taxonomy, memory
extraction schema, cross-namespace retrieval, and the required Qdrant URL
constraint.

## Run The Chainlit App

If you use the local service URLs from `.env.example`, start Qdrant and
PostgreSQL first:

```bash
docker compose -f docker-compose.yml -f docker-compose.chainlit-db.yml up -d qdrant chainlit-postgres
```

Then start the standalone embedding service:

```bash
source .venv/bin/activate
uvicorn embedding_server:app --host 127.0.0.1 --port 8001
```

Finally start Chainlit:

```bash
source .venv/bin/activate
env -u DEBUG chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The UI provides actions for:

- Viewing the Chainlit chat history list after login.
- Resuming a historical chat with the same LangGraph thread checkpoint.
- Viewing all long-term memories.
- Searching memories by query.
- Inspecting the current user, thread, model, and recent memory hits.
- Deleting individual memories after confirmation.

## Data And Ignored Files

The repository ignores local runtime artifacts:

- `.env`
- `qdrant_data/` (legacy local Qdrant data directory)
- `models/`
- `.chainlit/`
- `chainlit.md`
- Python cache directories
- Local ZIP archives

Do not commit API keys, legacy local Qdrant data, or downloaded model weights.

## Notes

- Memory storage is implemented in `QdrantMemoryStore`.
- `QdrantMemoryStore` only handles vector database reads and writes; embeddings
  are supplied by the remote `EmbeddingProvider`.
- Long-term memories use the fixed taxonomy in `memory_agent.memory_taxonomy`;
  each `memory_type` is written to its own Qdrant namespace and retrieved with
  a fixed per-type budget.
- Docker uses the standalone `embedding-service`, reducing model loading and
  inference pressure in the Chainlit main process.
- Chainlit chat history uses the SQLAlchemy data layer; LangGraph short-term
  context uses `AsyncPostgresSaver`. Both use the PostgreSQL database pointed
  to by `CHAINLIT_DATABASE_URL`.
- Memory extraction failures are logged and skipped instead of breaking the
  user-facing chat turn.
- The LLM client defaults to `LLM_TRUST_ENV=false` to avoid broken system proxy
  settings. Set it to `true` only if your endpoint requires environment proxy
  variables.
