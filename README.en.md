# Memory Agent

[中文](./README.md) | [English](./README.en.md)

Memory Agent is a long-term memory chatbot built with LangGraph, Chainlit,
Qdrant, and service-ready embedding.

The app stores user memories in Qdrant, retrieves relevant memories with
semantic search before each response, and extracts durable memory updates after
each turn.

## Features

- LangGraph conversation flow with a per-thread short-term checkpoint.
- Short-term thread checkpoints are stored in memory by default and are lost on restart.
- Long-term memories are stored in Qdrant with vector search, remote service,
  and local persistence support.
- Local or remote embedding provider. Docker moves embedding inference out of
  the main Chainlit process by default.
- OpenAI-compatible chat model configuration.
- Chainlit UI with streaming responses.
- Memory viewing, searching, deletion confirmation, and current-context display.
- CLI demo that exercises memory creation, update, and retrieval.

## Project Layout

```text
.
├── chainlit_app.py              # Chainlit web UI entrypoint
├── docker-compose.gpu.yml       # Optional NVIDIA GPU Compose override
├── docker-compose.yml           # Docker Compose runtime with Qdrant and embedding
├── Dockerfile                   # Application image
├── embedding_server.py          # FastAPI embedding service entrypoint
├── main.py                      # CLI demo entrypoint
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI/session helpers
│   ├── config.py                # Environment-backed settings
│   ├── embedding/
│   │   ├── base.py              # Embedding provider protocol
│   │   ├── factory.py           # Embedding provider factory
│   │   ├── local.py             # Local sentence-transformers provider
│   │   └── remote.py            # HTTP embedding service provider
│   ├── graph.py                 # LangGraph nodes and graph builder
│   ├── llm.py                   # OpenAI-compatible chat model loader
│   ├── memory_extractor.py      # Durable memory extraction logic
│   ├── store/
│   │   ├── base.py              # Memory store protocol
│   │   ├── factory.py           # Qdrant store factory
│   │   └── qdrant_store.py      # Qdrant vector memory store
│   ├── prompts.py               # Prompt templates
│   └── state.py                 # LangGraph state schema
├── pyproject.toml               # Python dependency metadata
└── .env.example                 # Safe environment template
```

## Requirements

- Python 3.12 or newer.
- An OpenAI-compatible chat completion endpoint.
- Qdrant service, or local persistence mode through `qdrant-client`.
- Enough disk space for the embedding model.

The project was developed against the dependency versions pinned in
`pyproject.toml`.

## Quick Start With Docker

Docker Compose starts Memory Agent, the embedding service, and Qdrant, and is
the recommended quick start path.

Create your environment file:

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
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

Start the CPU app:

```bash
docker compose up --build
```

Open:

```text
http://127.0.0.1:8000
```

Docker setup notes:

- The default Compose file starts dedicated `embedding-service` and `qdrant`
  services.
- `memory-agent` requests vectors through
  `EMBEDDING_SERVICE_URL=http://embedding-service:8001` and does not load the
  embedding model in the main process.
- Qdrant data is persisted in Docker named volume `qdrant_storage`.
- The app connects to Qdrant through `QDRANT_URL=http://qdrant:6333`.
- The default Compose file installs the CPU PyTorch wheel, so machines without
  a GPU can run the app predictably.
- The downloaded embedding model is mounted from `./models` into
  `embedding-service`.
- `.env` is read at runtime and is never copied into the image.

If you want to use a different embedding model, save it under `./models` and
update the `EMBEDDING_MODEL` container path in `docker-compose.yml` and
`EMBEDDING_DIMENSION` in `.env`.

### Docker With NVIDIA GPU

GPU support is optional. It is useful for faster embedding inference in
`embedding-service`, but
it requires an NVIDIA GPU, a working host driver, and NVIDIA Container Toolkit
configured for Docker. The Compose GPU reservation format follows Docker's
official GPU support guide:
https://docs.docker.com/compose/how-tos/gpu-support/

Start the GPU app:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The GPU override switches `embedding-service` to the CUDA 12.6 PyTorch wheel and
asks Docker Compose to reserve NVIDIA GPU devices for that service. Keep
`EMBEDDING_DEVICE=auto` to use CUDA when it is available, or set it explicitly:

```bash
EMBEDDING_DEVICE=cuda
```

You can limit GPU access by setting:

```bash
GPU_COUNT=1
```

To stop the app:

```bash
docker compose down
```

To reset Docker Qdrant data:

```bash
docker compose down -v
```

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

For local runs, if `QDRANT_URL` is not set, the app uses local Qdrant
persistence under `QDRANT_PATH`.

Local runs use `EMBEDDING_BACKEND=local` by default, which loads the embedding
model in the current process. To use the service-based embedding path during
local development, start the standalone service first:

```bash
source .venv/bin/activate
uvicorn embedding_server:app --host 127.0.0.1 --port 8001
```

Then set these before running the CLI or Chainlit app:

```bash
EMBEDDING_BACKEND=remote
EMBEDDING_SERVICE_URL=http://127.0.0.1:8001
```

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
| `LLM_STREAMING` | `true` | Whether normal chat responses should stream. |
| `QDRANT_PATH` | `./qdrant_data` | Local Qdrant persistence path for non-Docker runs. Ignored when `QDRANT_URL` is set. |
| `QDRANT_URL` | none | Remote Qdrant service URL. Docker Compose overrides this to `http://qdrant:6333`. |
| `QDRANT_API_KEY` | none | Remote Qdrant API key. |
| `QDRANT_COLLECTION` | `agent_memories` | Qdrant collection name for long-term memories. |
| `QDRANT_PREFER_GRPC` | `false` | Whether the Qdrant client should prefer gRPC. |
| `EMBEDDING_BACKEND` | `local` | Embedding provider, either `local` or `remote`. Docker Compose overrides this to `remote`. |
| `EMBEDDING_MODEL` | `./models/bge-m3` | Local or Hugging Face embedding model path. Docker Compose overrides this to `/app/models/bge-m3`. |
| `EMBEDDING_DEVICE` | `auto` | Embedding device. Use `auto`, `cpu`, `cuda`, or a device id such as `cuda:0`. |
| `EMBEDDING_DIMENSION` | `1024` | Current embedding model output dimension. It must match the Qdrant collection dimension. |
| `EMBEDDING_CONCURRENCY` | `1` | Local provider or embedding service inference concurrency. |
| `EMBEDDING_BATCH_SIZE` | `32` | Local provider or embedding service inference batch size. |
| `EMBEDDING_SERVICE_URL` | none | Embedding service URL used when `EMBEDDING_BACKEND=remote`. |
| `EMBEDDING_TIMEOUT` | `30` | Remote embedding request timeout in seconds. |
| `EMBEDDING_TRUST_ENV` | `false` | Whether the remote embedding HTTP client should use proxy variables from the environment. |
| `APP_UID` | `1000` | Docker image user id used by Compose build args. |
| `APP_GID` | `1000` | Docker image group id used by Compose build args. |
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

The demo writes to the configured Qdrant collection.

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
- `qdrant_data/`
- `models/`
- `.chainlit/`
- `chainlit.md`
- Python cache directories
- Local ZIP archives

Do not commit API keys, local Qdrant data, or downloaded model weights.

## Notes

- Memory storage is implemented in `QdrantMemoryStore`.
- `QdrantMemoryStore` only handles vector database reads and writes; embeddings
  are supplied by `EmbeddingProvider`.
- Docker uses the standalone `embedding-service` by default, reducing model
  loading and inference pressure in the Chainlit main process.
- Memory extraction failures are logged and skipped instead of breaking the
  user-facing chat turn.
- The LLM client defaults to `LLM_TRUST_ENV=false` to avoid broken system proxy
  settings. Set it to `true` only if your endpoint requires environment proxy
  variables.
