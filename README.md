# Memory Agent

[中文](./README.md) | [English](./README.en.md)

Memory Agent 是一个基于 LangGraph、Chainlit、Qdrant 和可服务化 embedding 的长期记忆聊天代理。

应用会将用户记忆保存到 Qdrant 向量数据库中，在每次回复前通过语义检索召回相关记忆，并在每轮对话后提取可长期保存的记忆更新。

## 功能特性

- 使用 LangGraph 编排对话流程，并为每个线程保留短期检查点。
- 短期线程检查点默认使用进程内存储，重启后会丢失。
- 长期记忆存储在 Qdrant 中，支持向量检索、远程服务和本地持久化。
- 支持本地或独立 embedding 服务，Docker 默认将 embedding 推理移出主进程。
- 支持 OpenAI-compatible 聊天模型配置。
- 提供支持流式回复的 Chainlit UI。
- 支持记忆查看、搜索、删除确认和当前上下文展示。
- 提供 CLI 演示，覆盖记忆创建、更新与检索流程。

## 项目结构

```text
.
├── chainlit_app.py              # Chainlit Web UI 入口
├── docker-compose.gpu.yml       # 可选 NVIDIA GPU Compose 覆盖配置
├── docker-compose.yml           # Docker Compose 运行配置，包含 Qdrant 和 embedding 服务
├── Dockerfile                   # 应用镜像
├── embedding_server.py          # FastAPI embedding 服务入口
├── main.py                      # CLI 演示入口
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI / 会话辅助逻辑
│   ├── config.py                # 基于环境变量的配置
│   ├── embedding/
│   │   ├── base.py              # Embedding provider 协议
│   │   ├── factory.py           # Embedding provider 创建入口
│   │   ├── local.py             # 本地 sentence-transformers provider
│   │   └── remote.py            # HTTP embedding 服务 provider
│   ├── graph.py                 # LangGraph 节点与图构建
│   ├── llm.py                   # OpenAI-compatible 聊天模型加载
│   ├── memory_extractor.py      # 持久记忆提取逻辑
│   ├── store/
│   │   ├── base.py              # 记忆存储协议
│   │   ├── factory.py           # Qdrant store 创建入口
│   │   └── qdrant_store.py      # Qdrant 向量记忆存储
│   ├── prompts.py               # 提示词模板
│   └── state.py                 # LangGraph 状态定义
├── pyproject.toml               # Python 依赖元数据
└── .env.example                 # 安全环境变量模板
```

## 运行要求

- Python 3.12 或更高版本。
- 一个 OpenAI-compatible chat completion endpoint。
- Qdrant 服务，或 `qdrant-client` 的本地持久化模式。
- 足够存放 embedding 模型的磁盘空间。

项目基于 `pyproject.toml` 中固定版本的依赖进行开发与验证。

## 使用 Docker 快速启动

Docker Compose 会同时启动 Memory Agent、embedding 服务和 Qdrant 服务，是推荐的快速启动方式。

创建环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少设置：

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

Linux 下建议保持 `APP_UID=1000` 与 `APP_GID=1000`（常见默认用户），或者在构建前改为 `id -u` / `id -g` 的输出。容器将以该非 root 用户运行。

在主机下载 embedding 模型：

如果下载较慢，可先设置 Hugging Face 镜像：

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

启动 CPU 版本：

```bash
docker compose up --build
```

打开：

```text
http://127.0.0.1:8000
```

Docker 配置原则：

- 默认 Compose 文件会启动独立的 `embedding-service` 和 `qdrant` 服务。
- `memory-agent` 通过 `EMBEDDING_SERVICE_URL=http://embedding-service:8001` 请求向量，不在主进程加载 embedding 模型。
- Qdrant 数据持久化在 Docker named volume `qdrant_storage`。
- 应用通过 `QDRANT_URL=http://qdrant:6333` 访问向量数据库。
- 默认 Compose 文件安装 CPU 版 PyTorch，确保无 GPU 机器也能稳定运行。
- 下载的 embedding 模型通过 `./models` 挂载到 `embedding-service`。
- `.env` 在运行时读取，不会复制进镜像。

若需更换 embedding 模型，请将模型保存到 `./models`，同步更新 `docker-compose.yml` 中的 `EMBEDDING_MODEL` 容器路径，并在 `.env` 中更新 `EMBEDDING_DIMENSION`。

### Docker + NVIDIA GPU

GPU 支持是可选项。可加速 `embedding-service` 中的 embedding 推理，但需要 NVIDIA GPU、可用的主机驱动以及 Docker 的 NVIDIA Container Toolkit。Compose 的 GPU 预留格式遵循 Docker 官方指南：
https://docs.docker.com/compose/how-tos/gpu-support/

启动 GPU 版本：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

GPU 覆盖配置会让 `embedding-service` 使用 CUDA 12.6 的 PyTorch wheel，并请求 Compose 预留 NVIDIA GPU 设备。推荐保留 `EMBEDDING_DEVICE=auto` 以便可用时自动启用 CUDA，或显式指定：

```bash
EMBEDDING_DEVICE=cuda
```

可通过以下变量限制 GPU 数量：

```bash
GPU_COUNT=1
```

停止应用：

```bash
docker compose down
```

重置 Docker 中的 Qdrant 数据：

```bash
docker compose down -v
```

## 本地环境安装

创建并激活虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

安装项目依赖：

```bash
pip install -e .
```

创建本地环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少设置：

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
```

将 embedding 模型下载到默认本地路径：

如果下载较慢，可先设置 Hugging Face 镜像：

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

本地运行时，如果不设置 `QDRANT_URL`，应用会使用 `QDRANT_PATH` 指向的本地 Qdrant 持久化目录。

默认本地运行使用 `EMBEDDING_BACKEND=local`，会在当前进程加载 embedding 模型。若希望本地开发也使用服务化 embedding，可以先启动独立服务：

```bash
source .venv/bin/activate
uvicorn embedding_server:app --host 127.0.0.1 --port 8001
```

然后在运行 CLI 或 Chainlit 前设置：

```bash
EMBEDDING_BACKEND=remote
EMBEDDING_SERVICE_URL=http://127.0.0.1:8001
```

## 配置说明

应用通过 `.env` 读取配置。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL` | `mimo-v2.5-pro` | 发送到 OpenAI-compatible endpoint 的聊天模型名。 |
| `LLM_API_KEY` | none | LLM endpoint 的 API Key。 |
| `LLM_BASE_URL` | none | OpenAI-compatible base URL，通常以 `/v1` 结尾。 |
| `LLM_TEMPERATURE` | `0.7` | 采样温度。 |
| `LLM_MAX_COMPLETION_TOKENS` | `2048` | 最大 completion token 数，会作为 `max_completion_tokens` 传给 OpenAI-compatible API。 |
| `LLM_MAX_TOKENS` | none | 旧版兼容变量；仅在未设置 `LLM_MAX_COMPLETION_TOKENS` 时作为 fallback。 |
| `LLM_TIMEOUT` | `30` | LLM 请求超时时间（秒）。 |
| `LLM_TRUST_ENV` | `false` | HTTP 客户端是否使用环境变量中的代理配置。 |
| `LLM_STREAMING` | `true` | 普通聊天回复是否启用流式输出。 |
| `QDRANT_PATH` | `./qdrant_data` | 非 Docker 本地运行时的 Qdrant 持久化目录。设置 `QDRANT_URL` 后不会使用。 |
| `QDRANT_URL` | none | 远程 Qdrant 服务地址；Docker Compose 会覆盖为 `http://qdrant:6333`。 |
| `QDRANT_API_KEY` | none | 远程 Qdrant API Key。 |
| `QDRANT_COLLECTION` | `agent_memories` | 存储长期记忆的 Qdrant collection 名称。 |
| `QDRANT_PREFER_GRPC` | `false` | Qdrant 客户端是否优先使用 gRPC。 |
| `EMBEDDING_BACKEND` | `local` | Embedding provider，可选 `local` 或 `remote`。Docker Compose 会覆盖为 `remote`。 |
| `EMBEDDING_MODEL` | `./models/bge-m3` | 本地或 Hugging Face embedding 模型路径。Docker Compose 会覆盖为 `/app/models/bge-m3`。 |
| `EMBEDDING_DEVICE` | `auto` | embedding 设备，可用 `auto`、`cpu`、`cuda` 或 `cuda:0` 这类设备 id。 |
| `EMBEDDING_DIMENSION` | `1024` | 当前 embedding 模型输出向量维度，必须与 Qdrant collection 维度一致。 |
| `EMBEDDING_CONCURRENCY` | `1` | 本地 provider 或 embedding 服务内部并发推理数量。 |
| `EMBEDDING_BATCH_SIZE` | `32` | 本地 provider 或 embedding 服务的批量推理大小。 |
| `EMBEDDING_SERVICE_URL` | none | `EMBEDDING_BACKEND=remote` 时使用的 embedding 服务地址。 |
| `EMBEDDING_TIMEOUT` | `30` | remote embedding 请求超时时间（秒）。 |
| `EMBEDDING_TRUST_ENV` | `false` | remote embedding HTTP 客户端是否使用环境代理。 |
| `APP_UID` | `1000` | Compose 构建参数使用的 Docker 镜像用户 ID。 |
| `APP_GID` | `1000` | Compose 构建参数使用的 Docker 镜像组 ID。 |
| `DEFAULT_USER_ID` | `user_001` | 默认用户命名空间。 |
| `APP_DEBUG` | `false` | 启用额外后端调试输出。 |

如果你的 shell 导出 `DEBUG=release`，Chainlit 可能会将其解析为自身布尔调试标记。建议按下文示例使用 `env -u DEBUG` 启动 Chainlit。

## 运行 CLI 演示

```bash
source .venv/bin/activate
python main.py
```

演示会执行三轮：

1. 存储“偏好 Python Agent 编码”的记忆。
2. 将偏好更新为“Rust 优先”。
3. 在新线程中读取并验证该长期记忆。

演示数据会写入配置的 Qdrant collection。

## 运行 Chainlit 应用

```bash
source .venv/bin/activate
env -u DEBUG chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

UI 提供：

- 查看全部长期记忆。
- 按查询语句搜索记忆。
- 查看当前用户、线程、模型与近期命中记忆。
- 开启新短期线程（保留长期记忆）。
- 对单条记忆进行删除确认。

## 数据与忽略文件

仓库默认忽略本地运行时产物：

- `.env`
- `qdrant_data/`
- `models/`
- `.chainlit/`
- `chainlit.md`
- Python 缓存目录
- 本地压缩包

请勿提交 API Key、本地 Qdrant 数据或下载的模型权重。

## 备注

- 记忆存储由 `QdrantMemoryStore` 实现。
- `QdrantMemoryStore` 只负责向量数据库读写，embedding 由 `EmbeddingProvider` 提供。
- Docker 默认使用独立 `embedding-service`，减少 Chainlit 主进程的模型加载和推理压力。
- 记忆提取失败会记录日志并跳过，不会中断用户可见的聊天流程。
- LLM 客户端默认 `LLM_TRUST_ENV=false`，以避免系统代理配置导致异常。仅在你的 endpoint 确实依赖环境代理时再设为 `true`。
