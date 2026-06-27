# Memory Agent

[中文](./README.md) | [English](./README.en.md)

Memory Agent 是一个基于 LangGraph、Chainlit、Qdrant 和可服务化 embedding 的长期记忆聊天代理。

应用会将用户记忆按长期记忆 taxonomy 分类保存到 Qdrant 向量数据库中，在每次回复前通过语义检索召回相关记忆，并在每轮对话后提取可长期保存的记忆更新。

## 功能特性

- 使用 LangGraph 编排对话流程，并为每个线程保留短期检查点。
- 短期线程检查点持久化到 PostgreSQL，并与 Chainlit UI 历史共用同一个数据库。
- 长期记忆存储在 Qdrant server 中，支持向量检索和远程服务访问。
- 长期记忆按 `persona`、`entity`、`project`、`task`、`episodic`、`procedural`、`knowledge` 分类，并写入类型化 namespace。
- 仅通过独立 embedding 服务生成向量，避免主进程加载 embedding 模型。
- 支持 OpenAI-compatible 聊天模型配置。
- 提供支持流式和非流式回复的 Chainlit UI。
- 支持 Chainlit 登录、会话历史列表、会话恢复、记忆查看、搜索、删除确认和当前上下文展示。

## 项目结构

```text
.
├── chainlit_app.py              # Chainlit Web UI 入口
├── docker-compose.yml           # Docker Compose 运行配置，包含全部运行服务
├── Dockerfile                   # 应用镜像
├── embedding_server.py          # FastAPI embedding 服务入口
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI / 会话辅助逻辑
│   ├── config.py                # 基于环境变量的配置
│   ├── context_builder.py       # 短期对话上下文构建与裁剪
│   ├── graph.py                 # LangGraph 节点与图构建
│   ├── llm.py                   # OpenAI-compatible 聊天模型加载
│   ├── long_term_memory/        # 长期记忆独立子系统
│   │   ├── consolidator.py      # 记忆抽取、去重、更新与持久化
│   │   ├── retrieval.py         # 跨 namespace 召回与提示词格式化
│   │   ├── taxonomy.py          # 记忆分类、namespace 与召回预算
│   │   ├── prompts.py           # 长期记忆抽取提示词
│   │   ├── embedding/
│   │   │   ├── base.py          # Embedding provider 协议
│   │   │   ├── factory.py       # Embedding provider 创建入口
│   │   │   └── remote.py        # HTTP embedding 服务 provider
│   │   └── store/
│   │       ├── base.py          # 记忆存储协议
│   │       ├── factory.py       # Qdrant store 创建入口
│   │       └── qdrant.py        # Qdrant 向量记忆存储
│   ├── prompts.py               # 聊天系统提示词
│   └── state.py                 # LangGraph 状态定义
├── scripts/download_embedding_model.py # 一次性 Docker 模型下载脚本
├── scripts/sql/init_chainlit_schema.sql # Chainlit UI 历史初始化 schema
├── pyproject.toml               # Python 依赖元数据
└── .env.example                 # 安全环境变量模板
```

`memory_agent/long_term_memory/` 封装长期记忆的分类、召回、整合写入、
embedding 客户端和 Qdrant 存储实现。`graph.py` 只负责调用这些公共能力并
编排对话流程，短期上下文仍由根目录的 `context_builder.py` 独立管理。

## 数据存储架构

项目有三套逻辑存储域：

| 存储域 | 实现 | 存储内容 | 主要索引 |
| --- | --- | --- | --- |
| UI 会话存储 | Chainlit SQLAlchemy DataLayer，使用 PostgreSQL | 用户、会话列表、聊天消息 step、元素、反馈 | Chainlit authenticated user 和 Chainlit thread id |
| Graph 状态存储 | LangGraph `AsyncPostgresSaver`，使用 PostgreSQL | 每个线程的 checkpoint、短期对话状态和图运行状态 | `configurable.thread_id` |
| 长期记忆存储 | `QdrantMemoryStore`，使用 Qdrant server | 跨线程可复用的长期记忆、taxonomy metadata、向量索引 | `user_id` 和 `memory_type` namespace |

UI 会话存储和 Graph 状态存储目前共用 `CHAINLIT_DATABASE_URL` 指向的同一个 PostgreSQL 数据库，但它们的职责不同：UI 会话存储负责历史列表和消息回放，Graph 状态存储负责恢复 LangGraph 线程执行上下文。长期记忆存储独立放在 Qdrant 中，通过 `("memories", user_id, memory_type)` namespace 隔离用户和记忆类型。

身份和关联规则：

- `CHAINLIT_AUTH_USER_ID` 是唯一用户身份来源，并写入 Chainlit session、LangGraph `Context.user_id` 和长期记忆 namespace。
- Chainlit thread id 会作为 LangGraph `configurable.thread_id`，用于连接 UI 会话和 Graph checkpoint。
- 长期记忆 payload 中的 `source.thread_id` 只记录来源线程，便于追踪，不是长期记忆的主索引。
- 删除 UI 会话不会自动删除 Graph checkpoint 或长期记忆；删除长期记忆也不会改写历史消息。

## 运行要求

- Docker Engine 与 Docker Compose 插件。
- 一个 OpenAI-compatible chat completion endpoint。
- 足够存放 embedding 模型的磁盘空间。

项目默认工作流是 Docker-only：`memory-agent`、`embedding-service`、`qdrant`
和 `chainlit-postgres` 都由 `docker compose` 统一启动与管理，不再推荐在宿主机
直接运行 `chainlit_app.py` 或 `embedding_server.py`。

项目镜像基于 `pyproject.toml` 中固定版本的依赖进行构建与验证。

## 使用 Docker 快速启动

Docker Compose 会同时启动 Memory Agent、embedding 服务、Qdrant 和 PostgreSQL，是推荐的快速启动方式。

创建环境变量文件：

```bash
cp .env.example .env
```

编辑 `.env`，至少设置：

```bash
LLM_API_KEY=replace-with-your-api-key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
LLM_MODEL=your-model-name
CHAINLIT_AUTH_SECRET=replace-with-a-random-secret
CHAINLIT_AUTH_USERNAME=your-login-name
CHAINLIT_AUTH_PASSWORD=your-login-password
CHAINLIT_AUTH_USER_ID=wenhao
```

Linux 下建议保持 `APP_UID=1000` 与 `APP_GID=1000`（常见默认用户），或者在构建前改为 `id -u` / `id -g` 的输出。容器将以该非 root 用户运行。

模型下载默认使用 Hugging Face 镜像；如需覆盖，可在 `.env` 中设置：

```env
HF_ENDPOINT=https://hf-mirror.com
```

启动应用：

```bash
docker compose up --build
```

首次启动时，Compose 会先运行一次性的 `model-downloader` 服务，把
`EMBEDDING_MODEL_SOURCE` 对应的模型下载到 Docker named volume
`embedding_models`。默认会使用 `HF_ENDPOINT=https://hf-mirror.com`；
如果模型已经存在且来源未变化，后续启动会直接复用。

打开：

```text
http://127.0.0.1:8000
```

Docker 配置原则：

- 默认 Compose 文件会同时启动 `model-downloader`、`embedding-service`、
  `qdrant` 和 `chainlit-postgres`。
- `docker-compose.yml` 会将 `scripts/sql/init_chainlit_schema.sql` 挂载到 Postgres 初始化目录；该脚本负责 Chainlit UI 历史表，只会在新的 `chainlit_postgres_data` volume 首次初始化时自动执行。LangGraph checkpoint 表由应用启动时自动创建或迁移。
- `memory-agent` 默认通过 `EMBEDDING_SERVICE_URL=http://embedding-service:8001` 请求向量，不在主进程加载 embedding 模型。
- `model-downloader` 会将 `EMBEDDING_MODEL_SOURCE` 下载到 Docker named volume `embedding_models`，`embedding-service` 只读挂载该 volume 并从 `/app/models/current` 加载模型。
- embedding 模型会持久化在 Docker named volume `embedding_models`。
- Qdrant 数据持久化在 Docker named volume `qdrant_storage`。
- Chainlit 会话列表、聊天历史和 LangGraph 短期 checkpoint 持久化在 Docker named volume `chainlit_postgres_data`。
- 应用默认通过 `QDRANT_URL=http://qdrant:6333` 访问向量数据库。
- 应用默认通过 `CHAINLIT_DATABASE_URL=postgresql+asyncpg://memory_agent:memory_agent@chainlit-postgres:5432/chainlit` 访问 PostgreSQL。
- `embedding-service` 会等待 `model-downloader` 成功完成后再启动。
- `memory-agent` 会等待 embedding 服务和 PostgreSQL 通过健康检查后再启动；Qdrant 的可连接性则由应用启动阶段的重试逻辑兜底，避免容器刚启动时短暂出现 `Connection refused`。
- 默认 Compose 文件中的 embedding 服务固定使用 `EMBEDDING_DEVICE=cpu`，不做 GPU/XPU/MPS 设备探测。
- `.env` 只保留用户通常需要改动的业务配置；容器内部服务地址默认不写进 `.env`。
- `.env` 在运行时读取，不会复制进镜像。

若需更换 embedding 模型，请在 `.env` 中更新 `EMBEDDING_MODEL_SOURCE`，
并同步更新 `EMBEDDING_DIMENSION`。不同模型生成的向量不能混用，即使它们的
维度相同；切换模型时应使用新的 `QDRANT_COLLECTION`，或迁移并重新生成已有
记忆的向量。如果要强制清空旧模型缓存，可执行 `docker compose down -v`。

停止应用：

```bash
docker compose down
```

重置 Docker 中的 embedding 模型、Qdrant 和 PostgreSQL 数据：

```bash
docker compose down -v
```

如果你已有旧的 `chainlit_postgres_data` volume，新增的 Chainlit 初始化脚本不会自动补跑；需要手动执行 SQL，或按上面的命令重置 volume。LangGraph checkpoint 表会在应用启动时自动创建或迁移。

## 配置说明

应用通过 `.env` 读取用户常用配置。Qdrant、embedding 服务与 PostgreSQL
的容器内地址会使用 Docker Compose 与应用代码中的默认值自动完成，一般无需
手动写入 `.env`。

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
| `LLM_STREAMING` | `true` | 普通聊天回复是否启用流式输出；关闭时 Chainlit 会使用模型最终输出。 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | `model-downloader` 默认使用的 Hugging Face 镜像地址。 |
| `QDRANT_COLLECTION` | `agent_memories` | 存储长期记忆的 Qdrant collection 名称。 |
| `EMBEDDING_MODEL_SOURCE` | `BAAI/bge-m3` | 首次启动时由 `model-downloader` 下载的 Hugging Face 模型标识。 |
| `EMBEDDING_DIMENSION` | `1024` | 当前 embedding 模型输出向量维度，必须与 Qdrant collection 维度一致；启动时会检查已有 collection 的维度。 |
| `EMBEDDING_TIMEOUT` | `30` | remote embedding 请求超时时间（秒）。 |
| `EMBEDDING_TRUST_ENV` | `false` | remote embedding HTTP 客户端是否使用环境代理。 |
| `CHAINLIT_AUTH_SECRET` | none | Chainlit 登录和会话 cookie 使用的密钥，启用历史列表时必须设置。 |
| `CHAINLIT_AUTH_USERNAME` | none | Chainlit 密码登录用户名。 |
| `CHAINLIT_AUTH_PASSWORD` | none | Chainlit 密码登录密码。 |
| `CHAINLIT_AUTH_USER_ID` | none | 必填；Chainlit authenticated user identifier。长期记忆、历史列表和 LangGraph context 都使用这个身份。 |
| `LANGGRAPH_STRICT_MSGPACK` | `true` | LangGraph checkpoint 序列化安全开关；建议保持开启。 |
| `CONVERSATION_MESSAGE_WINDOW` | `20` | 每轮短期对话上下文的最大消息预算；上下文构建器会优先保留当前用户消息和最近完整对话轮次，长期记忆仍通过 Qdrant 检索注入。 |
| `APP_UID` | `1000` | Compose 构建参数使用的 Docker 镜像用户 ID。 |
| `APP_GID` | `1000` | Compose 构建参数使用的 Docker 镜像组 ID。 |
| `APP_DEBUG` | `false` | 启用额外后端调试输出。 |

若你确实需要偏离默认拓扑，仍可手动添加高级环境变量，例如
`QDRANT_URL`、`QDRANT_API_KEY`、`QDRANT_PREFER_GRPC`、
`EMBEDDING_SERVICE_URL`、`CHAINLIT_DATABASE_URL`、
`EMBEDDING_CONCURRENCY`、`EMBEDDING_BATCH_SIZE` 或 `EMBEDDING_MODEL`。

身份来源只有 Chainlit authenticated user。典型配置中可以使用 `CHAINLIT_AUTH_USERNAME=admin` 作为登录账号，同时设置 `CHAINLIT_AUTH_USER_ID=wenhao`；应用内的 Qdrant namespace 按记忆类型拆分，例如 `memories/wenhao/persona` 和 `memories/wenhao/project`，Chainlit 历史用户和 LangGraph `Context.user_id` 也都是 `wenhao`。

## UI 能力

UI 提供：

- 登录后查看 Chainlit 会话历史列表。
- 从历史列表恢复会话，并继续使用同一个 LangGraph thread checkpoint。
- 查看全部长期记忆。
- 按查询语句搜索记忆。
- 查看当前用户、线程、模型与近期命中记忆。
- 在每轮助手回复末尾查看 LLM 首响耗时和该轮 LLM token 用量。
- 对单条记忆进行删除确认。

## 数据与忽略文件

仓库默认忽略本地运行时产物：

- `.env`
- `qdrant_data/`（旧版本地 Qdrant 数据目录）
- `models/`（旧版宿主机模型目录）
- `.chainlit/`
- `chainlit.md`
- Python 缓存目录
- 本地压缩包

请勿提交 API Key、旧版本地 Qdrant 数据或手动下载的模型权重。

## 备注

- 记忆存储由 `QdrantMemoryStore` 实现。
- `QdrantMemoryStore` 只负责向量数据库读写，embedding 由远程 `EmbeddingProvider` 提供。
- 短期对话上下文由 `context_builder` 构建；它过滤空消息、忽略孤立 assistant 消息，并优先保留当前用户消息和最近完整对话轮次。
- 长期记忆能力集中在 `memory_agent.long_term_memory` 子系统；固定 taxonomy 位于 `long_term_memory.taxonomy`，每种 `memory_type` 写入独立 Qdrant namespace，并在检索时按固定预算召回。
- Docker 使用独立 `embedding-service`，减少 Chainlit 主进程的模型加载和推理压力。
- Chainlit 会话历史使用 SQLAlchemy data layer；LangGraph 短期上下文使用 `AsyncPostgresSaver`，两者共用 `CHAINLIT_DATABASE_URL` 指向的 PostgreSQL。
- 记忆整合失败会记录日志并跳过，不会中断用户可见的聊天流程。
- LLM 客户端默认 `LLM_TRUST_ENV=false`，以避免系统代理配置导致异常。仅在你的 endpoint 确实依赖环境代理时再设为 `true`。
