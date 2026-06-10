# Memory Agent

[中文](./README.md) | [English](./README.en.md)

Memory Agent 是一个基于 LangGraph、Chainlit、SQLite 和本地 embedding 模型构建的长期记忆聊天代理。

应用会将用户记忆保存到 SQLite 数据库中，在每次回复前通过语义检索召回相关记忆，并在每轮对话后提取可长期保存的记忆更新。

## 功能特性

- 使用 LangGraph 编排对话流程，并为每个线程保留短期检查点。
- 基于 SQLite 的长期记忆存储。
- 使用本地 `sentence-transformers` embedding 模型进行语义记忆检索。
- 支持 OpenAI-compatible 聊天模型配置。
- 提供支持流式回复的 Chainlit UI。
- 支持记忆查看、搜索、删除确认和当前上下文展示。
- 提供 CLI 演示，覆盖记忆创建、更新与检索流程。

## 项目结构

```text
.
├── chainlit_app.py              # Chainlit Web UI 入口
├── docker-compose.gpu.yml       # 可选 NVIDIA GPU Compose 覆盖配置
├── docker-compose.yml           # Docker Compose 运行配置
├── Dockerfile                   # 轻量级应用镜像
├── main.py                      # CLI 演示入口
├── memory_agent/
│   ├── chainlit_ui.py           # Chainlit UI / 会话辅助逻辑
│   ├── config.py                # 基于环境变量的配置
│   ├── graph.py                 # LangGraph 节点与图构建
│   ├── llm.py                   # OpenAI-compatible 聊天模型加载
│   ├── memory_extractor.py      # 持久记忆提取逻辑
│   ├── persistent_store.py      # SQLite + embedding 记忆存储
│   ├── prompts.py               # 提示词模板
│   └── state.py                 # LangGraph 状态定义
├── pyproject.toml               # Python 依赖元数据
└── .env.example                 # 安全环境变量模板
```

## 运行要求

- Python 3.12 或更高版本。
- 一个 OpenAI-compatible chat completion endpoint。
- 足够存放本地 embedding 模型的磁盘空间。

项目基于 `pyproject.toml` 中固定版本的依赖进行开发与验证。

## 使用 Docker 快速启动

Docker 是最快的启动方式，适合不想手动管理本地 Python 环境的用户。

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

Linux 下建议保持 `APP_UID=1000` 与 `APP_GID=1000`（常见默认用户），
或者在构建前改为 `id -u` / `id -g` 的输出。
容器将以该非 root 用户运行，以保证挂载出来的记忆文件在主机侧可编辑。

在主机下载 embedding 模型：

如果下载较慢，可先设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
mkdir -p data models
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

- 镜像仅包含应用代码和 Python 依赖。
- 默认 Compose 文件安装 CPU 版 PyTorch，确保无 GPU 机器也能稳定运行。
- 应用以 `APP_UID` 与 `APP_GID` 指定的非 root 用户运行。
- 下载的 embedding 模型通过 `./models` 挂载。
- 长期记忆持久化在 `./data/memory.db`。
- `.env` 在运行时读取，不会复制进镜像。

若需更换 embedding 模型，请将模型保存到 `./models`，并在 `docker-compose.yml` 中把 `EMBEDDING_MODEL` 改为对应容器路径。

### Docker + NVIDIA GPU

GPU 支持是可选项。可加速本地 embedding 推理，但需要 NVIDIA GPU、可用的主机驱动以及 Docker 的 NVIDIA Container Toolkit。
Compose 的 GPU 预留格式遵循 Docker 官方指南：
https://docs.docker.com/compose/how-tos/gpu-support/

启动 GPU 版本：

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

GPU 覆盖配置会切换为 CUDA 12.6 的 PyTorch wheel，并请求 Compose 预留 NVIDIA GPU 设备。
推荐保留 `EMBEDDING_DEVICE=auto` 以便可用时自动启用 CUDA，或显式指定：

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

重置本地记忆：

```bash
rm -rf data
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

默认 `.env.example` 将 `EMBEDDING_MODEL` 指向 `./models/bge-m3`。

## 配置说明

应用通过 `.env` 读取配置。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_MODEL` | `mimo-v2.5-pro` | 发送到 OpenAI-compatible endpoint 的聊天模型名。 |
| `LLM_API_KEY` | none | LLM endpoint 的 API Key。 |
| `LLM_BASE_URL` | none | OpenAI-compatible base URL，通常以 `/v1` 结尾。 |
| `LLM_TEMPERATURE` | `0.7` | 采样温度。 |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token 数。 |
| `LLM_TIMEOUT` | `30` | LLM 请求超时时间（秒）。 |
| `LLM_TRUST_ENV` | `false` | HTTP 客户端是否使用环境变量中的代理配置。 |
| `LLM_STREAMING` | `true` | 普通聊天回复是否启用流式输出。 |
| `MEMORY_DB_PATH` | `./memory.db` | 长期记忆 SQLite 路径。Docker Compose 会覆盖为 `/app/data/memory.db`。 |
| `EMBEDDING_MODEL` | `./models/bge-m3` | 本地或 Hugging Face embedding 模型路径。Docker Compose 会覆盖为 `/app/models/bge-m3`。 |
| `EMBEDDING_DEVICE` | `auto` | embedding 设备，可用 `auto`、`cpu`、`cuda` 或 `cuda:0` 这类设备 id。 |
| `APP_UID` | `1000` | Compose 构建参数使用的 Docker 镜像用户 ID。 |
| `APP_GID` | `1000` | Compose 构建参数使用的 Docker 镜像组 ID。 |
| `DEFAULT_USER_ID` | `user_001` | 默认用户命名空间。 |
| `APP_DEBUG` | `false` | 启用额外后端调试输出。 |

如果你的 shell 导出 `DEBUG=release`，Chainlit 可能会将其解析为自身布尔调试标记。
建议按下文示例使用 `env -u DEBUG` 启动 Chainlit。

## 运行 CLI 演示

```bash
source .venv/bin/activate
python main.py
```

演示会执行三轮：

1. 存储“偏好 Python Agent 编码”的记忆。
2. 将偏好更新为“Rust 优先”。
3. 在新线程中读取并验证该长期记忆。

演示数据会写入 `MEMORY_DB_PATH` 指定的数据库。

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
- `*.db`
- `data/`
- `models/`
- `.chainlit/`
- `chainlit.md`
- Python 缓存目录

请勿提交 API Key、本地数据库或下载的模型权重。

## 备注

- 记忆存储由 `SQLiteVectorMemoryStore` 实现。
- SQLite 与 embedding 相关工作在异步友好的后台线程执行，减少 Chainlit 主流程阻塞概率。
- 记忆提取失败会记录日志并跳过，不会中断用户可见的聊天流程。
- LLM 客户端默认 `LLM_TRUST_ENV=false`，以避免系统代理配置导致异常。仅在你的 endpoint 确实依赖环境代理时再设为 `true`。
