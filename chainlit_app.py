"""Chainlit web entrypoint for the Memory Agent application."""

import asyncio
import logging
import time
from typing import Any

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from memory_agent import (
    Context,
    build_graph,
    create_memory_store,
    load_settings,
    to_psycopg_conninfo,
)
from memory_agent.chainlit_ui import (
    ASSISTANT_AUTHOR,
    answer_text,
    content_to_text,
    event_output_to_text,
    extract_token_usage,
    format_memory_card,
    format_response_latency,
    format_token_usage,
    get_last_memory_hits,
    get_session_thread_id,
    get_session_user_id,
    init_session,
    memory_actions,
    message_actions,
    remove_memory_hit,
    resume_session,
    serialize_memory,
    store_last_memory_hit_dicts,
    store_last_memory_hits,
)
from memory_agent.llm import close_llm_clients
from memory_agent.long_term_memory.taxonomy import (
    MEMORY_TYPE_VALUES,
    memory_namespace,
    normalize_memory_type,
)


load_dotenv(dotenv_path=".env", override=False)
settings = load_settings()
logger = logging.getLogger(__name__)


if not settings.chainlit_database_url:
    raise RuntimeError(
        "CHAINLIT_DATABASE_URL is required to enable Chainlit chat history."
    )

if (
    not settings.chainlit_auth_secret
    or not settings.chainlit_auth_username
    or not settings.chainlit_auth_password
    or not settings.chainlit_auth_user_id
):
    raise RuntimeError(
        "CHAINLIT_AUTH_SECRET, CHAINLIT_AUTH_USERNAME, "
        "CHAINLIT_AUTH_PASSWORD, and CHAINLIT_AUTH_USER_ID are required "
        "to enable Chainlit authentication, chat history, and memory identity."
    )


store = create_memory_store(settings)

chainlit_data_layer = SQLAlchemyDataLayer(
    conninfo=settings.chainlit_database_url,
    show_logger=settings.debug,
)
checkpoint_conninfo = to_psycopg_conninfo(settings.chainlit_database_url)

_graph: Any | None = None
_checkpointer_cm: Any | None = None
_checkpointer: Any | None = None
_embedding_dimension_validated = False
_graph_lock = asyncio.Lock()


@cl.data_layer
def get_data_layer():
    """Provide Chainlit with the configured SQLAlchemy data layer."""

    return chainlit_data_layer


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate the single configured password-based Chainlit user."""

    if (
        username == settings.chainlit_auth_username
        and password == settings.chainlit_auth_password
    ):
        return cl.User(
            identifier=settings.chainlit_auth_user_id,
            metadata={
                "role": "user",
                "provider": "credentials",
            },
        )

    return None


async def persist_message_step(message: Any) -> None:
    """Synchronously persist the final Chainlit step state.

    Chainlit persists message send/update operations in background tasks.
    Awaiting the data layer here makes resumed chat history much less likely
    to show a user message without the assistant's final answer.
    """

    try:
        await chainlit_data_layer.update_step(message.to_dict())
    except Exception:
        logger.exception("Failed to persist Chainlit message step")


async def ensure_embedding_dimension() -> None:
    """Validate the remote embedding dimension once before graph startup."""

    global _embedding_dimension_validated

    if _embedding_dimension_validated:
        return

    embedding_provider = getattr(store, "embedding_provider", None)
    adimension = getattr(embedding_provider, "adimension", None)

    if not callable(adimension):
        logger.warning(
            "Memory store embedding provider does not expose adimension(); "
            "skipping startup embedding dimension validation"
        )
        _embedding_dimension_validated = True
        return

    actual_dimension = int(await adimension())
    expected_dimension = int(settings.embedding_dimension)

    if actual_dimension != expected_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"EMBEDDING_DIMENSION={expected_dimension}, "
            f"embedding service returned {actual_dimension}."
        )

    _embedding_dimension_validated = True
    logger.info("Embedding dimension validated: %s", actual_dimension)


async def ensure_graph():
    """Lazily initialize the LangGraph app and PostgreSQL checkpointer."""

    global _graph, _checkpointer_cm, _checkpointer

    if _graph is not None:
        return _graph

    async with _graph_lock:
        if _graph is not None:
            return _graph

        try:
            await ensure_embedding_dimension()

            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                checkpoint_conninfo
            )
            _checkpointer = await _checkpointer_cm.__aenter__()
            await _checkpointer.setup()

            _graph = build_graph(
                store=store,
                checkpointer=_checkpointer,
            )

            logger.info("LangGraph checkpoint DB: PostgreSQL")

            return _graph

        except Exception:
            logger.exception("Failed to initialize LangGraph")

            if _checkpointer_cm is not None:
                try:
                    await _checkpointer_cm.__aexit__(None, None, None)
                except Exception:
                    logger.exception(
                        "Failed to close LangGraph checkpointer after "
                        "initialization failure"
                    )

            _checkpointer_cm = None
            _checkpointer = None
            _graph = None
            raise


async def _safe_search_memories(
    *,
    user_id: str,
    query: str = "",
    limit: int = 20,
    memory_type: str | None = None,
):
    """Search long-term memories across taxonomy namespaces safely."""

    memory_types = (
        [normalize_memory_type(memory_type)]
        if memory_type
        else list(MEMORY_TYPE_VALUES)
    )

    memories: list[Any] = []

    for current_type in memory_types:
        try:
            memories.extend(
                await store.asearch(
                    memory_namespace(user_id, current_type),
                    query=query,
                    limit=limit,
                )
            )
        except Exception:
            logger.exception("Failed to search %s memories", current_type)

    if query and query.strip():
        memories.sort(
            key=lambda mem: (
                getattr(mem, "score", None)
                if getattr(mem, "score", None) is not None
                else -1.0
            ),
            reverse=True,
        )
    else:
        memories = _sort_memories_latest_first(memories)

    return memories[:limit]


def _memory_updated_at(mem: Any) -> str:
    """Return a sortable update timestamp from a memory item."""

    value = getattr(mem, "value", {}) or {}
    if not isinstance(value, dict):
        return ""

    return str(value.get("updated_at") or value.get("created_at") or "")


def _sort_memories_latest_first(memories: list[Any]) -> list[Any]:
    """Sort memory items by newest update timestamp first."""

    return sorted(memories, key=_memory_updated_at, reverse=True)


@cl.on_app_shutdown
async def on_app_shutdown():
    """Release graph, memory-store, and LLM resources on shutdown."""

    global _checkpointer_cm, _checkpointer, _graph

    try:
        if _checkpointer_cm is not None:
            await _checkpointer_cm.__aexit__(None, None, None)
    except Exception:
        logger.exception("Failed to close LangGraph checkpointer")
    finally:
        _checkpointer_cm = None
        _checkpointer = None
        _graph = None

    try:
        await store.aclose()
    except Exception:
        logger.exception("Failed to close memory store")

    try:
        await close_llm_clients()
    except Exception:
        logger.exception("Failed to close LLM clients")


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new Chainlit chat session."""

    await ensure_graph()

    user_id, thread_id = init_session()

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=(
            "你好，我是你的 Memory Agent。\n\n"
            f"- user_id: `{user_id}`\n"
            f"- model: `{settings.llm_model}`\n"
            f"- thread_id: `{thread_id}`\n\n"
            "你可以直接开始对话。"
        ),
        actions=message_actions(),
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Restore session metadata when a Chainlit thread is resumed."""

    await ensure_graph()

    user_id, thread_id = resume_session(thread)

    if settings.debug:
        print("\n=== Chat Resumed ===")
        print(f"user_id={user_id}")
        print(f"thread_id={thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """Run the LangGraph workflow for one user message and stream the answer."""

    graph = await ensure_graph()

    user_id = get_session_user_id()
    thread_id = get_session_thread_id()

    # Clear stale UI memory hits. The graph returns the actual memories
    # injected into the model, which we store when call_model finishes.
    store_last_memory_hit_dicts([])

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = Context(
        user_id=user_id,
        thread_id=thread_id,
        model=settings.llm_model,
        debug=settings.debug,
        message_window=settings.conversation_message_window,
    )

    response_msg = cl.Message(
        content="",
        author=ASSISTANT_AUTHOR,
        actions=message_actions(),
    )

    streamed = False
    error_after_stream = False
    response_finalized = False
    token_usage: dict[str, int] = {}
    llm_started_at: float | None = None
    first_response_latency: float | None = None
    metrics_footer_started = False
    latency_displayed = False
    token_usage_displayed = False

    def mark_llm_started() -> None:
        """Record when the chat model call starts."""

        nonlocal llm_started_at

        if llm_started_at is None:
            llm_started_at = time.perf_counter()

    def mark_first_response_ready() -> None:
        """Record when the first chat model content is ready to display."""

        nonlocal first_response_latency

        if first_response_latency is None and llm_started_at is not None:
            first_response_latency = time.perf_counter() - llm_started_at

    async def maybe_append_response_metrics() -> None:
        """Append visible response timing and token usage metrics."""

        nonlocal latency_displayed, metrics_footer_started, token_usage_displayed

        latency_line = format_response_latency(first_response_latency)
        usage_line = format_token_usage(token_usage)

        lines: list[str] = []
        if latency_line and not latency_displayed:
            lines.append(latency_line)
            latency_displayed = True
        if usage_line and not token_usage_displayed:
            lines.append(usage_line)
            token_usage_displayed = True

        if not lines:
            return

        response_msg.content = response_msg.content.rstrip()
        separator = "\n" if metrics_footer_started else "\n\n---\n"
        response_msg.content += separator + "\n".join(lines)
        metrics_footer_started = True

        if response_finalized:
            await response_msg.update()
            await persist_message_step(response_msg)

    async def finalize_response(*, fallback_empty: bool = False) -> bool:
        """Send and persist the assistant response exactly once."""

        nonlocal response_finalized

        if response_finalized:
            return True

        if not response_msg.content.strip():
            if not fallback_empty:
                return False
            response_msg.content = "抱歉，这轮对话没有生成有效回复。请稍后重试。"

        await maybe_append_response_metrics()
        await response_msg.send()
        await persist_message_step(response_msg)
        response_finalized = True
        return True

    def maybe_store_graph_memory_hits(output: Any) -> None:
        """Store memory hits returned by the call_model graph node."""

        if not isinstance(output, dict):
            return

        memory_hits = output.get("memory_hits")
        if isinstance(memory_hits, list):
            store_last_memory_hit_dicts(memory_hits)

    try:
        async for event in graph.astream_events(
            {
                "messages": [
                    HumanMessage(content=message.content)
                ]
            },
            config=config,
            context=context,
            version="v2",
        ):
            event_type = event.get("event")
            metadata = event.get("metadata", {})
            node_name = metadata.get("langgraph_node")
            data = event.get("data", {})

            if node_name != "call_model":
                continue

            if event_type == "on_chat_model_start":
                mark_llm_started()
                continue

            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue

                token = content_to_text(getattr(chunk, "content", ""))

                if token:
                    mark_first_response_ready()
                    streamed = True
                    await response_msg.stream_token(token)

                continue

            if event_type in {"on_chat_model_end", "on_chain_end"}:
                output = data.get("output")
                current_token_usage = extract_token_usage(output)
                if current_token_usage:
                    token_usage = current_token_usage

                if event_type == "on_chain_end":
                    maybe_store_graph_memory_hits(output)

                final_text = event_output_to_text(output)
                if final_text and not response_msg.content.strip():
                    mark_first_response_ready()
                    response_msg.content = final_text
                await maybe_append_response_metrics()
                await finalize_response()

    except Exception:
        logger.exception("Graph execution failed")
        if response_finalized:
            error_after_stream = True
        elif not streamed:
            response_msg.content = (
                "抱歉，这轮对话没有成功完成。请检查后端日志和 `.env` 配置后重试。"
            )
        else:
            error_after_stream = True

    if not response_finalized:
        await finalize_response(fallback_empty=True)

    if error_after_stream:
        error_msg = cl.Message(
            author=ASSISTANT_AUTHOR,
            content="这轮回复已经输出，但后续处理时出现异常。请检查后端日志。",
            actions=message_actions(),
        )
        await error_msg.send()
        await persist_message_step(error_msg)


@cl.action_callback("show_memories")
async def show_memories(action: cl.Action):
    """Display the current user's long-term memories."""

    user_id = get_session_user_id()

    memories = await _safe_search_memories(
        user_id=user_id,
        query="",
        limit=20,
    )
    memories = _sort_memories_latest_first(memories)
    store_last_memory_hits(memories)

    if not memories:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="当前没有长期记忆。",
            actions=message_actions(),
        ).send()
        return

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=f"当前共有 `{len(memories)}` 条长期记忆。",
        actions=message_actions(),
    ).send()

    for index, mem in enumerate(memories, start=1):
        memory = serialize_memory(mem)
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content=format_memory_card(memory, index),
            actions=memory_actions(mem.key, memory["value"].get("memory_type")),
        ).send()


@cl.action_callback("search_memories")
async def search_memories(action: cl.Action):
    """Prompt for a search query and display matching memories."""

    user_id = get_session_user_id()
    answer = await cl.AskUserMessage(
        author=ASSISTANT_AUTHOR,
        content="输入记忆检索关键词",
        timeout=60,
    ).send()
    query = answer_text(answer)

    if not query:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="未输入检索关键词。",
            actions=message_actions(),
        ).send()
        return

    memories = await _safe_search_memories(
        user_id=user_id,
        query=query,
        limit=10,
    )
    store_last_memory_hits(memories)

    if not memories:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content=f"没有找到与 `{query}` 相关的长期记忆。",
            actions=message_actions(),
        ).send()
        return

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=f"找到 `{len(memories)}` 条相关长期记忆。",
        actions=message_actions(),
    ).send()

    for index, mem in enumerate(memories, start=1):
        memory = serialize_memory(mem)
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content=format_memory_card(memory, index),
            actions=memory_actions(mem.key, memory["value"].get("memory_type")),
        ).send()


@cl.action_callback("show_context")
async def show_context(action: cl.Action):
    """Display session identity and the latest model-injected memory hits."""

    hits = get_last_memory_hits()

    header = (
        "### 当前状态\n"
        f"**user_id**: `{get_session_user_id()}`\n"
        f"**thread_id**: `{get_session_thread_id()}`\n"
        f"**model**: `{settings.llm_model}`\n"
        f"**最近记忆命中**: `{len(hits)}`"
    )

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=header,
        actions=message_actions(),
    ).send()

    for index, memory in enumerate(hits[:5], start=1):
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content=format_memory_card(memory, index),
            actions=memory_actions(memory["key"], memory["value"].get("memory_type")),
        ).send()


@cl.action_callback("delete_memory")
async def delete_memory(action: cl.Action):
    """Confirm and delete a selected long-term memory."""

    user_id = get_session_user_id()
    memory_id = action.payload.get("memory_id")
    memory_type = action.payload.get("memory_type")

    if not memory_id:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="没有找到要删除的记忆 ID。",
            actions=message_actions(),
        ).send()
        return

    if not memory_type:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="没有找到这条记忆所属的 memory_type，无法定位 namespace。",
            actions=message_actions(),
        ).send()
        return

    namespace = memory_namespace(user_id, normalize_memory_type(memory_type))
    memory = await store.aget(namespace, memory_id)
    if memory is None:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content=f"记忆 `{memory_id}` 已不存在。",
            actions=message_actions(),
        ).send()
        return

    confirmed = await cl.AskActionMessage(
        author=ASSISTANT_AUTHOR,
        content=(
            "确认删除这条长期记忆？\n\n"
            f"> {memory.value.get('content', '')}"
        ),
        actions=[
            cl.Action(
                name="confirm_delete_memory",
                payload={"memory_id": memory_id, "memory_type": memory_type},
                label="确认删除",
                icon="trash-2",
            ),
            cl.Action(
                name="cancel_delete_memory",
                payload={},
                label="取消",
                icon="x",
            ),
        ],
        timeout=30,
    ).send()

    if not confirmed or confirmed.get("name") != "confirm_delete_memory":
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="已取消删除。",
            actions=message_actions(),
        ).send()
        return

    await store.adelete(namespace, memory_id)
    remove_memory_hit(memory_id)

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=f"已删除记忆 `{memory_id}`。",
        actions=message_actions(),
    ).send()
