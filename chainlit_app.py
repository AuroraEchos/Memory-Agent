import asyncio
import logging
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from memory_agent import Context, build_graph, create_memory_store, load_settings
from memory_agent.chainlit_ui import (
    ASSISTANT_AUTHOR,
    answer_text,
    format_memory_card,
    get_last_memory_hits,
    get_session_thread_id,
    get_session_user_id,
    init_session,
    memory_actions,
    message_actions,
    remove_memory_hit,
    resume_session,
    serialize_memory,
    store_last_memory_hits,
)
from memory_agent.llm import close_llm_clients


load_dotenv()
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
):
    raise RuntimeError(
        "CHAINLIT_AUTH_SECRET, CHAINLIT_AUTH_USERNAME, and "
        "CHAINLIT_AUTH_PASSWORD are required to enable Chainlit "
        "authentication and chat history."
    )


store = create_memory_store(settings)

chainlit_data_layer = SQLAlchemyDataLayer(
    conninfo=settings.chainlit_database_url,
    show_logger=settings.debug,
)

_graph: Any | None = None
_checkpointer_cm: Any | None = None
_checkpointer: Any | None = None
_graph_lock = asyncio.Lock()


@cl.data_layer
def get_data_layer():
    return chainlit_data_layer


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (
        username == settings.chainlit_auth_username
        and password == settings.chainlit_auth_password
    ):
        return cl.User(
            identifier=settings.chainlit_auth_user_id or username,
            metadata={
                "role": "user",
                "provider": "credentials",
            },
        )

    return None


async def ensure_graph():
    global _graph, _checkpointer_cm, _checkpointer

    if _graph is not None:
        return _graph

    async with _graph_lock:
        if _graph is not None:
            return _graph

        checkpoint_path = Path(settings.checkpoint_db_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        _checkpointer_cm = AsyncSqliteSaver.from_conn_string(
            str(checkpoint_path)
        )
        _checkpointer = await _checkpointer_cm.__aenter__()

        _graph = build_graph(
            store=store,
            checkpointer=_checkpointer,
        )

        logger.info("LangGraph checkpoint DB: %s", checkpoint_path)

        return _graph


async def _safe_search_memories(
    *,
    user_id: str,
    query: str = "",
    limit: int = 20,
):
    try:
        return await store.asearch(
            ("memories", user_id),
            query=query,
            limit=limit,
        )
    except Exception:
        logger.exception("Failed to search memories")
        return []


@cl.on_app_shutdown
async def on_app_shutdown():
    global _checkpointer_cm, _checkpointer, _graph

    try:
        if _checkpointer_cm is not None:
            await _checkpointer_cm.__aexit__(None, None, None)
    finally:
        _checkpointer_cm = None
        _checkpointer = None
        _graph = None

        await store.aclose()
        await close_llm_clients()


@cl.on_chat_start
async def on_chat_start():
    await ensure_graph()

    user_id, thread_id = init_session(settings.default_user_id)

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
    await ensure_graph()

    user_id, thread_id = resume_session(
        settings.default_user_id,
        thread,
    )

    if settings.debug:
        print("\n=== Chat Resumed ===")
        print(f"user_id={user_id}")
        print(f"thread_id={thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    graph = await ensure_graph()

    user_id = get_session_user_id(settings.default_user_id)
    thread_id = get_session_thread_id()

    memory_hits = await _safe_search_memories(
        user_id=user_id,
        query=message.content,
        limit=5,
    )
    store_last_memory_hits(memory_hits)

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = Context(
        user_id=user_id,
        model=settings.llm_model,
        debug=settings.debug,
    )

    response_msg = cl.Message(
        content="",
        author=ASSISTANT_AUTHOR,
        actions=message_actions(),
    )
    await response_msg.send()

    streamed = False

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

            if node_name != "call_model":
                continue

            if event_type != "on_chat_model_stream":
                continue

            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            token = getattr(chunk, "content", "")
            if isinstance(token, list):
                token = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in token
                )

            if token:
                streamed = True
                await response_msg.stream_token(token)

    except Exception:
        logger.exception("Graph execution failed")
        if not streamed:
            response_msg.content = (
                "抱歉，这轮对话没有成功完成。请检查后端日志和 `.env` 配置后重试。"
            )
        else:
            await cl.Message(
                author=ASSISTANT_AUTHOR,
                content="这轮回复已经输出，但后续处理时出现异常。请检查后端日志。",
                actions=message_actions(),
            ).send()

    await response_msg.update()


@cl.action_callback("show_memories")
async def show_memories(action: cl.Action):
    user_id = get_session_user_id(settings.default_user_id)

    memories = await _safe_search_memories(
        user_id=user_id,
        query="",
        limit=20,
    )
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
            actions=memory_actions(mem.key),
        ).send()


@cl.action_callback("search_memories")
async def search_memories(action: cl.Action):
    user_id = get_session_user_id(settings.default_user_id)
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
            actions=memory_actions(mem.key),
        ).send()


@cl.action_callback("show_context")
async def show_context(action: cl.Action):
    hits = get_last_memory_hits()

    header = (
        "### 当前状态\n"
        f"**user_id**: `{get_session_user_id(settings.default_user_id)}`\n"
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
            actions=memory_actions(memory["key"]),
        ).send()


@cl.action_callback("delete_memory")
async def delete_memory(action: cl.Action):
    user_id = get_session_user_id(settings.default_user_id)
    memory_id = action.payload.get("memory_id")

    if not memory_id:
        await cl.Message(
            author=ASSISTANT_AUTHOR,
            content="没有找到要删除的记忆 ID。",
            actions=message_actions(),
        ).send()
        return

    memory = await store.aget(("memories", user_id), memory_id)
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
                payload={"memory_id": memory_id},
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

    await store.adelete(("memories", user_id), memory_id)
    remove_memory_hit(memory_id)

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=f"已删除记忆 `{memory_id}`。",
        actions=message_actions(),
    ).send()
