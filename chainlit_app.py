import logging
from typing import Any
from uuid import uuid4

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from memory_agent import Context, SQLiteVectorMemoryStore, build_graph, load_settings
from memory_agent.llm import close_llm_clients


load_dotenv()
settings = load_settings()
logger = logging.getLogger(__name__)

store = SQLiteVectorMemoryStore(
    db_path=settings.memory_db_path,
    embedding_model=settings.embedding_model,
)

checkpointer = InMemorySaver()

graph = build_graph(
    store=store,
    checkpointer=checkpointer,
)


def _message_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="show_memories",
            payload={},
            label="查看记忆",
            tooltip="显示当前用户的长期记忆",
            icon="database",
        ),
        cl.Action(
            name="search_memories",
            payload={},
            label="搜索记忆",
            tooltip="按关键词检索长期记忆",
            icon="search",
        ),
        cl.Action(
            name="show_context",
            payload={},
            label="当前状态",
            tooltip="显示当前线程和最近一次记忆命中",
            icon="activity",
        ),
        cl.Action(
            name="new_thread",
            payload={},
            label="新线程",
            tooltip="保留长期记忆，开启新的短期对话上下文",
            icon="messages-square",
        ),
    ]


def _memory_actions(memory_id: str) -> list[cl.Action]:
    return [
        cl.Action(
            name="delete_memory",
            payload={"memory_id": memory_id},
            label="删除",
            tooltip="删除这条长期记忆",
            icon="trash-2",
        )
    ]


def _get_session_user_id() -> str:
    user_id = cl.user_session.get("user_id")
    if user_id:
        return user_id

    cl.user_session.set("user_id", settings.default_user_id)
    return settings.default_user_id


def _get_session_thread_id() -> str:
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        return thread_id

    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)
    return thread_id


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


def _serialize_memory(mem) -> dict[str, Any]:
    return {
        "key": mem.key,
        "value": mem.value,
        "score": mem.score,
    }


def _store_last_memory_hits(memories) -> None:
    cl.user_session.set(
        "last_memory_hits",
        [_serialize_memory(mem) for mem in memories],
    )


def _format_memory_card(memory: dict[str, Any], index: int | None = None) -> str:
    value = memory["value"]
    score = memory.get("score")
    title = f"### 记忆 {index}" if index is not None else "### 记忆"
    score_line = f"\n**Score**: `{score:.3f}`" if score is not None else ""

    return (
        f"{title}\n"
        f"**ID**: `{memory['key']}`\n"
        f"**Category**: `{value.get('category', 'general')}`\n"
        f"**Confidence**: `{value.get('confidence', 1.0)}`"
        f"{score_line}\n"
        f"**Updated**: `{value.get('updated_at', '')}`\n\n"
        f"{value.get('content', '')}\n\n"
        f"> {value.get('context', '') or '无额外上下文'}"
    )


def _answer_text(answer: Any) -> str:
    if not answer:
        return ""

    if isinstance(answer, dict):
        value = answer.get("output") or answer.get("content") or ""
        return str(value).strip()

    return str(answer).strip()


@cl.on_app_shutdown
async def on_app_shutdown():
    await close_llm_clients()


@cl.on_chat_start
async def on_chat_start():
    user_id = settings.default_user_id
    thread_id = f"chat-{uuid4()}"

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("last_memory_hits", [])

    await cl.Message(
        author="Memory Agent",
        content=(
            "你好，我是你的 Memory Agent。\n\n"
            f"- user_id: `{user_id}`\n"
            f"- model: `{settings.llm_model}`\n"
            f"- thread_id: `{thread_id}`\n\n"
            "你可以直接开始对话。"
        ),
        actions=_message_actions(),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_id = _get_session_user_id()
    thread_id = _get_session_thread_id()
    memory_hits = await _safe_search_memories(
        user_id=user_id,
        query=message.content,
        limit=5,
    )
    _store_last_memory_hits(memory_hits)

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
        author="Memory Agent",
        actions=_message_actions(),
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
                author="Memory Agent",
                content="这轮回复已经输出，但后续处理时出现异常。请检查后端日志。",
                actions=_message_actions(),
            ).send()

    await response_msg.update()


@cl.action_callback("show_memories")
async def show_memories(action: cl.Action):
    user_id = _get_session_user_id()

    memories = await _safe_search_memories(
        user_id=user_id,
        query="",
        limit=20,
    )
    _store_last_memory_hits(memories)

    if not memories:
        await cl.Message(
            author="Memory Agent",
            content="当前没有长期记忆。",
            actions=_message_actions(),
        ).send()
        return

    await cl.Message(
        author="Memory Agent",
        content=f"当前共有 `{len(memories)}` 条长期记忆。",
        actions=_message_actions(),
    ).send()

    for index, mem in enumerate(memories, start=1):
        memory = _serialize_memory(mem)
        await cl.Message(
            author="Memory Agent",
            content=_format_memory_card(memory, index),
            actions=_memory_actions(mem.key),
        ).send()


@cl.action_callback("search_memories")
async def search_memories(action: cl.Action):
    user_id = _get_session_user_id()
    answer = await cl.AskUserMessage(
        author="Memory Agent",
        content="输入记忆检索关键词",
        timeout=60,
    ).send()
    query = _answer_text(answer)

    if not query:
        await cl.Message(
            author="Memory Agent",
            content="未输入检索关键词。",
            actions=_message_actions(),
        ).send()
        return

    memories = await _safe_search_memories(
        user_id=user_id,
        query=query,
        limit=10,
    )
    _store_last_memory_hits(memories)

    if not memories:
        await cl.Message(
            author="Memory Agent",
            content=f"没有找到与 `{query}` 相关的长期记忆。",
            actions=_message_actions(),
        ).send()
        return

    await cl.Message(
        author="Memory Agent",
        content=f"找到 `{len(memories)}` 条相关长期记忆。",
        actions=_message_actions(),
    ).send()

    for index, mem in enumerate(memories, start=1):
        memory = _serialize_memory(mem)
        await cl.Message(
            author="Memory Agent",
            content=_format_memory_card(memory, index),
            actions=_memory_actions(mem.key),
        ).send()


@cl.action_callback("show_context")
async def show_context(action: cl.Action):
    hits = cl.user_session.get("last_memory_hits") or []

    header = (
        "### 当前状态\n"
        f"**user_id**: `{_get_session_user_id()}`\n"
        f"**thread_id**: `{_get_session_thread_id()}`\n"
        f"**model**: `{settings.llm_model}`\n"
        f"**最近记忆命中**: `{len(hits)}`"
    )

    await cl.Message(
        author="Memory Agent",
        content=header,
        actions=_message_actions(),
    ).send()

    for index, memory in enumerate(hits[:5], start=1):
        await cl.Message(
            author="Memory Agent",
            content=_format_memory_card(memory, index),
            actions=_memory_actions(memory["key"]),
        ).send()


@cl.action_callback("delete_memory")
async def delete_memory(action: cl.Action):
    user_id = _get_session_user_id()
    memory_id = action.payload.get("memory_id")

    if not memory_id:
        await cl.Message(
            author="Memory Agent",
            content="没有找到要删除的记忆 ID。",
            actions=_message_actions(),
        ).send()
        return

    memory = await store.aget(("memories", user_id), memory_id)
    if memory is None:
        await cl.Message(
            author="Memory Agent",
            content=f"记忆 `{memory_id}` 已不存在。",
            actions=_message_actions(),
        ).send()
        return

    confirmed = await cl.AskActionMessage(
        author="Memory Agent",
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
            author="Memory Agent",
            content="已取消删除。",
            actions=_message_actions(),
        ).send()
        return

    await store.adelete(("memories", user_id), memory_id)
    hits = cl.user_session.get("last_memory_hits") or []
    cl.user_session.set(
        "last_memory_hits",
        [memory for memory in hits if memory["key"] != memory_id],
    )

    await cl.Message(
        author="Memory Agent",
        content=f"已删除记忆 `{memory_id}`。",
        actions=_message_actions(),
    ).send()


@cl.action_callback("new_thread")
async def new_thread(action: cl.Action):
    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("last_memory_hits", [])

    await cl.Message(
        author="Memory Agent",
        content=(
            "已开启新的对话线程。\n\n"
            f"- user_id: `{_get_session_user_id()}`\n"
            f"- thread_id: `{thread_id}`"
        ),
        actions=_message_actions(),
    ).send()


# env -u DEBUG chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000
