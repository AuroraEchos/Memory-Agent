from typing import Any
from uuid import uuid4

import chainlit as cl


ASSISTANT_AUTHOR = "Memory Agent"
LAST_MEMORY_HITS_KEY = "last_memory_hits"


def message_actions() -> list[cl.Action]:
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


def memory_actions(memory_id: str) -> list[cl.Action]:
    return [
        cl.Action(
            name="delete_memory",
            payload={"memory_id": memory_id},
            label="删除",
            tooltip="删除这条长期记忆",
            icon="trash-2",
        )
    ]


def init_session(default_user_id: str) -> tuple[str, str]:
    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("user_id", default_user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set(LAST_MEMORY_HITS_KEY, [])
    return default_user_id, thread_id


def get_session_user_id(default_user_id: str) -> str:
    user_id = cl.user_session.get("user_id")
    if user_id:
        return user_id

    cl.user_session.set("user_id", default_user_id)
    return default_user_id


def get_session_thread_id() -> str:
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        return thread_id

    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)
    return thread_id


def reset_session_thread() -> str:
    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set(LAST_MEMORY_HITS_KEY, [])
    return thread_id


def serialize_memory(mem) -> dict[str, Any]:
    return {
        "key": mem.key,
        "value": mem.value,
        "score": mem.score,
    }


def store_last_memory_hits(memories) -> None:
    cl.user_session.set(
        LAST_MEMORY_HITS_KEY,
        [serialize_memory(mem) for mem in memories],
    )


def get_last_memory_hits() -> list[dict[str, Any]]:
    return cl.user_session.get(LAST_MEMORY_HITS_KEY) or []


def remove_memory_hit(memory_id: str) -> None:
    cl.user_session.set(
        LAST_MEMORY_HITS_KEY,
        [memory for memory in get_last_memory_hits() if memory["key"] != memory_id],
    )


def format_memory_card(memory: dict[str, Any], index: int | None = None) -> str:
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


def answer_text(answer: Any) -> str:
    if not answer:
        return ""

    if isinstance(answer, dict):
        value = answer.get("output") or answer.get("content") or ""
        return str(value).strip()

    return str(answer).strip()
