"""Chainlit UI helpers for sessions, actions, and memory rendering."""

from typing import Any
from uuid import uuid4

import chainlit as cl


ASSISTANT_AUTHOR = "Memory Agent"
LAST_MEMORY_HITS_KEY = "last_memory_hits"


def message_actions() -> list[cl.Action]:
    """Build action buttons attached to general assistant messages."""

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
    ]


def memory_actions(memory_id: str) -> list[cl.Action]:
    """Build action buttons attached to a single memory card."""

    return [
        cl.Action(
            name="delete_memory",
            payload={"memory_id": memory_id},
            label="删除",
            tooltip="删除这条长期记忆",
            icon="trash-2",
        )
    ]


def get_authenticated_user_id() -> str:
    """Return the authenticated Chainlit user identifier."""

    try:
        user = cl.user_session.get("user")
    except Exception as exc:
        raise RuntimeError("A Chainlit authenticated user is required.") from exc

    identifier = getattr(user, "identifier", None)
    if identifier:
        return str(identifier)

    raise RuntimeError("Authenticated Chainlit user has no identifier.")


def get_current_chainlit_thread_id() -> str:
    """Return the current Chainlit thread/session id.

    Different Chainlit versions expose this slightly differently.
    We prefer session.thread_id when available, then user_session["id"],
    then session.id, then a fallback uuid.
    """

    try:
        session = getattr(cl.context, "session", None)
    except Exception:
        session = None

    thread_id = getattr(session, "thread_id", None)
    if thread_id:
        return str(thread_id)

    try:
        session_id = cl.user_session.get("id")
    except Exception:
        session_id = None

    if session_id:
        return str(session_id)

    raw_session_id = getattr(session, "id", None)
    if raw_session_id:
        return str(raw_session_id)

    return str(uuid4())


def init_session(
    *,
    thread_id: str | None = None,
) -> tuple[str, str]:
    """Initialize Chainlit session state for a new chat."""

    user_id = get_authenticated_user_id()
    thread_id = thread_id or get_current_chainlit_thread_id()

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set(LAST_MEMORY_HITS_KEY, [])

    return user_id, thread_id


def resume_session(
    thread: dict[str, Any],
) -> tuple[str, str]:
    """Restore Chainlit session state from a resumed thread."""

    user_id = get_authenticated_user_id()

    thread_id = (
        thread.get("id")
        or thread.get("thread_id")
        or get_current_chainlit_thread_id()
    )
    thread_id = str(thread_id)

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)

    if cl.user_session.get(LAST_MEMORY_HITS_KEY) is None:
        cl.user_session.set(LAST_MEMORY_HITS_KEY, [])

    return user_id, thread_id


def get_session_user_id() -> str:
    """Return and refresh the authenticated session user id."""

    user_id = get_authenticated_user_id()
    cl.user_session.set("user_id", user_id)
    return user_id


def get_session_thread_id() -> str:
    """Return the current LangGraph thread id for this session."""

    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        return str(thread_id)

    thread_id = get_current_chainlit_thread_id()
    cl.user_session.set("thread_id", thread_id)
    return thread_id


def serialize_memory(mem) -> dict[str, Any]:
    """Convert a MemoryItem-like object into a UI-safe dictionary."""

    return {
        "key": mem.key,
        "value": mem.value,
        "score": mem.score,
    }


def store_last_memory_hits(memories) -> None:
    """Store recently displayed MemoryItem objects in the Chainlit session."""

    cl.user_session.set(
        LAST_MEMORY_HITS_KEY,
        [serialize_memory(mem) for mem in memories],
    )


def store_last_memory_hit_dicts(memories: list[dict[str, Any]]) -> None:
    """Store already-serialized memory hits in the Chainlit session."""

    cl.user_session.set(LAST_MEMORY_HITS_KEY, memories)


def get_last_memory_hits() -> list[dict[str, Any]]:
    """Return the latest memory hits cached in the Chainlit session."""

    return cl.user_session.get(LAST_MEMORY_HITS_KEY) or []


def remove_memory_hit(memory_id: str) -> None:
    """Remove one memory id from the cached memory hits."""

    cl.user_session.set(
        LAST_MEMORY_HITS_KEY,
        [memory for memory in get_last_memory_hits() if memory["key"] != memory_id],
    )


def content_to_text(content: Any) -> str:
    """Extract displayable text from LangChain/Chainlit content shapes."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                value = part.get("text") or part.get("content") or ""
                parts.append(str(value))
            else:
                parts.append(content_to_text(part))
        return "".join(parts)

    return str(content)


def event_output_to_text(output: Any) -> str:
    """Extract assistant text from LangGraph event output shapes."""

    if output is None:
        return ""

    content = getattr(output, "content", None)
    if content is not None:
        return content_to_text(content)

    message = getattr(output, "message", None)
    if message is not None:
        return event_output_to_text(message)

    generations = getattr(output, "generations", None)
    if generations:
        first_generation = generations[0]
        if isinstance(first_generation, list) and first_generation:
            first_generation = first_generation[0]
        return event_output_to_text(first_generation)

    if isinstance(output, dict):
        for key in ("content", "text", "output"):
            if key in output:
                return content_to_text(output[key])

        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            return event_output_to_text(messages[-1])

        generations = output.get("generations")
        if isinstance(generations, list) and generations:
            first_generation = generations[0]
            if isinstance(first_generation, list) and first_generation:
                first_generation = first_generation[0]
            return event_output_to_text(first_generation)

    return ""


def format_memory_card(memory: dict[str, Any], index: int | None = None) -> str:
    """Format one memory dictionary as a Markdown card."""

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
    """Normalize the return value of a Chainlit ask prompt."""

    if not answer:
        return ""

    if isinstance(answer, dict):
        value = answer.get("output") or answer.get("content") or ""
        return str(value).strip()

    return str(answer).strip()
