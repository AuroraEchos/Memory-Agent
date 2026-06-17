"""Chainlit UI helpers for sessions, actions, and memory rendering."""

from typing import Any
from uuid import uuid4

import chainlit as cl


ASSISTANT_AUTHOR = "Memory Agent"
LAST_MEMORY_HITS_KEY = "last_memory_hits"


def _safe_int(value: Any) -> int | None:
    """Coerce token count values to non-negative integers when possible."""

    if value is None or isinstance(value, bool):
        return None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    if number < 0:
        return None

    return number


def _normalize_token_usage(value: Any) -> dict[str, int]:
    """Normalize provider-specific token usage payloads."""

    if not isinstance(value, dict):
        return {}

    usage = value
    for nested_key in ("token_usage", "usage", "usage_metadata"):
        nested = usage.get(nested_key)
        if isinstance(nested, dict):
            usage = nested
            break

    def first_count(*keys: str) -> int | None:
        for key in keys:
            if key not in usage:
                continue
            count = _safe_int(usage.get(key))
            if count is not None:
                return count
        return None

    input_tokens = first_count("input_tokens", "prompt_tokens", "prompt")
    output_tokens = first_count("output_tokens", "completion_tokens", "completion")
    total_tokens = first_count("total_tokens", "total")

    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    normalized: dict[str, int] = {}
    if input_tokens is not None:
        normalized["input_tokens"] = input_tokens
    if output_tokens is not None:
        normalized["output_tokens"] = output_tokens
    if total_tokens is not None:
        normalized["total_tokens"] = total_tokens

    return normalized


def extract_token_usage(output: Any) -> dict[str, int]:
    """Extract token usage from common LangChain event output shapes."""

    if output is None:
        return {}

    usage_metadata = getattr(output, "usage_metadata", None)
    usage = _normalize_token_usage(usage_metadata)
    if usage:
        return usage

    response_metadata = getattr(output, "response_metadata", None)
    usage = _normalize_token_usage(response_metadata)
    if usage:
        return usage

    message = getattr(output, "message", None)
    if message is not None:
        usage = extract_token_usage(message)
        if usage:
            return usage

    generations = getattr(output, "generations", None)
    if generations:
        first_generation = generations[0]
        if isinstance(first_generation, list) and first_generation:
            first_generation = first_generation[0]
        usage = extract_token_usage(first_generation)
        if usage:
            return usage

    if isinstance(output, dict):
        for key in ("usage_metadata", "response_metadata", "token_usage", "usage"):
            usage = _normalize_token_usage(output.get(key))
            if usage:
                return usage

        messages = output.get("messages")
        if isinstance(messages, list) and messages:
            usage = extract_token_usage(messages[-1])
            if usage:
                return usage

        generations = output.get("generations")
        if isinstance(generations, list) and generations:
            first_generation = generations[0]
            if isinstance(first_generation, list) and first_generation:
                first_generation = first_generation[0]
            usage = extract_token_usage(first_generation)
            if usage:
                return usage

    return {}


def format_token_usage(usage: dict[str, int]) -> str:
    """Format normalized token usage for display in Chainlit messages."""

    parts: list[str] = []

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    total_tokens = usage.get("total_tokens")

    if input_tokens is not None:
        parts.append(f"输入 `{input_tokens:,}`")
    if output_tokens is not None:
        parts.append(f"输出 `{output_tokens:,}`")
    if total_tokens is not None:
        parts.append(f"总计 `{total_tokens:,}`")

    if not parts:
        return ""

    return "**Token 用量**: " + " · ".join(parts)


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


def memory_actions(memory_id: str, memory_type: str | None) -> list[cl.Action]:
    """Build action buttons attached to a single memory card."""

    return [
        cl.Action(
            name="delete_memory",
            payload={"memory_id": memory_id, "memory_type": memory_type or ""},
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


def _format_metadata_list(value: Any) -> str:
    """Format list-like metadata values for Markdown cards."""

    if not value:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value if str(item).strip())

    return str(value)


def format_memory_card(memory: dict[str, Any], index: int | None = None) -> str:
    """Format one memory dictionary as a Markdown card."""

    value = memory["value"]
    score = memory.get("score")
    title = f"### 记忆 {index}" if index is not None else "### 记忆"
    score_line = f"\n**Score**: `{score:.3f}`" if score is not None else ""
    memory_type = value.get("memory_type", "unknown")
    category = value.get("category", "")
    subject = value.get("subject", "")
    entities = _format_metadata_list(value.get("entities"))
    topics = _format_metadata_list(value.get("topics"))

    metadata = (
        f"**ID**: `{memory['key']}`\n"
        f"**Type**: `{memory_type}`\n"
        f"**Category**: `{category}`\n"
        f"**Confidence**: `{value.get('confidence', 1.0)}`"
        f"{score_line}\n"
        f"**Updated**: `{value.get('updated_at', '')}`"
    )

    if subject:
        metadata += f"\n**Subject**: `{subject}`"
    if entities:
        metadata += f"\n**Entities**: `{entities}`"
    if topics:
        metadata += f"\n**Topics**: `{topics}`"

    return (
        f"{title}\n"
        f"{metadata}\n\n"
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
