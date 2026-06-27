"""Chainlit UI helpers for sessions, actions, and memory rendering."""

import math
from typing import Any
from uuid import uuid4

import chainlit as cl


ASSISTANT_AUTHOR = "Memory Agent"


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


def format_response_latency(seconds: float | None) -> str:
    """Format LLM first-token latency for display in Chainlit messages."""

    if seconds is None:
        return ""

    try:
        elapsed = float(seconds)
    except (TypeError, ValueError):
        return ""

    if not math.isfinite(elapsed) or elapsed < 0:
        return ""

    if elapsed < 1:
        value = f"{elapsed * 1000:.0f} ms"
    else:
        value = f"{elapsed:.2f} s"

    return f"**LLM 首响耗时**: `{value}`"


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
