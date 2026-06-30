"""Build short-term conversation context for LLM calls."""

from __future__ import annotations

from typing import Any


DEFAULT_CONTEXT_MAX_CHARS = 16_000
DEFAULT_MESSAGE_MAX_CHARS = 4_000
TRUNCATION_MARKER = "\n...[truncated]"


def content_to_text(content: Any) -> str:
    """Convert message content variants into plain text."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(content_to_text(part) for part in content)

    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")

    return str(content)


def truncate_text(text: str, max_chars: int) -> str:
    """Clamp one text block to a stable size budget."""

    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    if max_chars <= len(TRUNCATION_MARKER):
        return text[:max_chars]

    head = max_chars - len(TRUNCATION_MARKER)
    return text[:head].rstrip() + TRUNCATION_MARKER


def _message_content(message: Any) -> str:
    """Return one message's text content regardless of its concrete type."""

    if isinstance(message, dict):
        return content_to_text(message.get("content", ""))

    return content_to_text(getattr(message, "content", ""))


def _role(message: Any) -> str:
    """Return the normalized message role."""

    if isinstance(message, dict):
        raw_role = message.get("role") or message.get("type") or ""
    else:
        raw_role = getattr(message, "type", "") or getattr(message, "role", "") or ""

    normalized = str(raw_role).lower().strip()

    if normalized == "human":
        return "user"
    if normalized == "ai":
        return "assistant"
    return normalized


def _has_displayable_content(message: Any) -> bool:
    """Return whether a message has non-empty text-like content."""

    return bool(_message_content(message).strip())


def _conversation_segments(messages: list[Any]) -> list[list[Any]]:
    """Group messages into user-started conversation segments."""

    segments: list[list[Any]] = []
    current: list[Any] | None = None

    for message in messages:
        if _role(message) == "user":
            if current:
                segments.append(current)
            current = [message]
            continue

        if current is not None:
            current.append(message)

    if current:
        segments.append(current)

    return segments


def _segment_char_count(segment: list[Any], max_message_chars: int) -> int:
    """Return the normalized character footprint of one segment."""

    return sum(
        len(truncate_text(_message_content(message), max_message_chars))
        for message in segment
    )


def _to_llm_message(message: Any, max_message_chars: int) -> dict[str, str]:
    """Convert one normalized message into a role/content payload."""

    return {
        "role": _role(message),
        "content": truncate_text(_message_content(message), max_message_chars),
    }


def build_context_messages(
    messages: list[Any],
    message_window: int,
    *,
    max_total_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    max_message_chars: int = DEFAULT_MESSAGE_MAX_CHARS,
) -> list[dict[str, str]]:
    """Build compact LLM-ready context from recent complete turns.

    The latest user-started segment is always retained so the current user
    message cannot be trimmed away. Older context is added only as complete
    user-started turns that fit within both the message budget and the
    approximate character budget.
    """

    window = max(1, int(message_window))
    message_limit = max(1, int(max_message_chars))
    char_budget = int(max_total_chars)
    filtered = [message for message in messages if _has_displayable_content(message)]
    segments = _conversation_segments(filtered)

    if not segments:
        return []

    selected: list[list[Any]] = [segments[-1]]
    remaining_messages = window - len(segments[-1])
    remaining_chars: int | None = None

    if char_budget > 0:
        remaining_chars = char_budget - _segment_char_count(segments[-1], message_limit)

    for segment in reversed(segments[:-1]):
        if remaining_messages <= 0:
            break
        if len(segment) < 2:
            continue
        if len(segment) > remaining_messages:
            continue

        segment_chars = _segment_char_count(segment, message_limit)
        if remaining_chars is not None and segment_chars > remaining_chars:
            continue

        selected.insert(0, segment)
        remaining_messages -= len(segment)
        if remaining_chars is not None:
            remaining_chars -= segment_chars

    flattened = [message for segment in selected for message in segment]
    llm_messages = [
        _to_llm_message(message, message_limit)
        for message in flattened
    ]
    return [message for message in llm_messages if message["content"].strip()]


def build_conversation_transcript(
    messages: list[Any],
    message_window: int,
    *,
    max_total_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
    max_message_chars: int = DEFAULT_MESSAGE_MAX_CHARS,
) -> str:
    """Build a compact plain-text transcript for memory extraction prompts."""

    context_messages = build_context_messages(
        messages,
        message_window,
        max_total_chars=max_total_chars,
        max_message_chars=max_message_chars,
    )

    if not context_messages:
        return ""

    labels = {
        "user": "User",
        "assistant": "Assistant",
        "system": "System",
        "tool": "Tool",
    }
    lines = [
        f"{labels.get(message['role'], message['role'].title())}: {message['content']}"
        for message in context_messages
    ]
    return "\n\n".join(lines)
