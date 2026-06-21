"""Build short-term conversation context for LLM calls."""

from __future__ import annotations

from typing import Any


def _content_to_text(content: Any) -> str:
    """Convert message content variants into plain text for filtering."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_content_to_text(part) for part in content)

    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")

    return str(content)


def _has_displayable_content(message: Any) -> bool:
    """Return whether a message has non-empty text-like content."""

    return bool(_content_to_text(getattr(message, "content", "")).strip())


def _role(message: Any) -> str:
    """Return the normalized LangChain message role."""

    return str(getattr(message, "type", "") or "").lower()


def _conversation_segments(messages: list[Any]) -> list[list[Any]]:
    """Group messages into human-started conversation segments."""

    segments: list[list[Any]] = []
    current: list[Any] | None = None

    for message in messages:
        if _role(message) == "human":
            if current:
                segments.append(current)
            current = [message]
            continue

        if current is not None:
            current.append(message)

    if current:
        segments.append(current)

    return segments


def build_context_messages(messages: list[Any], message_window: int) -> list[Any]:
    """Build a compact short-term context from recent complete turns.

    The latest human-started segment is always retained so the current user
    message cannot be trimmed away. Older context is added only as complete
    human-started turns that fit within the message-window budget.
    """

    window = max(1, int(message_window))
    filtered = [message for message in messages if _has_displayable_content(message)]
    segments = _conversation_segments(filtered)

    if not segments:
        return []

    selected: list[list[Any]] = [segments[-1]]
    remaining = window - len(segments[-1])

    for segment in reversed(segments[:-1]):
        if remaining <= 0:
            break
        if len(segment) < 2:
            continue
        if len(segment) > remaining:
            continue

        selected.insert(0, segment)
        remaining -= len(segment)

    return [message for segment in selected for message in segment]
