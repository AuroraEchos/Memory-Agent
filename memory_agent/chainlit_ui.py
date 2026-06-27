"""Chainlit UI helpers for sessions and content extraction."""

from typing import Any
from uuid import uuid4

import chainlit as cl


ASSISTANT_AUTHOR = "Memory Agent"


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
