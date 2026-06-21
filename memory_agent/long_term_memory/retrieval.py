"""Retrieve and format long-term memories for chat model calls."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from memory_agent.long_term_memory.store.base import MemoryItem, MemoryStore
from memory_agent.long_term_memory.taxonomy import (
    MAX_INJECTED_MEMORIES,
    MEMORY_RETRIEVAL_LIMITS,
    MEMORY_TYPE_VALUES,
    memory_namespace,
    memory_type_rank,
    normalize_category,
    normalize_memory_type,
)


logger = logging.getLogger(__name__)


def _content_to_text(content: Any) -> str:
    """Convert message content variants into plain text for retrieval."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_content_to_text(part) for part in content)

    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or content)

    return str(content)


def build_memory_query(messages: list[Any], human_message_limit: int = 2) -> str:
    """Build a retrieval query from the latest non-empty human messages."""

    chunks: list[str] = []
    limit = max(1, int(human_message_limit))

    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue

        text = _content_to_text(getattr(message, "content", "")).strip()
        if text:
            chunks.append(text)

        if len(chunks) >= limit:
            break

    return "\n".join(reversed(chunks))


def _memory_score(memory: Any) -> float:
    """Return a sortable score for a retrieved memory."""

    score = getattr(memory, "score", None)
    if score is None:
        return -1.0

    try:
        return float(score)
    except (TypeError, ValueError):
        return -1.0


def serialize_memory_hit(memory: Any) -> dict[str, Any]:
    """Serialize a memory hit for Chainlit session display."""

    return {
        "key": str(getattr(memory, "key", "")),
        "value": getattr(memory, "value", {}) or {},
        "score": getattr(memory, "score", None),
    }


def _format_list(items: Any) -> str:
    """Format a small list-like metadata value for prompt injection."""

    if not items:
        return ""

    if isinstance(items, str):
        return items

    if isinstance(items, (list, tuple, set)):
        return ", ".join(str(item) for item in items if str(item).strip())

    return str(items)


def format_memories(memories: list[Any]) -> str:
    """Format retrieved memories for injection into the system prompt."""

    if not memories:
        return "No relevant memories found."

    grouped: dict[str, list[Any]] = defaultdict(list)
    for memory in memories:
        value = getattr(memory, "value", {}) or {}
        memory_type = normalize_memory_type(value.get("memory_type"))
        grouped[memory_type].append(memory)

    lines: list[str] = ["<memories>"]

    for memory_type in MEMORY_TYPE_VALUES:
        type_memories = grouped.get(memory_type)
        if not type_memories:
            continue

        lines.append(f"[{memory_type}]")
        for memory in type_memories:
            value = getattr(memory, "value", {}) or {}
            category = normalize_category(value.get("category"), memory_type=memory_type)
            subject = str(value.get("subject", "")).strip()
            entities = _format_list(value.get("entities"))
            topics = _format_list(value.get("topics"))

            metadata_parts = [
                f"id={memory.key}",
                f"type={memory_type}",
                f"category={category}",
                f"confidence={value.get('confidence', 1.0)}",
            ]
            if subject:
                metadata_parts.append(f"subject={subject}")
            if entities:
                metadata_parts.append(f"entities={entities}")
            if topics:
                metadata_parts.append(f"topics={topics}")

            lines.append(
                "- "
                + "; ".join(metadata_parts)
                + f"; content={value.get('content', '')}; "
                + f"context={value.get('context', '')}"
            )

    lines.append("</memories>")
    return "\n".join(lines)


async def retrieve_relevant_memories(
    *,
    store: MemoryStore,
    user_id: str,
    query: str,
) -> list[MemoryItem]:
    """Retrieve memories from each taxonomy namespace using fixed budgets."""

    if not query.strip():
        return []

    memories: list[MemoryItem] = []

    for memory_type in MEMORY_TYPE_VALUES:
        try:
            memories.extend(
                await store.asearch(
                    memory_namespace(user_id, memory_type),
                    query=query,
                    limit=MEMORY_RETRIEVAL_LIMITS[memory_type],
                )
            )
        except Exception:
            logger.exception("Memory retrieval failed for type=%s", memory_type)

    memories.sort(
        key=lambda memory: (
            memory_type_rank((getattr(memory, "value", {}) or {}).get("memory_type")),
            -_memory_score(memory),
        )
    )
    return memories[:MAX_INJECTED_MEMORIES]
