"""Retrieve and format long-term memories for chat model calls."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from memory_agent.context_builder import content_to_text, truncate_text
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

MAX_MEMORY_QUERY_CHARS = 4_000
MAX_FORMATTED_CONTENT_CHARS = 320
MAX_FORMATTED_CONTEXT_CHARS = 180
MAX_FORMATTED_SUBJECT_CHARS = 120
MAX_FORMATTED_LIST_CHARS = 120
MAX_FORMATTED_MEMORY_LINES = MAX_INJECTED_MEMORIES


def build_memory_query(
    messages: list[Any],
    human_message_limit: int = 2,
    *,
    max_chars: int = MAX_MEMORY_QUERY_CHARS,
) -> str:
    """Build a retrieval query from the latest non-empty human messages."""

    chunks: list[str] = []
    limit = max(1, int(human_message_limit))

    for message in reversed(messages):
        if getattr(message, "type", None) != "human":
            continue

        text = content_to_text(getattr(message, "content", "")).strip()
        if text:
            chunks.append(text)

        if len(chunks) >= limit:
            break

    return truncate_text("\n".join(reversed(chunks)), max_chars)


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
        return truncate_text(items, MAX_FORMATTED_LIST_CHARS)

    if isinstance(items, (list, tuple, set)):
        return truncate_text(
            ", ".join(str(item) for item in items if str(item).strip()),
            MAX_FORMATTED_LIST_CHARS,
        )

    return truncate_text(str(items), MAX_FORMATTED_LIST_CHARS)


def _dedupe_key(memory: Any) -> str:
    """Return a stable dedupe key for prompt injection."""

    value = getattr(memory, "value", {}) or {}
    memory_type = normalize_memory_type(value.get("memory_type"))
    subject = str(value.get("subject", "")).strip().lower()
    content = str(value.get("content", "")).strip().lower()
    key = str(getattr(memory, "key", "")).strip()
    return key or f"{memory_type}::{subject}::{content}"


async def _build_shared_query_vector(
    store: MemoryStore,
    query: str,
) -> list[float] | None:
    """Compute one shared query embedding when the store exposes it."""

    embedding_provider = getattr(store, "embedding_provider", None)
    aembed = getattr(embedding_provider, "aembed", None)

    if not callable(aembed):
        return None

    try:
        return await aembed(query)
    except Exception:
        logger.exception(
            "Shared memory query embedding failed; falling back to per-search embedding"
        )
        return None


async def search_memory_namespaces(
    *,
    store: MemoryStore,
    namespace_limits: list[tuple[tuple[str, ...], int]],
    query: str,
    metadata_filter: dict[str, Any] | None = None,
) -> list[MemoryItem]:
    """Search multiple namespaces with one shared embedding when possible."""

    if not query.strip():
        return []

    query_vector = await _build_shared_query_vector(store, query)
    tasks = [
        store.asearch(
            namespace,
            query=query,
            limit=limit,
            metadata_filter=metadata_filter,
            query_vector=query_vector,
        )
        for namespace, limit in namespace_limits
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    memories: list[MemoryItem] = []

    for (namespace, _limit), result in zip(namespace_limits, results):
        if isinstance(result, Exception):
            logger.error(
                "Memory retrieval failed for namespace=%s",
                "/".join(namespace),
                exc_info=(type(result), result, result.__traceback__),
            )
            continue

        memories.extend(result)

    return memories


def format_memories(memories: list[Any]) -> str:
    """Format retrieved memories for injection into the system prompt."""

    if not memories:
        return "No relevant memories found."

    grouped: dict[str, list[Any]] = defaultdict(list)
    for memory in memories[:MAX_FORMATTED_MEMORY_LINES]:
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
            subject = truncate_text(
                str(value.get("subject", "")).strip(),
                MAX_FORMATTED_SUBJECT_CHARS,
            )
            entities = _format_list(value.get("entities"))
            topics = _format_list(value.get("topics"))
            content = truncate_text(
                str(value.get("content", "")),
                MAX_FORMATTED_CONTENT_CHARS,
            )
            context = truncate_text(
                str(value.get("context", "")),
                MAX_FORMATTED_CONTEXT_CHARS,
            )

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
                + f"; content={content}; "
                + f"context={context}"
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

    namespace_limits = [
        (
            memory_namespace(user_id, memory_type),
            MEMORY_RETRIEVAL_LIMITS[memory_type],
        )
        for memory_type in MEMORY_TYPE_VALUES
    ]
    memories = await search_memory_namespaces(
        store=store,
        namespace_limits=namespace_limits,
        query=query,
    )

    memories.sort(
        key=lambda memory: (
            -_memory_score(memory),
            memory_type_rank((getattr(memory, "value", {}) or {}).get("memory_type")),
        )
    )

    deduped: list[MemoryItem] = []
    seen: set[str] = set()

    for memory in memories:
        dedupe_key = _dedupe_key(memory)
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        deduped.append(memory)

        if len(deduped) >= MAX_INJECTED_MEMORIES:
            break

    return deduped
