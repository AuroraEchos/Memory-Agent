"""Consolidate durable conversation memories into long-term storage."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from memory_agent.context_builder import (
    build_conversation_transcript,
    content_to_text,
    truncate_text,
)
from memory_agent.llm import load_chat_model
from memory_agent.long_term_memory.prompts import MEMORY_EXTRACTION_PROMPT
from memory_agent.long_term_memory.retrieval import search_memory_namespaces
from memory_agent.long_term_memory.taxonomy import (
    MEMORY_SCHEMA_VERSION,
    MEMORY_TYPE_VALUES,
    MemoryType,
    memory_namespace,
    normalize_category,
    normalize_memory_type,
    normalize_str_list,
    taxonomy_prompt,
)


logger = logging.getLogger(__name__)

EXTRACTION_MESSAGE_WINDOW = 8
EXTRACTION_MAX_TOTAL_CHARS = 8_000
EXTRACTION_MAX_MESSAGE_CHARS = 2_500
MAX_EXISTING_MEMORY_FIELD_CHARS = 240
MAX_MEMORY_CONTENT_CHARS = 600
MAX_MEMORY_CONTEXT_CHARS = 300
MAX_MEMORY_SUBJECT_CHARS = 120
MAX_MEMORY_LIST_ITEM_CHARS = 80
MAX_MEMORY_LIST_ITEMS = 8


_SENSITIVE_TEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?im)^\s*[A-Z0-9_.-]*(?:API[_-]?KEY|SECRET|PASSWORD|PASSWD|"
        r"TOKEN|PRIVATE[_-]?KEY|ACCESS[_-]?KEY|REFRESH[_-]?TOKEN|"
        r"CLIENT[_-]?SECRET|AUTHORIZATION)[A-Z0-9_.-]*\s*[:=]\s*"
        r"['\"]?[^'\"\s#]{6,}"
    ),
    re.compile(
        r"(?i)\b(?:api[_-]?key|secret|password|passwd|token|private[_-]?key|"
        r"access[_-]?key|refresh[_-]?token|client[_-]?secret|authorization)"
        r"\b\s*[:=]\s*['\"]?[^'\"`\s]{6,}"
    ),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


class MemoryDecision(BaseModel):
    """Single create/update/ignore decision from the extraction model."""

    action: Literal["create", "update", "ignore"] = Field(
        description="Whether to create, update, or ignore memory."
    )
    memory_id: str | None = Field(
        default=None,
        description="Existing memory id if action is update.",
    )
    memory_type: MemoryType = Field(
        default="knowledge",
        description="Canonical long-term memory taxonomy type.",
    )
    content: str | None = None
    context: str | None = None
    category: str | None = None
    subject: str = ""
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    confidence: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def normalize_decision(cls, data: Any) -> Any:
        """Normalize permissive model outputs into the canonical shape."""

        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        if "memory_id" not in normalized and "id" in normalized:
            normalized["memory_id"] = normalized["id"]

        if "memory_type" not in normalized:
            for key in ("type", "kind", "taxonomy", "memory_kind"):
                if key in normalized:
                    normalized["memory_type"] = normalized[key]
                    break

        normalized["memory_type"] = normalize_memory_type(
            normalized.get("memory_type"),
            default="knowledge",
        )

        if "content" not in normalized:
            for key in ("memory", "fact", "preference", "summary"):
                if key in normalized:
                    normalized["content"] = normalized[key]
                    break

        if "action" not in normalized and "decision" in normalized:
            normalized["action"] = normalized["decision"]

        if "action" not in normalized:
            if normalized.get("memory_id"):
                normalized["action"] = "update"
            elif normalized.get("content"):
                normalized["action"] = "create"
            else:
                normalized["action"] = "ignore"

        action = normalized.get("action")
        if isinstance(action, str):
            action = action.strip().lower()
            if action in {"none", "no_action", "no-op", "noop", "skip"}:
                action = "ignore"
            normalized["action"] = action

        memory_type = normalize_memory_type(normalized.get("memory_type"))
        normalized["category"] = normalize_category(
            normalized.get("category"),
            memory_type=memory_type,
        )

        if normalized.get("confidence") is None:
            normalized["confidence"] = 1.0

        return normalized

    @field_validator("memory_type", mode="before")
    @classmethod
    def validate_memory_type(cls, value: object) -> MemoryType:
        """Accept known aliases but store only canonical taxonomy values."""

        return normalize_memory_type(value)

    @field_validator("category", mode="after")
    @classmethod
    def validate_category(cls, value: str | None, info: Any) -> str:
        """Ensure category is specific and never the old general bucket."""

        memory_type = normalize_memory_type(info.data.get("memory_type", "knowledge"))
        category = normalize_category(value, memory_type=memory_type)
        if category == "general":
            return normalize_category(None, memory_type=memory_type)
        return category

    @field_validator("entities", "topics", mode="before")
    @classmethod
    def validate_str_list(cls, value: object) -> list[str]:
        """Normalize entities/topics into short string lists."""

        return normalize_str_list(value)

    @field_validator("confidence", mode="after")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        """Keep model confidence in a stable numeric range."""

        return max(0.0, min(float(value), 1.0))


class MemoryExtractionResult(BaseModel):
    """Structured output wrapper for memory extraction decisions."""

    memories: list[MemoryDecision]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_memory_decision(cls, data: Any) -> Any:
        """Accept common list or single-object variants from the model."""

        if isinstance(data, list):
            return {"memories": data}

        if isinstance(data, dict) and "memories" not in data:
            for key in ("decisions", "memory_decisions", "items", "results"):
                if key in data:
                    return {"memories": data[key]}

            if any(
                key in data
                for key in (
                    "action",
                    "decision",
                    "id",
                    "memory_id",
                    "memory_type",
                    "content",
                    "memory",
                    "fact",
                    "preference",
                )
            ):
                return {"memories": [data]}

        return data


def _contains_sensitive_text(text: str | None) -> bool:
    """Return whether text looks like a secret or credential."""

    if not text:
        return False

    return any(pattern.search(text) for pattern in _SENSITIVE_TEXT_PATTERNS)


def _decision_contains_sensitive_text(decision: MemoryDecision) -> bool:
    """Return whether an extracted memory decision contains sensitive text."""

    return (
        _contains_sensitive_text(decision.content)
        or _contains_sensitive_text(decision.context)
        or _contains_sensitive_text(decision.subject)
        or any(_contains_sensitive_text(item) for item in decision.entities)
        or any(_contains_sensitive_text(item) for item in decision.topics)
    )


def _normalize_memory_text(text: str | None, max_chars: int) -> str:
    """Normalize and truncate memory text fields before storage or prompting."""

    return truncate_text(str(text or "").strip(), max_chars)


def _normalize_memory_list(values: object) -> list[str]:
    """Normalize, deduplicate, and clamp memory list-like fields."""

    normalized_values = normalize_str_list(values)
    result: list[str] = []
    seen: set[str] = set()

    for value in normalized_values:
        normalized = _normalize_memory_text(value, MAX_MEMORY_LIST_ITEM_CHARS)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        result.append(normalized)

        if len(result) >= MAX_MEMORY_LIST_ITEMS:
            break

    return result


def _format_existing_memories(memories: list[Any]) -> str:
    """Format existing memories for the extractor while filtering secrets."""

    if not memories:
        return "No existing memories."

    lines: list[str] = []

    for mem in memories:
        value = getattr(mem, "value", {}) or {}
        content = _normalize_memory_text(
            value.get("content", ""),
            MAX_EXISTING_MEMORY_FIELD_CHARS,
        )
        context = _normalize_memory_text(
            value.get("context", ""),
            MAX_EXISTING_MEMORY_FIELD_CHARS,
        )
        subject = _normalize_memory_text(
            value.get("subject", ""),
            MAX_EXISTING_MEMORY_FIELD_CHARS,
        )
        memory_type = normalize_memory_type(value.get("memory_type"))
        category = normalize_category(value.get("category"), memory_type=memory_type)

        if (
            _contains_sensitive_text(content)
            or _contains_sensitive_text(context)
            or _contains_sensitive_text(subject)
        ):
            continue

        lines.append(
            f"- id={mem.key}; "
            f"type={memory_type}; "
            f"category={category}; "
            f"subject={subject}; "
            f"content={content}; "
            f"context={context}"
        )

    return "\n".join(lines) if lines else "No existing memories."


async def _search_existing_memories(
    *,
    store: Any,
    user_id: str,
    query: str,
    per_type_limit: int = 4,
) -> list[Any]:
    """Search across taxonomy namespaces for possible updates/duplicates."""

    namespace_limits = [
        (memory_namespace(user_id, memory_type), per_type_limit)
        for memory_type in MEMORY_TYPE_VALUES
    ]
    found = await search_memory_namespaces(
        store=store,
        namespace_limits=namespace_limits,
        query=query,
    )

    found.sort(
        key=lambda mem: (
            getattr(mem, "score", None)
            if getattr(mem, "score", None) is not None
            else -1.0
        ),
        reverse=True,
    )
    return found[:20]


async def consolidate_memories(
    *,
    messages: list[Any],
    user_id: str,
    store: Any,
    model_name: str,
    thread_id: str | None = None,
    debug: bool = False,
    message_window: int = 20,
    message_char_window: int = 16000,
    message_max_chars: int = 4000,
) -> list[MemoryDecision]:
    """Extract, reconcile, and store durable memories from recent conversation."""

    latest_user_message = next(
        (
            m
            for m in reversed(messages)
            if str(getattr(m, "type", "")).lower() == "human"
        ),
        None,
    )

    user_text = content_to_text(getattr(latest_user_message, "content", "")).strip()

    if not user_text.strip():
        return []

    if _contains_sensitive_text(user_text):
        logger.warning(
            "Skipping memory consolidation for sensitive-looking user message"
        )
        if debug:
            print("\n=== Memory Consolidator skipped sensitive-looking input ===")
        return []

    extraction_window = max(2, min(int(message_window), EXTRACTION_MESSAGE_WINDOW))
    extraction_char_window = min(
        max(1, int(message_char_window)),
        EXTRACTION_MAX_TOTAL_CHARS,
    )
    extraction_message_chars = min(
        max(1, int(message_max_chars)),
        EXTRACTION_MAX_MESSAGE_CHARS,
    )
    latest_user_message_text = truncate_text(user_text, extraction_message_chars)
    conversation_transcript = build_conversation_transcript(
        messages,
        extraction_window,
        max_total_chars=extraction_char_window,
        max_message_chars=extraction_message_chars,
    )

    existing = await _search_existing_memories(
        store=store,
        user_id=user_id,
        query=latest_user_message_text,
    )
    existing_by_id = {
        str(getattr(memory, "key", "")): memory
        for memory in existing
        if getattr(memory, "key", None)
    }

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        memory_taxonomy=taxonomy_prompt(),
        existing_memories=_format_existing_memories(existing),
        conversation_transcript=conversation_transcript
        or "No recent conversation context.",
        latest_user_message=latest_user_message_text,
    )

    llm = load_chat_model(model_name, streaming=False)

    extractor = llm.with_structured_output(
        MemoryExtractionResult,
        include_raw=True,
    )
    raw_result = await extractor.ainvoke(
        prompt,
        config={
            "callbacks": [],
            "tags": ["memory_consolidator"],
        },
    )

    if isinstance(raw_result, MemoryExtractionResult):
        result = raw_result
    elif isinstance(raw_result, dict):
        parsing_error = raw_result.get("parsing_error")
        if parsing_error is not None:
            logger.warning("Memory extraction parsing failed: %s", parsing_error)
            if debug:
                print("\n=== Memory Consolidator Parse Error ===")
                print(parsing_error)
            return []

        parsed = raw_result.get("parsed")
        if parsed is None:
            logger.warning("Memory extraction returned no parsed result")
            return []

        try:
            result = (
                parsed
                if isinstance(parsed, MemoryExtractionResult)
                else MemoryExtractionResult.model_validate(parsed)
            )
        except Exception as exc:
            logger.warning(
                "Memory extraction parsed payload failed validation: %s",
                exc,
            )
            if debug:
                print("\n=== Memory Consolidator Validation Error ===")
                print(exc)
            return []
    else:
        logger.warning(
            "Memory extraction returned unsupported result type: %s",
            type(raw_result).__name__,
        )
        return []

    saved: list[MemoryDecision] = []

    for decision in result.memories:
        if decision.action == "ignore":
            continue

        if not decision.content:
            continue

        if _decision_contains_sensitive_text(decision):
            logger.warning(
                "Skipping sensitive-looking memory consolidation decision"
            )
            if debug:
                print("\n=== Memory Consolidator skipped sensitive-looking decision ===")
            continue

        existing_memory = None
        if decision.action == "update" and decision.memory_id:
            existing_memory = existing_by_id.get(decision.memory_id)

        if decision.action == "update" and existing_memory is None:
            logger.warning(
                "Memory extraction requested update for unknown memory_id=%r; "
                "saving as create instead.",
                decision.memory_id,
            )

        memory_type = normalize_memory_type(decision.memory_type)
        if existing_memory is not None:
            existing_value = getattr(existing_memory, "value", {}) or {}
            memory_type = normalize_memory_type(
                existing_value.get("memory_type"),
                default=memory_type,
            )

        category = normalize_category(decision.category, memory_type=memory_type)

        mem_id = existing_memory.key if existing_memory is not None else str(uuid.uuid4())
        content = _normalize_memory_text(decision.content, MAX_MEMORY_CONTENT_CHARS)
        context = _normalize_memory_text(decision.context, MAX_MEMORY_CONTEXT_CHARS)
        subject = _normalize_memory_text(decision.subject, MAX_MEMORY_SUBJECT_CHARS)
        entities = _normalize_memory_list(decision.entities)
        topics = _normalize_memory_list(decision.topics)

        if not content:
            continue

        now = datetime.now(timezone.utc).isoformat()
        value = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memory_type": memory_type,
            "content": content,
            "context": context,
            "category": category,
            "subject": subject,
            "entities": entities,
            "topics": topics,
            "confidence": decision.confidence,
            "source": {
                "type": "conversation",
                "thread_id": thread_id or "",
            },
            "updated_at": now,
        }

        await store.aput(
            memory_namespace(user_id, memory_type),
            key=mem_id,
            value=value,
        )

        saved.append(decision)

    if debug:
        print("\n=== Memory Consolidator Decisions ===")
        for item in result.memories:
            print(item.model_dump())

    return saved
