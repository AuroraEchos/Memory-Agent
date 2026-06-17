"""Durable memory extraction, taxonomy classification, and safe persistence."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from memory_agent.llm import load_chat_model
from memory_agent.memory_taxonomy import (
    MEMORY_SCHEMA_VERSION,
    MEMORY_TYPE_VALUES,
    MemoryType,
    memory_namespace,
    normalize_category,
    normalize_memory_type,
    normalize_str_list,
    taxonomy_prompt,
)
from memory_agent.prompts import MEMORY_EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


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


def _format_existing_memories(memories: list[Any]) -> str:
    """Format existing memories for the extractor while filtering secrets."""

    if not memories:
        return "No existing memories."

    lines: list[str] = []

    for mem in memories:
        value = getattr(mem, "value", {}) or {}
        content = str(value.get("content", ""))
        context = str(value.get("context", ""))
        subject = str(value.get("subject", ""))
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


def _content_to_text(content: Any) -> str:
    """Convert message content variants into plain text."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or part.get("content") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)

    return str(content)


async def _search_existing_memories(
    *,
    store: Any,
    user_id: str,
    query: str,
    per_type_limit: int = 4,
) -> list[Any]:
    """Search across taxonomy namespaces for possible updates/duplicates."""

    found: list[Any] = []

    for memory_type in MEMORY_TYPE_VALUES:
        try:
            found.extend(
                await store.asearch(
                    memory_namespace(user_id, memory_type),
                    query=query,
                    limit=per_type_limit,
                )
            )
        except Exception:
            logger.exception(
                "Failed to search existing %s memories during extraction",
                memory_type,
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


async def extract_and_store_memories(
    *,
    messages: list[Any],
    user_id: str,
    store: Any,
    model_name: str,
    thread_id: str | None = None,
    debug: bool = False,
) -> list[MemoryDecision]:
    """Extract durable memories from the latest user message and store them."""

    latest_user_message = next(
        (
            m
            for m in reversed(messages)
            if getattr(m, "type", None) == "human"
        ),
        None,
    )

    user_text = _content_to_text(getattr(latest_user_message, "content", ""))

    if not user_text.strip():
        return []

    if _contains_sensitive_text(user_text):
        logger.warning(
            "Skipping memory extraction for sensitive-looking user message"
        )
        if debug:
            print("\n=== Memory Extractor skipped sensitive-looking input ===")
        return []

    existing = await _search_existing_memories(
        store=store,
        user_id=user_id,
        query=user_text,
    )

    prompt = MEMORY_EXTRACTION_PROMPT.format(
        memory_taxonomy=taxonomy_prompt(),
        existing_memories=_format_existing_memories(existing),
        user_text=user_text,
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
            "tags": ["memory_extractor"],
        },
    )

    if isinstance(raw_result, MemoryExtractionResult):
        result = raw_result
    elif isinstance(raw_result, dict):
        parsing_error = raw_result.get("parsing_error")
        if parsing_error is not None:
            logger.warning("Memory extraction parsing failed: %s", parsing_error)
            if debug:
                print("\n=== Memory Extractor Parse Error ===")
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
                print("\n=== Memory Extractor Validation Error ===")
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
                "Skipping sensitive-looking memory extraction decision"
            )
            if debug:
                print("\n=== Memory Extractor skipped sensitive-looking decision ===")
            continue

        memory_type = normalize_memory_type(decision.memory_type)
        category = normalize_category(decision.category, memory_type=memory_type)

        mem_id = decision.memory_id if decision.action == "update" else None
        mem_id = mem_id or str(uuid.uuid4())

        now = datetime.now(timezone.utc).isoformat()
        value = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "memory_type": memory_type,
            "content": decision.content,
            "context": decision.context or "",
            "category": category,
            "subject": decision.subject.strip(),
            "entities": decision.entities,
            "topics": decision.topics,
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
        print("\n=== Memory Extractor Decisions ===")
        for item in result.memories:
            print(item.model_dump())

    return saved
