"""Durable memory extraction and safe persistence helpers."""

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from memory_agent.llm import load_chat_model
from memory_agent.prompts import MEMORY_EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


# Keep code-level secret filtering here instead of relying only on the
# extraction prompt. Anything saved to long-term memory can be retrieved later,
# injected into prompts, and displayed in the UI, so sensitive-looking content
# must be rejected before and after model extraction.
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


def _contains_sensitive_text(text: str | None) -> bool:
    """Return whether text looks like a secret or credential."""

    if not text:
        return False

    return any(pattern.search(text) for pattern in _SENSITIVE_TEXT_PATTERNS)


def _decision_contains_sensitive_text(decision: "MemoryDecision") -> bool:
    """Return whether an extracted memory decision contains sensitive text."""

    return (
        _contains_sensitive_text(decision.content)
        or _contains_sensitive_text(decision.context)
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
    content: str | None = None
    context: str | None = None
    category: str = "general"
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

        if "content" not in normalized:
            for key in ("memory", "fact", "preference"):
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

        if normalized.get("category") is None:
            normalized["category"] = "general"

        if normalized.get("confidence") is None:
            normalized["confidence"] = 1.0

        return normalized


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
                    "content",
                    "memory",
                    "fact",
                    "preference",
                )
            ):
                return {"memories": [data]}

        return data


def _format_existing_memories(memories: list[Any]) -> str:
    """Format existing memories for the extractor while filtering secrets."""

    if not memories:
        return "No existing memories."

    lines: list[str] = []

    for mem in memories:
        value = mem.value
        content = str(value.get("content", ""))
        context = str(value.get("context", ""))

        # Never feed sensitive-looking stored data back into the extractor.
        # This is a defensive layer for legacy/bad data that may already exist.
        if _contains_sensitive_text(content) or _contains_sensitive_text(context):
            continue

        lines.append(
            f"- id={mem.key}; "
            f"category={value.get('category', 'general')}; "
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


async def extract_and_store_memories(
    *,
    messages: list[Any],
    user_id: str,
    store: Any,
    model_name: str,
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

    existing = await store.asearch(
        ("memories", user_id),
        query=user_text,
        limit=10,
    )

    prompt = MEMORY_EXTRACTION_PROMPT.format(
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

        mem_id = decision.memory_id if decision.action == "update" else None
        mem_id = mem_id or str(uuid.uuid4())

        value = {
            "content": decision.content,
            "context": decision.context or "",
            "category": decision.category,
            "confidence": max(0.0, min(float(decision.confidence), 1.0)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        await store.aput(
            ("memories", user_id),
            key=mem_id,
            value=value,
        )

        saved.append(decision)

    if debug:
        print("\n=== Memory Extractor Decisions ===")
        for item in result.memories:
            print(item.model_dump())

    return saved
