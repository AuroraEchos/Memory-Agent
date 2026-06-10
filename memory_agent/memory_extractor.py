import logging
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from memory_agent.llm import load_chat_model
from memory_agent.prompts import MEMORY_EXTRACTION_PROMPT


logger = logging.getLogger(__name__)


class MemoryDecision(BaseModel):
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
    memories: list[MemoryDecision]

    @model_validator(mode="before")
    @classmethod
    def wrap_single_memory_decision(cls, data: Any) -> Any:
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
    if not memories:
        return "No existing memories."

    return "\n".join(
        f"- id={mem.key}; "
        f"category={mem.value.get('category', 'general')}; "
        f"content={mem.value.get('content', '')}; "
        f"context={mem.value.get('context', '')}"
        for mem in memories
    )


async def extract_and_store_memories(
    *,
    messages: list[Any],
    user_id: str,
    store: Any,
    model_name: str,
    debug: bool = False,
) -> list[MemoryDecision]:
    latest_user_messages = [
        getattr(m, "content", "")
        for m in messages
        if getattr(m, "type", None) == "human"
    ][-3:]

    user_text = "\n".join(str(x) for x in latest_user_messages)

    if not user_text.strip():
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

        mem_id = decision.memory_id if decision.action == "update" else None
        mem_id = mem_id or str(uuid.uuid4())

        value = {
            "content": decision.content,
            "context": decision.context or "",
            "category": decision.category,
            "confidence": max(0.0, min(float(decision.confidence), 1.0)),
            "updated_at": datetime.now().isoformat(),
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
