import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from memory_agent.llm import load_chat_model


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

            if "action" in data or "decision" in data:
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

    prompt = f"""You are a memory extraction model.

Your job is to decide whether the latest user message contains durable information
that should be stored as long-term memory.

Store only:
- stable personal facts
- long-term preferences
- recurring instructions
- durable goals
- corrections to existing memory

Do not store:
- secrets
- API keys
- passwords
- temporary requests
- one-off tasks
- highly sensitive information

If new information conflicts with an existing memory, return action="update"
and use that existing memory_id.

Existing memories:
{_format_existing_memories(existing)}

Latest user message:
{user_text}

Return a JSON object using exactly this shape:
{{
  "memories": [
    {{
      "action": "create|update|ignore",
      "memory_id": "existing memory id when action is update, otherwise null",
      "content": "concise durable memory, or null when action is ignore",
      "context": "brief background, or empty string",
      "category": "general",
      "confidence": 1.0
    }}
  ]
}}

If there is no durable information to store, return one item with action="ignore".
"""

    llm = load_chat_model(model_name)

    extractor = llm.with_structured_output(MemoryExtractionResult)
    result: MemoryExtractionResult = await extractor.ainvoke(
        prompt,
        config={
            "callbacks": [],
            "tags": ["memory_extractor"],
        },
    )

    saved: list[MemoryDecision] = []

    for decision in result.memories:
        if decision.action == "ignore":
            continue

        if not decision.content:
            continue

        mem_id = decision.memory_id if decision.action == "update" else str(uuid.uuid4())

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
