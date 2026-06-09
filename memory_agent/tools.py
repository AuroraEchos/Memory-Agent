import uuid
from datetime import datetime
from typing import Any

from typing_extensions import Annotated
from langchain_core.tools import InjectedToolArg


async def upsert_memory(
    content: str,
    context: str,
    *,
    category: str = "general",
    confidence: float = 1.0,
    memory_id: uuid.UUID | str | None = None,
    user_id: Annotated[str, InjectedToolArg],
    store: Annotated[Any, InjectedToolArg],
) -> str:
    """Create or update a long-term memory segment for the current user.

    Args:
        content: The core fact, knowledge, user preference, or project insight to remember. Be concise and factual.
        context: The background context or historical situation in which this memory was generated (e.g., specific error messages or project names).
        category: Category tag to classify the memory (e.g., 'user_preference', 'coding_error', 'project_milestone'). Defaults to 'general'.
        confidence: A float between 0.0 and 1.0 representing how reliable or verified this information is.
        memory_id: Use only when updating an existing memory to overwrite it. Leave empty for new memories.
    """

    mem_id = str(memory_id or uuid.uuid4())

    confidence = max(0.0, min(float(confidence), 1.0))

    await store.aput(
        ("memories", user_id),
        key=mem_id,
        value={
            "content": content,
            "context": context,
            "category": category,
            "confidence": confidence,
            "updated_at": datetime.now().isoformat(),
        },
    )

    return f"Stored memory {mem_id}"