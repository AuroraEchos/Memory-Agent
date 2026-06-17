"""Canonical long-term memory taxonomy and routing helpers.

The project is still in development, so the taxonomy is intentionally strict:
new memories must be written into one of these namespaces instead of a single
catch-all bucket.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


MemoryType = Literal[
    "persona",
    "entity",
    "project",
    "task",
    "episodic",
    "procedural",
    "knowledge",
]

MEMORY_SCHEMA_VERSION = 1

MEMORY_TYPE_VALUES: tuple[MemoryType, ...] = (
    "persona",
    "entity",
    "project",
    "task",
    "episodic",
    "procedural",
    "knowledge",
)

MEMORY_TYPE_DESCRIPTIONS: dict[MemoryType, str] = {
    "persona": "User identity, stable preferences, communication style, recurring instructions.",
    "entity": "Facts about people, organizations, products, places, accounts, or named concepts.",
    "project": "Durable project context, requirements, constraints, decisions, architecture, or domain assumptions.",
    "task": "Longer-lived goals, open tasks, commitments, milestones, or task-specific constraints.",
    "episodic": "Summaries of prior conversations, interaction outcomes, events, and user feedback.",
    "procedural": "Reusable workflows, tool-use habits, SOPs, preferred procedures, or automation patterns.",
    "knowledge": "Reusable domain knowledge or reference facts that are not primarily about the user or a named entity.",
}

MEMORY_CATEGORY_HINTS: dict[MemoryType, tuple[str, ...]] = {
    "persona": (
        "preference",
        "profile",
        "communication_style",
        "recurring_instruction",
    ),
    "entity": (
        "person",
        "organization",
        "product",
        "place",
        "account",
        "concept",
        "fact",
    ),
    "project": (
        "project_context",
        "requirement",
        "constraint",
        "decision",
        "architecture",
        "domain_assumption",
    ),
    "task": (
        "goal",
        "todo",
        "status",
        "milestone",
        "constraint",
        "commitment",
    ),
    "episodic": (
        "conversation_summary",
        "interaction",
        "event",
        "outcome",
        "feedback",
    ),
    "procedural": (
        "workflow",
        "tool_habit",
        "process",
        "sop",
        "automation_pattern",
    ),
    "knowledge": (
        "domain_fact",
        "reference",
        "concept",
        "document_fact",
    ),
}

# Retrieval is deterministic for now. Persona/procedural memories are usually
# compact and high-value, so they get small but consistent budget. Entity,
# project, and task memories get enough budget for specificity.
MEMORY_RETRIEVAL_LIMITS: dict[MemoryType, int] = {
    "persona": 4,
    "project": 4,
    "task": 3,
    "procedural": 3,
    "entity": 4,
    "episodic": 3,
    "knowledge": 3,
}

MAX_INJECTED_MEMORIES = 14


@dataclass(frozen=True)
class MemoryRoute:
    """Concrete storage route for one memory type."""

    memory_type: MemoryType
    namespace: tuple[str, ...]
    limit: int = 5


def normalize_memory_type(value: object, *, default: MemoryType = "knowledge") -> MemoryType:
    """Normalize a model- or UI-provided value to the canonical taxonomy."""

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "user": "persona",
            "user_preference": "persona",
            "preference": "persona",
            "profile": "persona",
            "entity_memory": "entity",
            "fact": "knowledge",
            "semantic": "knowledge",
            "summary": "episodic",
            "conversation": "episodic",
            "conversation_summary": "episodic",
            "workflow": "procedural",
            "toolbox": "procedural",
            "tool": "procedural",
            "tool_habit": "procedural",
            "goal": "task",
            "project_context": "project",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in MEMORY_TYPE_VALUES:
            return normalized  # type: ignore[return-value]

    return default


def normalize_category(value: object, *, memory_type: MemoryType) -> str:
    """Normalize a subtype/category string without collapsing it to general."""

    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized:
            return normalized

    return MEMORY_CATEGORY_HINTS[memory_type][0]


def normalize_str_list(value: object) -> list[str]:
    """Normalize model-provided entity/topic collections to a short string list."""

    if value is None:
        return []

    if isinstance(value, str):
        candidates = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(item).strip() for item in value]
    else:
        candidates = [str(value).strip()]

    seen: set[str] = set()
    result: list[str] = []
    for item in candidates:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= 12:
            break

    return result


def memory_namespace(user_id: str, memory_type: MemoryType | str) -> tuple[str, ...]:
    """Return the canonical Qdrant namespace for a user's memory type."""

    return ("memories", str(user_id), normalize_memory_type(memory_type))


def iter_memory_routes(user_id: str) -> list[MemoryRoute]:
    """Return all canonical memory namespaces in retrieval priority order."""

    return [
        MemoryRoute(
            memory_type=memory_type,
            namespace=memory_namespace(user_id, memory_type),
            limit=MEMORY_RETRIEVAL_LIMITS[memory_type],
        )
        for memory_type in MEMORY_TYPE_VALUES
    ]


def memory_type_rank(memory_type: object) -> int:
    """Return a stable display/retrieval order for a memory type."""

    normalized = normalize_memory_type(memory_type)
    try:
        return MEMORY_TYPE_VALUES.index(normalized)
    except ValueError:
        return len(MEMORY_TYPE_VALUES)


def taxonomy_prompt() -> str:
    """Return a compact taxonomy description for LLM prompts."""

    lines: list[str] = []
    for memory_type in MEMORY_TYPE_VALUES:
        categories = ", ".join(MEMORY_CATEGORY_HINTS[memory_type])
        lines.append(
            f"- {memory_type}: {MEMORY_TYPE_DESCRIPTIONS[memory_type]} "
            f"Suggested categories: {categories}."
        )
    return "\n".join(lines)
