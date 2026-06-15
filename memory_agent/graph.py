"""LangGraph workflow nodes for chat response and memory extraction."""

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from memory_agent.config import Context
from memory_agent.llm import load_chat_model
from memory_agent.memory_extractor import extract_and_store_memories
from memory_agent.state import State


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


def _latest_human_text(messages: list[Any], n: int = 2) -> str:
    """Return the latest non-empty human messages as retrieval query text."""

    chunks: list[str] = []

    for msg in reversed(messages):
        if getattr(msg, "type", None) != "human":
            continue

        text = _content_to_text(getattr(msg, "content", "")).strip()
        if text:
            chunks.append(text)

        if len(chunks) >= n:
            break

    return "\n".join(reversed(chunks))


def _recent_messages(messages: list[Any], window: int) -> list[Any]:
    """Keep only the recent conversation window sent to the chat model."""

    window = max(1, int(window))
    return messages[-window:]


def _serialize_memory_hit(mem: Any) -> dict[str, Any]:
    """Serialize a memory hit for Chainlit session display."""

    return {
        "key": str(getattr(mem, "key", "")),
        "value": getattr(mem, "value", {}) or {},
        "score": getattr(mem, "score", None),
    }


def _format_memories(memories: list[Any]) -> str:
    """Format retrieved memories for injection into the system prompt."""

    if not memories:
        return "No relevant memories found."

    lines: list[str] = []

    for mem in memories:
        value = mem.value

        lines.append(
            "- "
            f"id={mem.key}; "
            f"category={value.get('category', 'general')}; "
            f"confidence={value.get('confidence', 1.0)}; "
            f"content={value.get('content', '')}; "
            f"context={value.get('context', '')}"
        )

    return "<memories>\n" + "\n".join(lines) + "\n</memories>"


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Retrieve relevant memories, call the chat model, and return hits."""

    if runtime.store is None:
        raise RuntimeError("runtime.store is None. Compile the graph with a store.")

    user_id = runtime.context.user_id
    query = _latest_human_text(state.messages, n=2)

    try:
        memories = await runtime.store.asearch(
            ("memories", user_id),
            query=query,
            limit=5,
        )
    except Exception as exc:
        logger.exception("Memory retrieval failed")
        if runtime.context.debug:
            print("\n=== Memory Retrieval Error ===")
            print(f"{type(exc).__name__}: {exc}")
        memories = []

    if runtime.context.debug:
        print("\n=== Retrieved Memories ===")
        for mem in memories:
            print(mem.key, mem.value, "score=", mem.score)

    system_message = runtime.context.system_prompt.format(
        user_info=_format_memories(memories),
        time=datetime.now(timezone.utc).isoformat(),
    )

    llm = load_chat_model(runtime.context.model)
    recent_messages = _recent_messages(
        state.messages,
        runtime.context.message_window,
    )

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_message},
            *recent_messages,
        ]
    )

    return {
        "messages": [response],
        "memory_hits": [_serialize_memory_hit(mem) for mem in memories],
    }


async def extract_memory(state: State, runtime: Runtime[Context]) -> dict:
    """Extract durable memories after a chat response has been generated."""

    if runtime.store is None:
        raise RuntimeError("runtime.store is None. Compile the graph with a store.")

    try:
        await extract_and_store_memories(
            messages=state.messages,
            user_id=runtime.context.user_id,
            store=runtime.store,
            model_name=runtime.context.model,
            debug=runtime.context.debug,
        )
    except Exception as exc:
        logger.exception("Memory extraction failed")
        if runtime.context.debug:
            print("\n=== Memory Extractor Error ===")
            print(f"{type(exc).__name__}: {exc}")

    return {}


def build_graph(store: Any, checkpointer: Any | None = None):
    """Compile the Memory Agent LangGraph workflow."""

    builder = StateGraph(State, context_schema=Context)

    builder.add_node("call_model", call_model)
    builder.add_node("extract_memory", extract_memory)

    builder.add_edge("__start__", "call_model")
    builder.add_edge("call_model", "extract_memory")
    builder.add_edge("extract_memory", END)

    graph = builder.compile(
        store=store,
        checkpointer=checkpointer,
    )
    graph.name = "MemoryAgent"

    return graph
