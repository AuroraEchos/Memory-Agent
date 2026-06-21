"""LangGraph workflow nodes for chat response and memory consolidation."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from memory_agent.config import Context
from memory_agent.context_builder import build_context_messages
from memory_agent.llm import load_chat_model
from memory_agent.long_term_memory.consolidator import consolidate_memories
from memory_agent.long_term_memory.retrieval import (
    build_memory_query,
    format_memories,
    retrieve_relevant_memories,
    serialize_memory_hit,
)
from memory_agent.long_term_memory.store.base import MemoryStore
from memory_agent.state import State


logger = logging.getLogger(__name__)


async def call_model(state: State, runtime: Runtime[Context]) -> dict:
    """Retrieve relevant memories, call the chat model, and return hits."""

    if runtime.store is None:
        raise RuntimeError("runtime.store is None. Compile the graph with a store.")

    user_id = runtime.context.user_id
    query = build_memory_query(state.messages, human_message_limit=2)

    try:
        memories = await retrieve_relevant_memories(
            store=runtime.store,
            user_id=user_id,
            query=query,
        )
    except Exception as exc:
        logger.exception("Memory retrieval failed")
        if runtime.context.debug:
            print("\n=== Memory Retrieval Error ===")
            print(f"{type(exc).__name__}: {exc}")
        memories = []

    if runtime.context.debug:
        print("\n=== Retrieved Memories ===")
        for memory in memories:
            print(memory.key, memory.value, "score=", memory.score)

    system_message = runtime.context.system_prompt.format(
        user_info=format_memories(memories),
        time=datetime.now(timezone.utc).isoformat(),
    )

    llm = load_chat_model(runtime.context.model)
    context_messages = build_context_messages(
        state.messages,
        runtime.context.message_window,
    )

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_message},
            *context_messages,
        ]
    )

    return {
        "messages": [response],
        "memory_hits": [serialize_memory_hit(memory) for memory in memories],
    }


async def extract_memory(state: State, runtime: Runtime[Context]) -> dict:
    """Consolidate durable memories after a chat response has been generated."""

    if runtime.store is None:
        raise RuntimeError("runtime.store is None. Compile the graph with a store.")

    try:
        await consolidate_memories(
            messages=state.messages,
            user_id=runtime.context.user_id,
            store=runtime.store,
            model_name=runtime.context.model,
            thread_id=getattr(runtime.context, "thread_id", None),
            debug=runtime.context.debug,
        )
    except Exception as exc:
        logger.exception("Memory consolidation failed")
        if runtime.context.debug:
            print("\n=== Memory Consolidation Error ===")
            print(f"{type(exc).__name__}: {exc}")

    return {}


def build_graph(store: MemoryStore, checkpointer: Any | None = None):
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
