import logging
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.runtime import Runtime

from memory_agent.config import Context
from memory_agent.llm import load_chat_model
from memory_agent.memory_extractor import extract_and_store_memories
from memory_agent.state import State


logger = logging.getLogger(__name__)


def _message_text(messages: list[Any], n: int = 3) -> str:
    chunks: list[str] = []

    for msg in messages[-n:]:
        content = getattr(msg, "content", "")

        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.append(str(content))
        else:
            chunks.append(repr(content))

    return "\n".join(chunks)


def _format_memories(memories: list[Any]) -> str:
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
    if runtime.store is None:
        raise RuntimeError("runtime.store is None. Compile the graph with a store.")

    user_id = runtime.context.user_id
    query = _message_text(state.messages, n=3)

    try:
        memories = await runtime.store.asearch(
            ("memories", user_id),
            query=query,
            limit=10,
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
        time=datetime.now().isoformat(),
    )

    llm = load_chat_model(runtime.context.model, streaming=True)

    response = await llm.ainvoke(
        [
            {"role": "system", "content": system_message},
            *state.messages,
        ]
    )

    return {"messages": [response]}


async def extract_memory(state: State, runtime: Runtime[Context]) -> dict:
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
