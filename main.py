import asyncio
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from memory_agent import Context, SQLiteVectorMemoryStore, build_graph, load_settings


async def run_turn(
    *,
    graph,
    user_id: str,
    thread_id: str,
    content: str,
    context: Context,
):
    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=content)
            ]
        },
        config=config,
        context=context,
    )

    print("\nUser:")
    print(content)

    print("\nAssistant:")
    print(result["messages"][-1].content)

    return result


async def print_memories(store, user_id: str, query: str = "") -> None:
    memories = await store.asearch(
        ("memories", user_id),
        query=query,
        limit=20,
    )

    print("\n=== Current Long-term Memories ===")

    if not memories:
        print("No memories.")
        return

    for mem in memories:
        print(f"\nID: {mem.key}")
        print(f"Score: {mem.score}")
        print(f"Value: {mem.value}")


async def main() -> None:
    load_dotenv()

    settings = load_settings()

    store = SQLiteVectorMemoryStore(
        db_path=settings.memory_db_path,
        embedding_model=settings.embedding_model,
    )

    checkpointer = InMemorySaver()

    graph = build_graph(
        store=store,
        checkpointer=checkpointer,
    )

    user_id = settings.default_user_id
    thread_id = f"test-thread-{uuid4()}"

    context = Context(
        user_id=user_id,
        model=settings.llm_model,
        debug=True,
    )

    await run_turn(
        graph=graph,
        user_id=user_id,
        thread_id=thread_id,
        context=context,
        content="我喜欢用 Python 写 Agent，并且偏好代码风格简洁、类型清晰。",
    )

    await print_memories(
        store=store,
        user_id=user_id,
        query="用户喜欢什么代码风格？",
    )

    await run_turn(
        graph=graph,
        user_id=user_id,
        thread_id=thread_id,
        context=context,
        content="其实我现在更喜欢用 Rust 写 Agent，但 Python 也还会用。",
    )

    await print_memories(
        store=store,
        user_id=user_id,
        query="用户喜欢用什么语言写 Agent？",
    )

    new_thread_id = f"new-chat-{uuid4()}"

    await run_turn(
        graph=graph,
        user_id=user_id,
        thread_id=new_thread_id,
        context=context,
        content="新开一个对话后，你还记得我写 Agent 的语言偏好吗？",
    )


if __name__ == "__main__":
    asyncio.run(main())
