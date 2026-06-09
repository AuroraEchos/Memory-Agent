import os
from uuid import uuid4

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from memory_agent import Context, SQLiteVectorMemoryStore, build_graph


load_dotenv()

store = SQLiteVectorMemoryStore(
    db_path=os.environ.get("MEMORY_DB_PATH", "./memory.db"),
    embedding_model=os.environ.get("EMBEDDING_MODEL", "./models/bge-m3"),
)

checkpointer = InMemorySaver()

graph = build_graph(
    store=store,
    checkpointer=checkpointer,
)


@cl.on_chat_start
async def on_chat_start():
    user_id = os.environ.get("DEFAULT_USER_ID")
    thread_id = f"chat-{uuid4()}"

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        content=(
            "你好，我是你的 Memory Agent。\n\n"
            f"- user_id: `{user_id}`\n"
            f"- thread_id: `{thread_id}`\n\n"
            "你可以直接开始对话。"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_id = cl.user_session.get("user_id")
    thread_id = cl.user_session.get("thread_id")

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = Context(
        user_id=user_id,
        model=os.environ.get("MIMO_MODEL", "mimo-v2.5-pro"),
        debug=False,
    )

    response_msg = cl.Message(content="")
    await response_msg.send()

    async for event in graph.astream_events(
        {
            "messages": [
                HumanMessage(content=message.content)
            ]
        },
        config=config,
        context=context,
        version="v2",
    ):
        event_type = event.get("event")
        metadata = event.get("metadata", {})
        node_name = metadata.get("langgraph_node")

        if node_name != "call_model":
            continue

        if event_type != "on_chat_model_stream":
            continue

        chunk = event["data"].get("chunk")
        if chunk is None:
            continue

        token = getattr(chunk, "content", "")
        if isinstance(token, list):
            token = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in token
            )

        if token:
            await response_msg.stream_token(token)

    await response_msg.update()


@cl.action_callback("show_memories")
async def show_memories(action: cl.Action):
    user_id = cl.user_session.get("user_id")

    memories = await store.asearch(
        ("memories", user_id),
        query="",
        limit=20,
    )

    if not memories:
        await cl.Message(content="当前没有长期记忆。").send()
        return

    content = "\n\n".join(
        f"**ID**: `{mem.key}`\n"
        f"**Score**: `{mem.score}`\n"
        f"**Content**: {mem.value.get('content', '')}\n"
        f"**Category**: `{mem.value.get('category', 'general')}`\n"
        f"**Context**: {mem.value.get('context', '')}"
        for mem in memories
    )

    await cl.Message(content=content).send()


# chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000