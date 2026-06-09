import logging
from uuid import uuid4

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.memory import InMemorySaver

from memory_agent import Context, SQLiteVectorMemoryStore, build_graph, load_settings
from memory_agent.llm import close_llm_clients


load_dotenv()
settings = load_settings()
logger = logging.getLogger(__name__)

store = SQLiteVectorMemoryStore(
    db_path=settings.memory_db_path,
    embedding_model=settings.embedding_model,
)

checkpointer = InMemorySaver()

graph = build_graph(
    store=store,
    checkpointer=checkpointer,
)


def _message_actions() -> list[cl.Action]:
    return [
        cl.Action(
            name="show_memories",
            payload={},
            label="查看记忆",
            tooltip="显示当前用户的长期记忆",
            icon="database",
        ),
        cl.Action(
            name="new_thread",
            payload={},
            label="新线程",
            tooltip="保留长期记忆，开启新的短期对话上下文",
            icon="messages-square",
        ),
    ]


def _get_session_user_id() -> str:
    user_id = cl.user_session.get("user_id")
    if user_id:
        return user_id

    cl.user_session.set("user_id", settings.default_user_id)
    return settings.default_user_id


def _get_session_thread_id() -> str:
    thread_id = cl.user_session.get("thread_id")
    if thread_id:
        return thread_id

    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)
    return thread_id


def _format_memory_panel(memories) -> str:
    if not memories:
        return "当前没有长期记忆。"

    blocks: list[str] = []

    for index, mem in enumerate(memories, start=1):
        value = mem.value
        score_line = ""
        if mem.score is not None:
            score_line = f"\n**Score**: `{mem.score:.3f}`"

        blocks.append(
            f"### 记忆 {index}\n"
            f"**ID**: `{mem.key}`\n"
            f"**Category**: `{value.get('category', 'general')}`\n"
            f"**Confidence**: `{value.get('confidence', 1.0)}`"
            f"{score_line}\n"
            f"**Updated**: `{value.get('updated_at', '')}`\n\n"
            f"{value.get('content', '')}\n\n"
            f"> {value.get('context', '') or '无额外上下文'}"
        )

    return "\n\n---\n\n".join(blocks)


@cl.on_app_shutdown
async def on_app_shutdown():
    await close_llm_clients()


@cl.on_chat_start
async def on_chat_start():
    user_id = settings.default_user_id
    thread_id = f"chat-{uuid4()}"

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        author="Memory Agent",
        content=(
            "你好，我是你的 Memory Agent。\n\n"
            f"- user_id: `{user_id}`\n"
            f"- model: `{settings.llm_model}`\n"
            f"- thread_id: `{thread_id}`\n\n"
            "你可以直接开始对话。"
        ),
        actions=_message_actions(),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    user_id = _get_session_user_id()
    thread_id = _get_session_thread_id()

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = Context(
        user_id=user_id,
        model=settings.llm_model,
        debug=settings.debug,
    )

    response_msg = cl.Message(
        content="",
        author="Memory Agent",
        actions=_message_actions(),
    )
    await response_msg.send()

    streamed = False

    try:
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
                streamed = True
                await response_msg.stream_token(token)
    except Exception:
        logger.exception("Graph execution failed")
        if not streamed:
            response_msg.content = (
                "抱歉，这轮对话没有成功完成。请检查后端日志和 `.env` 配置后重试。"
            )
        else:
            await cl.Message(
                author="Memory Agent",
                content="这轮回复已经输出，但后续处理时出现异常。请检查后端日志。",
                actions=_message_actions(),
            ).send()

    await response_msg.update()


@cl.action_callback("show_memories")
async def show_memories(action: cl.Action):
    user_id = _get_session_user_id()

    memories = await store.asearch(
        ("memories", user_id),
        query="",
        limit=20,
    )

    await cl.Message(
        author="Memory Agent",
        content=_format_memory_panel(memories),
        actions=_message_actions(),
    ).send()


@cl.action_callback("new_thread")
async def new_thread(action: cl.Action):
    thread_id = f"chat-{uuid4()}"
    cl.user_session.set("thread_id", thread_id)

    await cl.Message(
        author="Memory Agent",
        content=(
            "已开启新的对话线程。\n\n"
            f"- user_id: `{_get_session_user_id()}`\n"
            f"- thread_id: `{thread_id}`"
        ),
        actions=_message_actions(),
    ).send()


# env -u DEBUG chainlit run chainlit_app.py -w --host 127.0.0.1 --port 8000
