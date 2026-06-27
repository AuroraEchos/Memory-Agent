"""Chainlit web entrypoint for the Memory Agent application."""

import asyncio
import logging
from typing import Any

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from memory_agent import (
    Context,
    build_graph,
    create_memory_store,
    load_settings,
    to_psycopg_conninfo,
)
from memory_agent.chainlit_ui import (
    ASSISTANT_AUTHOR,
    content_to_text,
    event_output_to_text,
    get_session_thread_id,
    get_session_user_id,
    init_session,
    resume_session,
)
from memory_agent.llm import close_llm_clients


load_dotenv(dotenv_path=".env", override=False)
settings = load_settings()
logger = logging.getLogger(__name__)


if (
    not settings.chainlit_auth_secret
    or not settings.chainlit_auth_username
    or not settings.chainlit_auth_password
    or not settings.chainlit_auth_user_id
):
    raise RuntimeError(
        "CHAINLIT_AUTH_SECRET, CHAINLIT_AUTH_USERNAME, "
        "CHAINLIT_AUTH_PASSWORD, and CHAINLIT_AUTH_USER_ID are required "
        "to enable Chainlit authentication, chat history, and memory identity."
    )


store = create_memory_store(settings)

chainlit_data_layer = SQLAlchemyDataLayer(
    conninfo=settings.chainlit_database_url,
    show_logger=settings.debug,
)
checkpoint_conninfo = to_psycopg_conninfo(settings.chainlit_database_url)

_graph: Any | None = None
_checkpointer_cm: Any | None = None
_checkpointer: Any | None = None
_embedding_dimension_validated = False
_graph_lock = asyncio.Lock()


@cl.data_layer
def get_data_layer():
    """Provide Chainlit with the configured SQLAlchemy data layer."""

    return chainlit_data_layer


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate the single configured password-based Chainlit user."""

    if (
        username == settings.chainlit_auth_username
        and password == settings.chainlit_auth_password
    ):
        return cl.User(
            identifier=settings.chainlit_auth_user_id,
            metadata={
                "role": "user",
                "provider": "credentials",
            },
        )

    return None


async def persist_message_step(message: Any) -> None:
    """Synchronously persist the final Chainlit step state.

    Chainlit persists message send/update operations in background tasks.
    Awaiting the data layer here makes resumed chat history much less likely
    to show a user message without the assistant's final answer.
    """

    try:
        await chainlit_data_layer.update_step(message.to_dict())
    except Exception:
        logger.exception("Failed to persist Chainlit message step")


async def ensure_embedding_dimension() -> None:
    """Validate the remote embedding dimension once before graph startup."""

    global _embedding_dimension_validated

    if _embedding_dimension_validated:
        return

    embedding_provider = getattr(store, "embedding_provider", None)
    adimension = getattr(embedding_provider, "adimension", None)

    if not callable(adimension):
        logger.warning(
            "Memory store embedding provider does not expose adimension(); "
            "skipping startup embedding dimension validation"
        )
        _embedding_dimension_validated = True
        return

    actual_dimension = int(await adimension())
    expected_dimension = int(settings.embedding_dimension)

    if actual_dimension != expected_dimension:
        raise RuntimeError(
            "Embedding dimension mismatch: "
            f"EMBEDDING_DIMENSION={expected_dimension}, "
            f"embedding service returned {actual_dimension}."
        )

    _embedding_dimension_validated = True
    logger.info("Embedding dimension validated: %s", actual_dimension)


async def ensure_graph():
    """Lazily initialize the LangGraph app and PostgreSQL checkpointer."""

    global _graph, _checkpointer_cm, _checkpointer

    if _graph is not None:
        return _graph

    async with _graph_lock:
        if _graph is not None:
            return _graph

        try:
            await ensure_embedding_dimension()

            _checkpointer_cm = AsyncPostgresSaver.from_conn_string(
                checkpoint_conninfo
            )
            _checkpointer = await _checkpointer_cm.__aenter__()
            await _checkpointer.setup()

            _graph = build_graph(
                store=store,
                checkpointer=_checkpointer,
            )

            logger.info("LangGraph checkpoint DB: PostgreSQL")

            return _graph

        except Exception:
            logger.exception("Failed to initialize LangGraph")

            if _checkpointer_cm is not None:
                try:
                    await _checkpointer_cm.__aexit__(None, None, None)
                except Exception:
                    logger.exception(
                        "Failed to close LangGraph checkpointer after "
                        "initialization failure"
                    )

            _checkpointer_cm = None
            _checkpointer = None
            _graph = None
            raise


@cl.on_app_shutdown
async def on_app_shutdown():
    """Release graph, memory-store, and LLM resources on shutdown."""

    global _checkpointer_cm, _checkpointer, _graph

    try:
        if _checkpointer_cm is not None:
            await _checkpointer_cm.__aexit__(None, None, None)
    except Exception:
        logger.exception("Failed to close LangGraph checkpointer")
    finally:
        _checkpointer_cm = None
        _checkpointer = None
        _graph = None

    try:
        await store.aclose()
    except Exception:
        logger.exception("Failed to close memory store")

    try:
        await close_llm_clients()
    except Exception:
        logger.exception("Failed to close LLM clients")


@cl.on_chat_start
async def on_chat_start():
    """Initialize a new Chainlit chat session."""

    await ensure_graph()

    user_id, thread_id = init_session()

    await cl.Message(
        author=ASSISTANT_AUTHOR,
        content=(
            "你好，我是你的 Memory Agent。\n\n"
            f"- user_id: `{user_id}`\n"
            f"- model: `{settings.llm_model}`\n"
            f"- thread_id: `{thread_id}`\n\n"
            "你可以直接开始对话。"
        ),
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Restore session metadata when a Chainlit thread is resumed."""

    await ensure_graph()

    user_id, thread_id = resume_session(thread)

    if settings.debug:
        print("\n=== Chat Resumed ===")
        print(f"user_id={user_id}")
        print(f"thread_id={thread_id}")


@cl.on_message
async def on_message(message: cl.Message):
    """Run the LangGraph workflow for one user message and stream the answer."""

    graph = await ensure_graph()

    user_id = get_session_user_id()
    thread_id = get_session_thread_id()

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    context = Context(
        user_id=user_id,
        thread_id=thread_id,
        model=settings.llm_model,
        debug=settings.debug,
        message_window=settings.conversation_message_window,
    )

    response_msg = cl.Message(
        content="",
        author=ASSISTANT_AUTHOR,
    )

    streamed = False
    error_after_stream = False
    response_finalized = False

    async def finalize_response(*, fallback_empty: bool = False) -> bool:
        """Send and persist the assistant response exactly once."""

        nonlocal response_finalized

        if response_finalized:
            return True

        if not response_msg.content.strip():
            if not fallback_empty:
                return False
            response_msg.content = "抱歉，这轮对话没有生成有效回复。请稍后重试。"

        await response_msg.send()
        await persist_message_step(response_msg)
        response_finalized = True
        return True

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
            data = event.get("data", {})

            if node_name != "call_model":
                continue

            if event_type == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue

                token = content_to_text(getattr(chunk, "content", ""))

                if token:
                    streamed = True
                    await response_msg.stream_token(token)

                continue

            if event_type in {"on_chat_model_end", "on_chain_end"}:
                output = data.get("output")

                final_text = event_output_to_text(output)
                if final_text and not response_msg.content.strip():
                    response_msg.content = final_text
                await finalize_response()

    except Exception:
        logger.exception("Graph execution failed")
        if response_finalized:
            error_after_stream = True
        elif not streamed:
            response_msg.content = (
                "抱歉，这轮对话没有成功完成。请检查后端日志和 `.env` 配置后重试。"
            )
        else:
            error_after_stream = True

    if not response_finalized:
        await finalize_response(fallback_empty=True)

    if error_after_stream:
        error_msg = cl.Message(
            author=ASSISTANT_AUTHOR,
            content="这轮回复已经输出，但后续处理时出现异常。请检查后端日志。",
        )
        await error_msg.send()
        await persist_message_step(error_msg)
