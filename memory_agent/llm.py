import threading

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from memory_agent.config import load_settings


_HTTP_CLIENTS: dict[bool, tuple[httpx.Client, httpx.AsyncClient]] = {}
_HTTP_CLIENT_LOCK = threading.Lock()


def _get_http_clients(trust_env: bool) -> tuple[httpx.Client, httpx.AsyncClient]:
    with _HTTP_CLIENT_LOCK:
        clients = _HTTP_CLIENTS.get(trust_env)

        if clients is None:
            clients = (
                httpx.Client(trust_env=trust_env),
                httpx.AsyncClient(trust_env=trust_env),
            )
            _HTTP_CLIENTS[trust_env] = clients

        return clients


async def close_llm_clients() -> None:
    with _HTTP_CLIENT_LOCK:
        clients = list(_HTTP_CLIENTS.values())
        _HTTP_CLIENTS.clear()

    for client, async_client in clients:
        client.close()
        await async_client.aclose()


def load_chat_model(
    model_name: str | None = None,
    *,
    streaming: bool | None = None,
) -> BaseChatModel:
    """
    Load a chat model based on the provided model name.

    Args:
        model_name (str): The name of the chat model to load.

    Returns:
        BaseChatModel: An instance of the loaded chat model.
    """
    settings = load_settings()

    if not settings.llm_api_key:
        raise RuntimeError("Missing LLM_API_KEY in .env")

    if not settings.llm_base_url:
        raise RuntimeError("Missing LLM_BASE_URL in .env")

    http_client, http_async_client = _get_http_clients(settings.llm_trust_env)

    return init_chat_model(
        model=model_name or settings.llm_model,
        model_provider="openai",
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_completion_tokens=settings.llm_max_completion_tokens,
        timeout=settings.llm_timeout,
        http_client=http_client,
        http_async_client=http_async_client,
        http_socket_options=(),
        streaming=settings.llm_streaming if streaming is None else streaming,
    )
