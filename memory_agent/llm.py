import os
import httpx
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.lower() in {"1", "true", "yes", "on"}

def load_chat_model(model_name: str) -> BaseChatModel:
    """
    Load a chat model based on the provided model name.

    Args:
        model_name (str): The name of the chat model to load.

    Returns:
        BaseChatModel: An instance of the loaded chat model.
    """
    # Load environment variables for LLM configuration
    llm_api_key = os.getenv("LLM_API_KEY")
    llm_base_url = os.getenv("LLM_BASE_URL")
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    llm_timeout = int(os.getenv("LLM_TIMEOUT", "30"))
    llm_trust_env = _env_bool("LLM_TRUST_ENV", default=False)

    # Initialize and return the chat model
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=llm_api_key,
        base_url=llm_base_url,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
        timeout=llm_timeout,
        http_client=httpx.Client(trust_env=llm_trust_env),
        http_async_client=httpx.AsyncClient(trust_env=llm_trust_env),
        http_socket_options=(),
        streaming=True,
    )