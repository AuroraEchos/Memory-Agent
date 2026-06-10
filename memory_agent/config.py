import os
from dataclasses import dataclass, field, fields
from typing import ClassVar
from typing_extensions import Annotated

from memory_agent.prompts import SYSTEM_PROMPT


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()
    return value or default


@dataclass(frozen=True, kw_only=True)
class AppSettings:
    llm_model: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout: int
    llm_trust_env: bool
    llm_streaming: bool
    memory_db_path: str
    embedding_model: str
    embedding_device: str
    default_user_id: str
    debug: bool


def load_settings() -> AppSettings:
    return AppSettings(
        llm_model=env_str("LLM_MODEL", "mimo-v2.5-pro") or "mimo-v2.5-pro",
        llm_api_key=env_str("LLM_API_KEY"),
        llm_base_url=env_str("LLM_BASE_URL"),
        llm_temperature=env_float("LLM_TEMPERATURE", 0.7),
        llm_max_tokens=env_int("LLM_MAX_TOKENS", 2048),
        llm_timeout=env_int("LLM_TIMEOUT", 30),
        llm_trust_env=env_bool("LLM_TRUST_ENV", False),
        llm_streaming=env_bool("LLM_STREAMING", True),
        memory_db_path=env_str("MEMORY_DB_PATH", "./memory.db") or "./memory.db",
        embedding_model=env_str("EMBEDDING_MODEL", "./models/bge-m3") or "./models/bge-m3",
        embedding_device=env_str("EMBEDDING_DEVICE", "auto") or "auto",
        default_user_id=env_str("DEFAULT_USER_ID", "default_user") or "default_user",
        debug=env_bool("APP_DEBUG", False),
    )


@dataclass(kw_only=True)
class Context:
    user_id: str = field(
        default="default_user",
        metadata={"env": "DEFAULT_USER_ID"},
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="mimo-v2.5-pro",
        metadata={
            "description": "Model name used by the OpenAI-compatible endpoint.",
            "env": "LLM_MODEL",
        },
    )

    debug: bool = field(
        default=False,
        metadata={"env": "APP_DEBUG"},
    )

    system_prompt: ClassVar[str] = SYSTEM_PROMPT

    def __post_init__(self) -> None:
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            default_value = f.default

            if current_value == default_value:
                env_name = f.metadata.get("env", f.name.upper())
                env_value = os.environ.get(env_name)
                if env_value is not None:
                    if isinstance(default_value, bool):
                        setattr(self, f.name, env_bool(env_name, default_value))
                    else:
                        setattr(self, f.name, env_value)
