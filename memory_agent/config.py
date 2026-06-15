"""Environment-backed configuration for the Memory Agent application."""

import os
from dataclasses import dataclass, field, fields
from urllib.parse import urlsplit, urlunsplit
from typing import ClassVar
from typing_extensions import Annotated

from memory_agent.prompts import SYSTEM_PROMPT


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with common truthy spellings."""

    value = os.getenv(name)

    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    """Read an integer environment variable and raise on invalid values."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def env_int_any(names: tuple[str, ...], default: int) -> int:
    """Read the first configured integer from a list of environment names."""

    for name in names:
        value = os.getenv(name)

        if value is None:
            continue

        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {value!r}") from exc

    return default


def env_float(name: str, default: float) -> float:
    """Read a float environment variable and raise on invalid values."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def env_str(name: str, default: str | None = None) -> str | None:
    """Read a stripped string environment variable, treating blanks as absent."""

    value = os.getenv(name)

    if value is None:
        return default

    value = value.strip()
    return value or default


def required_env_str(name: str) -> str:
    """Read a required string environment variable."""

    value = env_str(name)

    if not value:
        raise ValueError(
            f"{name} is required. For host development use "
            f"{name}=http://localhost:6333; inside Docker Compose use "
            f"{name}=http://qdrant:6333."
        )

    return value


def to_psycopg_conninfo(database_url: str) -> str:
    """Convert a SQLAlchemy Postgres URL into a psycopg connection URL."""

    database_url = database_url.strip()
    parsed = urlsplit(database_url)
    scheme = parsed.scheme

    if "+" in scheme:
        scheme = scheme.split("+", 1)[0]

    if scheme == "postgres":
        scheme = "postgresql"

    if scheme != "postgresql":
        raise ValueError(
            "CHAINLIT_DATABASE_URL must use a PostgreSQL URL because "
            "LangGraph checkpoints are stored in PostgreSQL."
        )

    return urlunsplit(
        (
            scheme,
            parsed.netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


@dataclass(frozen=True, kw_only=True)
class AppSettings:
    """Process-wide settings loaded from environment variables."""

    llm_model: str
    llm_api_key: str | None
    llm_base_url: str | None
    llm_temperature: float
    llm_max_completion_tokens: int
    llm_timeout: int
    llm_trust_env: bool
    llm_streaming: bool

    qdrant_url: str
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_prefer_grpc: bool

    embedding_dimension: int
    embedding_service_url: str | None
    embedding_timeout: float
    embedding_trust_env: bool

    chainlit_database_url: str | None
    chainlit_auth_secret: str | None
    chainlit_auth_username: str | None
    chainlit_auth_password: str | None
    chainlit_auth_user_id: str | None

    conversation_message_window: int
    debug: bool


def load_settings() -> AppSettings:
    """Build immutable application settings from the current environment."""

    return AppSettings(
        llm_model=env_str("LLM_MODEL"),
        llm_api_key=env_str("LLM_API_KEY"),
        llm_base_url=env_str("LLM_BASE_URL"),
        llm_temperature=env_float("LLM_TEMPERATURE", 0.7),
        llm_max_completion_tokens=env_int_any(
            ("LLM_MAX_COMPLETION_TOKENS", "LLM_MAX_TOKENS"),
            2048,
        ),
        llm_timeout=env_int("LLM_TIMEOUT", 30),
        llm_trust_env=env_bool("LLM_TRUST_ENV", False),
        llm_streaming=env_bool("LLM_STREAMING", True),

        qdrant_url=required_env_str("QDRANT_URL"),
        qdrant_api_key=env_str("QDRANT_API_KEY"),
        qdrant_collection=env_str("QDRANT_COLLECTION", "agent_memories")
        or "agent_memories",
        qdrant_prefer_grpc=env_bool("QDRANT_PREFER_GRPC", False),

        embedding_dimension=env_int("EMBEDDING_DIMENSION", 1024),
        embedding_service_url=env_str("EMBEDDING_SERVICE_URL"),
        embedding_timeout=env_float("EMBEDDING_TIMEOUT", 30.0),
        embedding_trust_env=env_bool("EMBEDDING_TRUST_ENV", False),

        chainlit_database_url=env_str("CHAINLIT_DATABASE_URL"),
        chainlit_auth_secret=env_str("CHAINLIT_AUTH_SECRET"),
        chainlit_auth_username=env_str("CHAINLIT_AUTH_USERNAME"),
        chainlit_auth_password=env_str("CHAINLIT_AUTH_PASSWORD"),
        chainlit_auth_user_id=env_str("CHAINLIT_AUTH_USER_ID"),

        conversation_message_window=env_int("CONVERSATION_MESSAGE_WINDOW", 20),
        debug=env_bool("APP_DEBUG", False),
    )


@dataclass(kw_only=True)
class Context:
    """Per-run LangGraph context passed to graph nodes."""

    user_id: str

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

    message_window: int = field(
        default=20,
        metadata={"env": "CONVERSATION_MESSAGE_WINDOW"},
    )

    system_prompt: ClassVar[str] = SYSTEM_PROMPT

    def __post_init__(self) -> None:
        """Apply environment overrides for default context values."""

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
                    elif isinstance(default_value, int):
                        setattr(self, f.name, env_int(env_name, default_value))
                    elif isinstance(default_value, float):
                        setattr(self, f.name, env_float(env_name, default_value))
                    else:
                        setattr(self, f.name, env_value)
