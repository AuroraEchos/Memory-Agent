"""Factory for constructing the configured embedding provider."""

from memory_agent.config import AppSettings
from memory_agent.embedding.base import EmbeddingProvider
from memory_agent.embedding.remote import RemoteEmbeddingProvider


def create_embedding_provider(settings: AppSettings) -> EmbeddingProvider:
    """Create the remote embedding provider required by the main app."""

    if not settings.embedding_service_url:
        raise ValueError(
            "EMBEDDING_SERVICE_URL is required. Start embedding_server.py "
            "or provide a remote embedding service URL."
        )

    return RemoteEmbeddingProvider(
        base_url=settings.embedding_service_url,
        timeout=settings.embedding_timeout,
        trust_env=settings.embedding_trust_env,
    )
