from memory_agent.config import AppSettings
from memory_agent.embedding.base import EmbeddingProvider
from memory_agent.embedding.local import LocalEmbeddingProvider
from memory_agent.embedding.remote import RemoteEmbeddingProvider


def create_embedding_provider(settings: AppSettings) -> EmbeddingProvider:
    backend = settings.embedding_backend.strip().lower()

    if backend == "local":
        return LocalEmbeddingProvider(
            model_name_or_path=settings.embedding_model,
            device=settings.embedding_device,
            concurrency=settings.embedding_concurrency,
            batch_size=settings.embedding_batch_size,
        )

    if backend == "remote":
        if not settings.embedding_service_url:
            raise ValueError(
                "EMBEDDING_BACKEND=remote requires EMBEDDING_SERVICE_URL."
            )

        return RemoteEmbeddingProvider(
            base_url=settings.embedding_service_url,
            timeout=settings.embedding_timeout,
            trust_env=settings.embedding_trust_env,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_BACKEND={settings.embedding_backend!r}. "
        "Expected 'local' or 'remote'."
    )