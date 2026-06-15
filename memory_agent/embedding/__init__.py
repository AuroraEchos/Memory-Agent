"""Embedding provider exports for the Memory Agent package."""

from memory_agent.embedding.base import EmbeddingProvider
from memory_agent.embedding.factory import create_embedding_provider
from memory_agent.embedding.remote import RemoteEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "RemoteEmbeddingProvider",
    "create_embedding_provider",
]
