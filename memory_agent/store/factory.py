from memory_agent.config import AppSettings
from memory_agent.embedding import create_embedding_provider
from memory_agent.store.base import MemoryStore
from memory_agent.store.qdrant_store import QdrantMemoryStore


def create_memory_store(settings: AppSettings) -> MemoryStore:
    embedding_provider = create_embedding_provider(settings)

    return QdrantMemoryStore(
        collection_name=settings.qdrant_collection,
        embedding_provider=embedding_provider,
        vector_size=settings.embedding_dimension,
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )
