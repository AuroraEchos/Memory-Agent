from memory_agent.config import AppSettings
from memory_agent.store.base import MemoryStore
from memory_agent.store.qdrant_store import QdrantMemoryStore


def create_memory_store(settings: AppSettings) -> MemoryStore:
    return QdrantMemoryStore(
        collection_name=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key,
        prefer_grpc=settings.qdrant_prefer_grpc,
    )
