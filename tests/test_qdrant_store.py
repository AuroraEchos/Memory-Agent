import unittest

from memory_agent.store.qdrant_store import QdrantMemoryStore


class DummyEmbeddingProvider:
    async def aembed(self, text: str) -> list[float]:
        return [0.0]

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    async def adimension(self) -> int:
        return 1

    async def aclose(self) -> None:
        return None


class QdrantMemoryStoreTests(unittest.TestCase):
    def test_requires_qdrant_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "QDRANT_URL is required"):
            QdrantMemoryStore(
                collection_name="test_memories",
                embedding_provider=DummyEmbeddingProvider(),
                vector_size=1,
                qdrant_url="",
            )


if __name__ == "__main__":
    unittest.main()
