import asyncio
import os
import unittest
from unittest.mock import patch

from memory_agent.config import load_settings
from memory_agent.long_term_memory.embedding.factory import create_embedding_provider
from memory_agent.long_term_memory.embedding.remote import RemoteEmbeddingProvider


class EmbeddingFactoryTests(unittest.TestCase):
    def test_requires_embedding_service_url(self) -> None:
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "http://127.0.0.1:6333"},
            clear=True,
        ):
            settings = load_settings()

        with self.assertRaisesRegex(ValueError, "EMBEDDING_SERVICE_URL is required"):
            create_embedding_provider(settings)

    def test_creates_remote_provider(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://127.0.0.1:6333",
                "EMBEDDING_SERVICE_URL": "http://127.0.0.1:8001",
            },
            clear=True,
        ):
            settings = load_settings()

        provider = create_embedding_provider(settings)
        try:
            self.assertIsInstance(provider, RemoteEmbeddingProvider)
        finally:
            asyncio.run(provider.aclose())


if __name__ == "__main__":
    unittest.main()
