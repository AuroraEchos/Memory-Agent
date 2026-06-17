import os
import unittest
from unittest.mock import patch

from memory_agent.config import Context, load_settings, to_psycopg_conninfo


class PostgresConninfoTests(unittest.TestCase):
    def test_converts_asyncpg_sqlalchemy_url_for_psycopg(self) -> None:
        self.assertEqual(
            to_psycopg_conninfo(
                "postgresql+asyncpg://user:pass@localhost:5432/app?ssl=require"
            ),
            "postgresql://user:pass@localhost:5432/app?ssl=require",
        )

    def test_normalizes_postgres_scheme(self) -> None:
        self.assertEqual(
            to_psycopg_conninfo("postgres://user:pass@localhost/app"),
            "postgresql://user:pass@localhost/app",
        )

    def test_rejects_non_postgres_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "PostgreSQL"):
            to_psycopg_conninfo("sqlite+aiosqlite:///tmp/checkpoints.sqlite")


class SettingsTests(unittest.TestCase):
    def test_qdrant_url_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "QDRANT_URL is required"):
                load_settings()

    def test_embedding_service_url_defaults_to_none(self) -> None:
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "http://127.0.0.1:6333"},
            clear=True,
        ):
            settings = load_settings()

        self.assertIsNone(settings.embedding_service_url)

    def test_embedding_backend_is_not_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://127.0.0.1:6333",
                "EMBEDDING_BACKEND": "local",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(hasattr(settings, "embedding_backend"))
        self.assertFalse(hasattr(settings, "embedding_model"))
        self.assertFalse(hasattr(settings, "embedding_device"))

    def test_checkpoint_db_path_is_not_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://127.0.0.1:6333",
                "CHECKPOINT_DB_PATH": "/tmp/old.sqlite",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(hasattr(settings, "checkpoint_db_path"))

    def test_qdrant_path_is_not_loaded(self) -> None:
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "http://127.0.0.1:6333",
                "QDRANT_PATH": "/tmp/old-qdrant-data",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertFalse(hasattr(settings, "qdrant_path"))
        self.assertEqual(settings.qdrant_url, "http://127.0.0.1:6333")

    def test_qdrant_url_can_be_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {"QDRANT_URL": "http://qdrant.example:6333"},
            clear=True,
        ):
            settings = load_settings()

        self.assertEqual(settings.qdrant_url, "http://qdrant.example:6333")

    def test_empty_qdrant_url_is_rejected(self) -> None:
        with patch.dict(os.environ, {"QDRANT_URL": ""}, clear=True):
            with self.assertRaisesRegex(ValueError, "QDRANT_URL is required"):
                load_settings()


class ContextTests(unittest.TestCase):
    def test_thread_id_is_part_of_runtime_context(self) -> None:
        context = Context(user_id="wenhao", thread_id="thread-123")

        self.assertEqual(context.user_id, "wenhao")
        self.assertEqual(context.thread_id, "thread-123")


if __name__ == "__main__":
    unittest.main()
