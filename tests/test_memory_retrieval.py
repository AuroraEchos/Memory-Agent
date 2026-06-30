import unittest

from langchain_core.messages import HumanMessage

from memory_agent.long_term_memory.retrieval import (
    build_memory_query,
    format_memories,
    retrieve_relevant_memories,
)
from memory_agent.long_term_memory.store.base import MemoryItem
from memory_agent.long_term_memory.taxonomy import memory_namespace


class FakeMemoryStore:
    def __init__(self, results_by_namespace, embedding_provider=None):
        self.results_by_namespace = results_by_namespace
        self.embedding_provider = embedding_provider
        self.queries = []

    async def asearch(self, namespace, query, limit=10, metadata_filter=None, **kwargs):
        self.queries.append(
            {
                "namespace": namespace,
                "query": query,
                "query_vector": kwargs.get("query_vector"),
            }
        )
        return list(self.results_by_namespace.get(namespace, []))[:limit]


class FakeEmbeddingProvider:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    async def aembed(self, text):
        self.calls += 1
        return list(self.vector)


class MemoryRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_relevant_memories_prefers_score_and_dedupes(self):
        user_id = "u-1"
        duplicate = MemoryItem(
            key="shared",
            value={"memory_type": "persona", "content": "User prefers concise answers."},
            score=0.30,
        )
        stronger = MemoryItem(
            key="project-1",
            value={"memory_type": "project", "content": "Project uses PostgreSQL checkpoints."},
            score=0.95,
        )
        duplicate_stronger = MemoryItem(
            key="shared",
            value={"memory_type": "knowledge", "content": "User prefers concise answers."},
            score=0.85,
        )

        store = FakeMemoryStore(
            {
                memory_namespace(user_id, "persona"): [duplicate],
                memory_namespace(user_id, "project"): [stronger],
                memory_namespace(user_id, "knowledge"): [duplicate_stronger],
            }
        )

        results = await retrieve_relevant_memories(
            store=store,
            user_id=user_id,
            query="How should we store checkpoints?",
        )

        self.assertEqual(results[0].key, "project-1")
        self.assertEqual(len([item for item in results if item.key == "shared"]), 1)

    async def test_retrieve_relevant_memories_reuses_one_query_embedding(self):
        user_id = "u-1"
        embedding_provider = FakeEmbeddingProvider([0.1, 0.2, 0.3])
        store = FakeMemoryStore(
            {
                memory_namespace(user_id, "persona"): [
                    MemoryItem(
                        key="m-1",
                        value={"memory_type": "persona", "content": "User prefers fast answers."},
                        score=0.8,
                    )
                ]
            },
            embedding_provider=embedding_provider,
        )

        await retrieve_relevant_memories(
            store=store,
            user_id=user_id,
            query="How can we make this faster?",
        )

        self.assertEqual(embedding_provider.calls, 1)
        self.assertTrue(store.queries)
        self.assertTrue(
            all(query["query_vector"] == [0.1, 0.2, 0.3] for query in store.queries)
        )

    def test_build_memory_query_truncates_long_user_input(self):
        query = build_memory_query(
            [HumanMessage(content="a" * 100)],
            max_chars=32,
        )

        self.assertLessEqual(len(query), 32)
        self.assertIn("[truncated]", query)

    def test_format_memories_truncates_long_fields(self):
        prompt = format_memories(
            [
                MemoryItem(
                    key="m-1",
                    value={
                        "memory_type": "project",
                        "category": "project_context",
                        "subject": "s" * 200,
                        "content": "c" * 500,
                        "context": "k" * 400,
                        "entities": ["entity" * 20],
                        "topics": ["topic" * 20],
                        "confidence": 0.9,
                    },
                    score=0.9,
                )
            ]
        )

        self.assertIn("[truncated]", prompt)


if __name__ == "__main__":
    unittest.main()
