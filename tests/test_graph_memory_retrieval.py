import asyncio
import unittest

from memory_agent.graph import _retrieve_relevant_memories
from memory_agent.memory_taxonomy import (
    MEMORY_RETRIEVAL_LIMITS,
    MEMORY_TYPE_VALUES,
    memory_namespace,
)
from memory_agent.store.base import MemoryItem


class FakeMemoryStore:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str, int]] = []

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str,
        limit: int = 10,
        **_: object,
    ) -> list[MemoryItem]:
        self.calls.append((namespace, query, limit))
        memory_type = namespace[-1]
        return [
            MemoryItem(
                key=f"{memory_type}-memory",
                value={
                    "memory_type": memory_type,
                    "content": f"{memory_type} content",
                },
                score=0.5,
            )
        ]


class MemoryRetrievalTests(unittest.TestCase):
    def test_retrieves_from_all_taxonomy_namespaces(self) -> None:
        store = FakeMemoryStore()

        memories = asyncio.run(
            _retrieve_relevant_memories(
                store=store,
                user_id="wenhao",
                query="memory agent",
            )
        )

        self.assertEqual(
            store.calls,
            [
                (
                    memory_namespace("wenhao", memory_type),
                    "memory agent",
                    MEMORY_RETRIEVAL_LIMITS[memory_type],
                )
                for memory_type in MEMORY_TYPE_VALUES
            ],
        )
        self.assertEqual(
            [memory.value["memory_type"] for memory in memories],
            list(MEMORY_TYPE_VALUES),
        )

    def test_blank_query_skips_retrieval(self) -> None:
        store = FakeMemoryStore()

        memories = asyncio.run(
            _retrieve_relevant_memories(
                store=store,
                user_id="wenhao",
                query=" ",
            )
        )

        self.assertEqual(memories, [])
        self.assertEqual(store.calls, [])


if __name__ == "__main__":
    unittest.main()
