import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from memory_agent.long_term_memory.consolidator import (
    MemoryDecision,
    MemoryExtractionResult,
    consolidate_memories,
)
from memory_agent.long_term_memory.store.base import MemoryItem
from memory_agent.long_term_memory.taxonomy import memory_namespace


class FakeStore:
    def __init__(self, search_results=None):
        self.search_results = search_results or {}
        self.put_calls = []

    async def asearch(self, namespace, query, limit=10, metadata_filter=None, **kwargs):
        return list(self.search_results.get(namespace, []))[:limit]

    async def aput(self, namespace, key, value, **kwargs):
        self.put_calls.append((namespace, key, value))


class FakeExtractor:
    def __init__(self, result):
        self.result = result
        self.prompt = None

    async def ainvoke(self, prompt, config=None):
        self.prompt = prompt
        return self.result


class FakeLLM:
    def __init__(self, extractor):
        self.extractor = extractor

    def with_structured_output(self, *_args, **_kwargs):
        return self.extractor


class MemoryConsolidatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_consolidate_memories_uses_recent_transcript_and_preserves_existing_type(self):
        user_id = "u-1"
        existing = MemoryItem(
            key="mem-1",
            value={
                "memory_type": "task",
                "content": "Keep PostgreSQL checkpoints.",
            },
            score=0.8,
        )
        store = FakeStore(
            {
                memory_namespace(user_id, "task"): [existing],
            }
        )
        extractor = FakeExtractor(
            MemoryExtractionResult(
                memories=[
                    MemoryDecision(
                        action="update",
                        memory_id="mem-1",
                        memory_type="knowledge",
                        content="Project keeps PostgreSQL checkpoints.",
                        context="Confirmed during implementation.",
                        category="project_context",
                        subject="Checkpoint storage",
                        entities=["PostgreSQL"],
                        topics=["checkpoints"],
                        confidence=0.9,
                    )
                ]
            )
        )
        llm = FakeLLM(extractor)

        with patch(
            "memory_agent.long_term_memory.consolidator.load_chat_model",
            return_value=llm,
        ):
            await consolidate_memories(
                messages=[
                    HumanMessage(content="Use PostgreSQL for checkpoints."),
                    AIMessage(content="Understood, checkpoints stay in PostgreSQL."),
                    HumanMessage(content="Please remember this for the project."),
                ],
                user_id=user_id,
                store=store,
                model_name="dummy",
                message_window=6,
                message_char_window=500,
                message_max_chars=200,
            )

        self.assertIn("Assistant: Understood, checkpoints stay in PostgreSQL.", extractor.prompt)
        self.assertEqual(len(store.put_calls), 1)
        namespace, key, value = store.put_calls[0]
        self.assertEqual(namespace, memory_namespace(user_id, "task"))
        self.assertEqual(key, "mem-1")
        self.assertEqual(value["memory_type"], "task")


if __name__ == "__main__":
    unittest.main()
