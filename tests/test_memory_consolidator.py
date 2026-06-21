import unittest

from memory_agent.long_term_memory.consolidator import (
    MemoryDecision,
    MemoryExtractionResult,
    _decision_contains_sensitive_text,
)


class MemoryDecisionTests(unittest.TestCase):
    def test_normalizes_aliases_and_metadata(self) -> None:
        decision = MemoryDecision.model_validate(
            {
                "decision": "create",
                "type": "user-preference",
                "preference": "The user prefers concise status updates.",
                "category": "general",
                "entities": "Codex, Memory Agent, Codex",
                "topics": [" status ", "", "memory"],
                "confidence": 1.7,
            }
        )

        self.assertEqual(decision.action, "create")
        self.assertEqual(decision.memory_type, "persona")
        self.assertEqual(decision.category, "preference")
        self.assertEqual(
            decision.content,
            "The user prefers concise status updates.",
        )
        self.assertEqual(decision.entities, ["Codex", "Memory Agent"])
        self.assertEqual(decision.topics, ["status", "memory"])
        self.assertEqual(decision.confidence, 1.0)

    def test_infers_update_action_from_memory_id(self) -> None:
        decision = MemoryDecision.model_validate(
            {
                "memory_id": "existing-id",
                "memory_type": "project_context",
                "content": "Memory Agent uses typed Qdrant namespaces.",
            }
        )

        self.assertEqual(decision.action, "update")
        self.assertEqual(decision.memory_type, "project")
        self.assertEqual(decision.category, "project_context")

    def test_normalizes_noop_to_ignore(self) -> None:
        decision = MemoryDecision.model_validate(
            {
                "action": "no-op",
                "memory_type": "summary",
            }
        )

        self.assertEqual(decision.action, "ignore")
        self.assertEqual(decision.memory_type, "episodic")
        self.assertEqual(decision.category, "conversation_summary")

    def test_detects_sensitive_text_in_metadata(self) -> None:
        decision = MemoryDecision.model_validate(
            {
                "action": "create",
                "memory_type": "entity",
                "content": "A service account was mentioned.",
                "subject": "API_KEY=super-secret-token",
            }
        )

        self.assertTrue(_decision_contains_sensitive_text(decision))


class MemoryExtractionResultTests(unittest.TestCase):
    def test_wraps_single_memory_decision_object(self) -> None:
        result = MemoryExtractionResult.model_validate(
            {
                "action": "create",
                "memory_type": "task",
                "content": "Track long-term memory migration work.",
            }
        )

        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].memory_type, "task")

    def test_accepts_common_collection_keys(self) -> None:
        result = MemoryExtractionResult.model_validate(
            {
                "decisions": [
                    {
                        "action": "ignore",
                    }
                ]
            }
        )

        self.assertEqual(len(result.memories), 1)
        self.assertEqual(result.memories[0].action, "ignore")


if __name__ == "__main__":
    unittest.main()
