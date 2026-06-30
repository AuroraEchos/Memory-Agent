import unittest

from langchain_core.messages import AIMessage, HumanMessage

from memory_agent.context_builder import (
    build_context_messages,
    build_conversation_transcript,
)


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_messages_respects_char_budget_and_keeps_latest_user_turn(self):
        messages = [
            HumanMessage(content="old request"),
            AIMessage(content="old answer that should be dropped by the tight budget"),
            HumanMessage(content="x" * 80),
        ]

        context_messages = build_context_messages(
            messages,
            message_window=10,
            max_total_chars=30,
            max_message_chars=20,
        )

        self.assertEqual(len(context_messages), 1)
        self.assertEqual(context_messages[0]["role"], "user")
        self.assertIn("[truncated]", context_messages[0]["content"])

    def test_build_conversation_transcript_labels_roles(self):
        transcript = build_conversation_transcript(
            [
                HumanMessage(content="Need Postgres checkpoints."),
                AIMessage(content="We will keep Postgres checkpoints."),
            ],
            message_window=4,
            max_total_chars=200,
            max_message_chars=100,
        )

        self.assertIn("User: Need Postgres checkpoints.", transcript)
        self.assertIn("Assistant: We will keep Postgres checkpoints.", transcript)


if __name__ == "__main__":
    unittest.main()
