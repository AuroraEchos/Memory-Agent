import unittest

from langchain_core.messages import AIMessage, HumanMessage

from memory_agent.context_builder import build_context_messages


def contents(messages) -> list[str]:
    return [message.content for message in messages]


class ContextBuilderTests(unittest.TestCase):
    def test_keeps_current_user_message_when_window_is_small(self) -> None:
        messages = [
            HumanMessage(content="old question"),
            AIMessage(content="old answer"),
            HumanMessage(content="current question"),
        ]

        self.assertEqual(
            contents(build_context_messages(messages, message_window=1)),
            ["current question"],
        )

    def test_keeps_recent_complete_turns_without_starting_from_ai(self) -> None:
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
        ]

        selected = build_context_messages(messages, message_window=3)

        self.assertEqual(contents(selected), ["q2", "a2", "q3"])
        self.assertEqual(selected[0].type, "human")

    def test_drops_oldest_complete_turn_when_it_does_not_fit(self) -> None:
        messages = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
        ]

        self.assertEqual(
            contents(build_context_messages(messages, message_window=2)),
            ["q3"],
        )

    def test_filters_empty_messages_and_orphan_ai_messages(self) -> None:
        messages = [
            AIMessage(content="orphan answer"),
            HumanMessage(content="q1"),
            AIMessage(content=""),
            AIMessage(content="a1"),
            HumanMessage(content="   "),
            HumanMessage(content="q2"),
        ]

        selected = build_context_messages(messages, message_window=4)

        self.assertEqual(contents(selected), ["q1", "a1", "q2"])
        self.assertTrue(all(message.content.strip() for message in selected))
        self.assertEqual(selected[0].type, "human")

    def test_skips_incomplete_older_user_segment(self) -> None:
        messages = [
            HumanMessage(content="abandoned question"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
        ]

        self.assertEqual(
            contents(build_context_messages(messages, message_window=5)),
            ["q2", "a2", "q3"],
        )


if __name__ == "__main__":
    unittest.main()
