import unittest

from langchain_core.messages import AIMessage

from memory_agent.chainlit_ui import extract_token_usage, format_token_usage


class TokenUsageDisplayTests(unittest.TestCase):
    def test_extracts_langchain_usage_metadata(self) -> None:
        message = AIMessage(
            content="hello",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

        self.assertEqual(
            extract_token_usage(message),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        )

    def test_extracts_openai_style_response_metadata(self) -> None:
        message = AIMessage(
            content="hello",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                }
            },
        )

        self.assertEqual(
            extract_token_usage(message),
            {
                "input_tokens": 12,
                "output_tokens": 8,
                "total_tokens": 20,
            },
        )

    def test_extracts_usage_from_graph_node_output(self) -> None:
        message = AIMessage(
            content="hello",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                }
            },
        )

        self.assertEqual(
            extract_token_usage({"messages": [message]}),
            {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
        )

    def test_formats_token_usage_for_chainlit(self) -> None:
        self.assertEqual(
            format_token_usage(
                {
                    "input_tokens": 1234,
                    "output_tokens": 56,
                    "total_tokens": 1290,
                }
            ),
            "**Token 用量**: 输入 `1,234` · 输出 `56` · 总计 `1,290`",
        )

    def test_preserves_zero_token_counts(self) -> None:
        self.assertEqual(
            extract_token_usage(
                {
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    }
                }
            ),
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )

    def test_missing_usage_formats_as_empty_string(self) -> None:
        self.assertEqual(format_token_usage({}), "")
        self.assertEqual(extract_token_usage({"messages": []}), {})


if __name__ == "__main__":
    unittest.main()
