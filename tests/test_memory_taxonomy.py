import unittest

from memory_agent.long_term_memory.taxonomy import (
    MEMORY_CATEGORY_HINTS,
    MEMORY_RETRIEVAL_LIMITS,
    MEMORY_TYPE_VALUES,
    iter_memory_routes,
    memory_namespace,
    normalize_category,
    normalize_memory_type,
    normalize_str_list,
    taxonomy_prompt,
)


class MemoryTaxonomyTests(unittest.TestCase):
    def test_memory_namespace_includes_canonical_type(self) -> None:
        self.assertEqual(
            memory_namespace("wenhao", "user-preference"),
            ("memories", "wenhao", "persona"),
        )

    def test_normalizes_memory_type_aliases(self) -> None:
        self.assertEqual(normalize_memory_type("semantic"), "knowledge")
        self.assertEqual(normalize_memory_type("conversation summary"), "episodic")
        self.assertEqual(normalize_memory_type("tool-habit"), "procedural")
        self.assertEqual(normalize_memory_type("unknown"), "knowledge")

    def test_normalizes_category_with_type_specific_default(self) -> None:
        self.assertEqual(
            normalize_category(None, memory_type="project"),
            MEMORY_CATEGORY_HINTS["project"][0],
        )
        self.assertEqual(
            normalize_category("Domain Fact", memory_type="knowledge"),
            "domain_fact",
        )

    def test_normalizes_str_list_and_caps_length(self) -> None:
        values = normalize_str_list(["alpha", " beta ", "alpha", "", *range(20)])

        self.assertEqual(values[:3], ["alpha", "beta", "0"])
        self.assertEqual(len(values), 12)

    def test_iter_memory_routes_covers_all_types_in_priority_order(self) -> None:
        routes = iter_memory_routes("wenhao")

        self.assertEqual(
            [route.memory_type for route in routes],
            list(MEMORY_TYPE_VALUES),
        )
        self.assertEqual(
            [route.limit for route in routes],
            [MEMORY_RETRIEVAL_LIMITS[memory_type] for memory_type in MEMORY_TYPE_VALUES],
        )

    def test_taxonomy_prompt_mentions_every_type(self) -> None:
        prompt = taxonomy_prompt()

        for memory_type in MEMORY_TYPE_VALUES:
            self.assertIn(f"- {memory_type}:", prompt)


if __name__ == "__main__":
    unittest.main()
