import asyncio
import math
import weakref
from dataclasses import dataclass
from typing import Any

from memory_agent.embedding.base import EmbeddingProvider


@dataclass(frozen=True)
class MemoryRouteDecision:
    should_route: bool
    reason: str
    positive_score: float
    negative_score: float
    margin: float
    query_vector: list[float] | None = None


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []

        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text is not None:
                    chunks.append(str(text))
            else:
                chunks.append(str(item))

        return "\n".join(chunks)

    return str(content)


def latest_human_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            return content_to_text(getattr(msg, "content", ""))

    return ""


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot / (na * nb)


class SemanticMemoryRouter:
    """Embedding-based memory router.

    It makes two independent decisions:
    - whether the current turn needs long-term memory retrieval
    - whether the current turn is worth long-term memory extraction

    This is intentionally not an LLM router.
    """

    retrieval_positive_examples = [
        "你还记得我的偏好吗？",
        "按照我的代码风格帮我重构这个模块。",
        "根据我之前的项目背景给我建议。",
        "结合我的长期目标，帮我规划下一步。",
        "你知道我喜欢用什么语言写 Agent 吗？",
        "帮我写一篇适合我个人网站风格的博客。",
        "基于我的研究方向，帮我判断这个方案是否合适。",
        "按照我之前说过的要求继续优化。",
        "Based on my preferences, help me write this code.",
        "Do you remember what coding style I prefer?",
        "Use my previous project context to answer this.",
        "Help me plan based on my long-term goals.",
    ]

    retrieval_negative_examples = [
        "什么是 LangGraph？",
        "解释一下 Transformer 的注意力机制。",
        "Python 怎么读取 JSON 文件？",
        "给我介绍一下 Qdrant。",
        "这段代码为什么报错？",
        "写一个 hello world。",
        "你好。",
        "谢谢。",
        "继续。",
        "What is a vector database?",
        "Explain how async works in Python.",
        "How do I read a JSON file in Python?",
        "What does this error mean?",
    ]

    extraction_positive_examples = [
        "记住，我偏好简洁、类型清晰的代码。",
        "以后给我写代码时尽量使用类型标注。",
        "我喜欢用 Rust 写 Agent。",
        "我不喜欢过度设计的架构。",
        "我的目标是长期研究 Agent Memory。",
        "其实我现在更喜欢 Qdrant，而不是 SQLite。",
        "下次你帮我写博客时，风格要冷静克制。",
        "我正在做一个开源的 Memory Agent 项目。",
        "I prefer concise Python code with type hints.",
        "Remember that I am building a personal memory agent.",
        "From now on, use Rust first when writing agent code for me.",
        "My long-term goal is to research agent memory systems.",
    ]

    extraction_negative_examples = [
        "什么是 LangGraph 的 Store？",
        "帮我解释一下这段代码。",
        "写一个 hello world。",
        "这个报错是什么意思？",
        "继续讲。",
        "谢谢。",
        "给我一个示例。",
        "What is Qdrant?",
        "Explain the difference between list and tuple.",
        "Show me a simple example.",
        "Can you continue?",
    ]

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        retrieval_threshold: float = 0.58,
        extraction_threshold: float = 0.58,
        margin_threshold: float = 0.03,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.retrieval_threshold = retrieval_threshold
        self.extraction_threshold = extraction_threshold
        self.margin_threshold = margin_threshold

        self._initialized = False
        self._lock = asyncio.Lock()

        self._retrieval_positive_vectors: list[list[float]] = []
        self._retrieval_negative_vectors: list[list[float]] = []
        self._extraction_positive_vectors: list[list[float]] = []
        self._extraction_negative_vectors: list[list[float]] = []

    async def ainitialize(self) -> None:
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            texts = (
                self.retrieval_positive_examples
                + self.retrieval_negative_examples
                + self.extraction_positive_examples
                + self.extraction_negative_examples
            )

            vectors = await self.embedding_provider.aembed_batch(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    "Embedding provider returned unexpected vector count: "
                    f"expected {len(texts)}, got {len(vectors)}."
                )

            rp_end = len(self.retrieval_positive_examples)
            rn_end = rp_end + len(self.retrieval_negative_examples)
            ep_end = rn_end + len(self.extraction_positive_examples)

            self._retrieval_positive_vectors = vectors[:rp_end]
            self._retrieval_negative_vectors = vectors[rp_end:rn_end]
            self._extraction_positive_vectors = vectors[rn_end:ep_end]
            self._extraction_negative_vectors = vectors[ep_end:]

            self._initialized = True

    async def decide_retrieval(
        self,
        messages: list[Any],
        query_vector: list[float] | None = None,
    ) -> MemoryRouteDecision:
        text = latest_human_text(messages).strip()

        if not text:
            return MemoryRouteDecision(
                should_route=False,
                reason="empty_user_message",
                positive_score=0.0,
                negative_score=0.0,
                margin=0.0,
            )

        fast = self._retrieval_fast_path(text)
        if fast is not None:
            return fast

        await self.ainitialize()

        if query_vector is None:
            query_vector = await self.embedding_provider.aembed(text)

        return self._decide(
            query_vector=query_vector,
            positive_vectors=self._retrieval_positive_vectors,
            negative_vectors=self._retrieval_negative_vectors,
            threshold=self.retrieval_threshold,
            reason_if_true="semantic_retrieval_match",
            reason_if_false="semantic_retrieval_not_needed",
        )

    async def decide_extraction(
        self,
        messages: list[Any],
        query_vector: list[float] | None = None,
    ) -> MemoryRouteDecision:
        text = latest_human_text(messages).strip()

        if not text:
            return MemoryRouteDecision(
                should_route=False,
                reason="empty_user_message",
                positive_score=0.0,
                negative_score=0.0,
                margin=0.0,
            )

        fast = self._extraction_fast_path(text)
        if fast is not None:
            return fast

        await self.ainitialize()

        if query_vector is None:
            query_vector = await self.embedding_provider.aembed(text)

        return self._decide(
            query_vector=query_vector,
            positive_vectors=self._extraction_positive_vectors,
            negative_vectors=self._extraction_negative_vectors,
            threshold=self.extraction_threshold,
            reason_if_true="semantic_extraction_match",
            reason_if_false="semantic_extraction_not_needed",
        )

    def _retrieval_fast_path(self, text: str) -> MemoryRouteDecision | None:
        normalized = text.strip().lower()

        short_ack_or_greeting = {
            "hi",
            "hello",
            "ok",
            "thanks",
            "thank you",
            "你好",
            "谢谢",
            "好的",
            "继续",
            "嗯",
        }

        if normalized in short_ack_or_greeting:
            return MemoryRouteDecision(
                should_route=False,
                reason="fast_skip_short_ack_or_greeting",
                positive_score=0.0,
                negative_score=0.0,
                margin=0.0,
            )

        explicit_retrieval_signals = [
            "你还记得",
            "你记得",
            "记得我",
            "我之前",
            "之前我",
            "上次我",
            "我的偏好",
            "我的习惯",
            "按照我的",
            "根据我的",
            "结合我的",
            "适合我",
            "do you remember",
            "based on my",
            "my preference",
        ]

        if any(signal in normalized for signal in explicit_retrieval_signals):
            return MemoryRouteDecision(
                should_route=True,
                reason="fast_explicit_retrieval_signal",
                positive_score=1.0,
                negative_score=0.0,
                margin=1.0,
            )

        return None

    def _extraction_fast_path(self, text: str) -> MemoryRouteDecision | None:
        normalized = text.strip().lower()

        if len(normalized) < 8:
            return MemoryRouteDecision(
                should_route=False,
                reason="fast_skip_too_short",
                positive_score=0.0,
                negative_score=0.0,
                margin=0.0,
            )

        explicit_write_signals = [
            "记住",
            "记一下",
            "你要记得",
            "以后你",
            "下次你",
            "我的偏好是",
            "我偏好",
            "我喜欢",
            "我不喜欢",
            "remember",
            "from now on",
            "next time",
            "i prefer",
            "i like",
            "i don't like",
        ]

        if any(signal in normalized for signal in explicit_write_signals):
            return MemoryRouteDecision(
                should_route=True,
                reason="fast_explicit_extraction_signal",
                positive_score=1.0,
                negative_score=0.0,
                margin=1.0,
            )

        return None

    def _decide(
        self,
        *,
        query_vector: list[float],
        positive_vectors: list[list[float]],
        negative_vectors: list[list[float]],
        threshold: float,
        reason_if_true: str,
        reason_if_false: str,
    ) -> MemoryRouteDecision:
        positive_score = max(
            (
                cosine_similarity(query_vector, vector)
                for vector in positive_vectors
            ),
            default=0.0,
        )

        negative_score = max(
            (
                cosine_similarity(query_vector, vector)
                for vector in negative_vectors
            ),
            default=0.0,
        )

        margin = positive_score - negative_score

        should_route = (
            positive_score >= threshold
            and margin >= self.margin_threshold
        )

        return MemoryRouteDecision(
            should_route=should_route,
            reason=reason_if_true if should_route else reason_if_false,
            positive_score=positive_score,
            negative_score=negative_score,
            margin=margin,
            query_vector=query_vector,
        )


_router_cache: weakref.WeakKeyDictionary[Any, SemanticMemoryRouter] = (
    weakref.WeakKeyDictionary()
)


def get_memory_router(
    store: Any,
    *,
    retrieval_threshold: float,
    extraction_threshold: float,
    margin_threshold: float,
) -> SemanticMemoryRouter:
    router = _router_cache.get(store)

    if router is not None:
        return router

    embedding_provider = getattr(store, "embedding_provider", None)

    if embedding_provider is None:
        raise RuntimeError(
            "Semantic memory router requires store.embedding_provider."
        )

    router = SemanticMemoryRouter(
        embedding_provider=embedding_provider,
        retrieval_threshold=retrieval_threshold,
        extraction_threshold=extraction_threshold,
        margin_threshold=margin_threshold,
    )

    _router_cache[store] = router
    return router


def filter_retrieved_memories(
    memories: list[Any],
    *,
    min_score: float,
    max_memories: int,
) -> list[Any]:
    filtered: list[Any] = []

    for mem in memories:
        score = getattr(mem, "score", None)

        if score is not None and score < min_score:
            continue

        filtered.append(mem)

    return filtered[:max_memories]


def route_to_dict(decision: MemoryRouteDecision) -> dict[str, Any]:
    return {
        "should_route": decision.should_route,
        "reason": decision.reason,
        "positive_score": decision.positive_score,
        "negative_score": decision.negative_score,
        "margin": decision.margin,
        "has_query_vector": decision.query_vector is not None,
    }
