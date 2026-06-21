"""Shared protocol for asynchronous embedding providers."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol implemented by embedding providers used by memory stores."""

    async def aembed(self, text: str) -> list[float]:
        """Embed one text string into a dense vector."""

        ...

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into dense vectors."""

        ...

    async def adimension(self) -> int:
        """Return the vector dimension produced by the provider."""

        ...

    async def aclose(self) -> None:
        """Release any network or model resources held by the provider."""

        ...
