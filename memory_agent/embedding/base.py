from typing import Protocol


class EmbeddingProvider(Protocol):
    async def aembed(self, text: str) -> list[float]:
        ...

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        ...

    async def adimension(self) -> int:
        ...

    async def aclose(self) -> None:
        ...
