from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class MemoryItem:
    key: str
    value: dict[str, Any]
    score: float | None = None


class MemoryStore(Protocol):
    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        ...

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        **kwargs: Any,
    ) -> MemoryItem | None:
        ...

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        ...

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str,
        limit: int = 10,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemoryItem]:
        ...
