"""Abstract memory store contract shared by graph and UI code."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class MemoryItem:
    """Normalized memory item returned by a memory store."""

    key: str
    value: dict[str, Any]
    score: float | None = None


class MemoryStore(Protocol):
    """Protocol implemented by asynchronous long-term memory stores."""

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Create or update one memory item."""

        ...

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        **kwargs: Any,
    ) -> MemoryItem | None:
        """Fetch one memory item by namespace and key."""

        ...

    async def adelete(
        self,
        namespace: tuple[str, ...],
        key: str,
    ) -> None:
        """Delete one memory item by namespace and key."""

        ...

    async def asearch(
        self,
        namespace: tuple[str, ...],
        query: str,
        limit: int = 10,
        metadata_filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[MemoryItem]:
        """Search memory items by query text or list them when query is blank.

        Implementations may accept optional kwargs such as `query_vector` to
        reuse a precomputed embedding across multiple namespace searches.
        """

        ...

    async def aclose(self) -> None:
        """Release resources held by the store."""

        ...
