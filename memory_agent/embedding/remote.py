from typing import Any

import httpx


class RemoteEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 30.0,
        trust_env: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=trust_env,
        )

    async def aembed(self, text: str) -> list[float]:
        vectors = await self.aembed_batch([text])
        return vectors[0]

    async def adimension(self) -> int:
        response = await self.client.get(f"{self.base_url}/dimension")
        response.raise_for_status()

        data = response.json()
        dimension = data.get("dimension")

        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError("Embedding service returned invalid dimension.")

        return dimension

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        response = await self.client.post(
            f"{self.base_url}/embed",
            json={"texts": texts},
        )
        response.raise_for_status()

        data = response.json()
        embeddings = data.get("embeddings")

        return self._normalize_embeddings(
            embeddings=embeddings,
            expected_count=len(texts),
        )

    @staticmethod
    def _normalize_embeddings(
        *,
        embeddings: Any,
        expected_count: int,
    ) -> list[list[float]]:
        if not isinstance(embeddings, list):
            raise RuntimeError("Embedding service returned invalid response.")

        if len(embeddings) != expected_count:
            raise RuntimeError(
                "Embedding service returned unexpected embedding count: "
                f"expected {expected_count}, got {len(embeddings)}."
            )

        normalized: list[list[float]] = []

        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list):
                raise RuntimeError(
                    "Embedding service returned invalid vector at "
                    f"index {index}."
                )

            try:
                normalized.append([float(value) for value in vector])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Embedding service returned non-numeric vector value at "
                    f"index {index}."
                ) from exc

        return normalized

    async def aclose(self) -> None:
        await self.client.aclose()
