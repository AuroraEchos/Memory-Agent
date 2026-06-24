"""HTTP client for the standalone embedding service."""

import math
from typing import Any

import httpx


class RemoteEmbeddingProvider:
    """Asynchronous embedding provider backed by the FastAPI service."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float = 30.0,
        trust_env: bool = False,
    ) -> None:
        """Create a reusable HTTP client for embedding service requests."""

        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=timeout,
            trust_env=trust_env,
        )
        self._dimension: int | None = None
        self._model: str | None = None

    @property
    def model(self) -> str | None:
        """Return the model identity reported by the remote service."""

        return self._model

    async def aembed(self, text: str) -> list[float]:
        """Embed a single text by delegating to the batch endpoint."""

        vectors = await self.aembed_batch([text])
        return vectors[0]

    async def adimension(self) -> int:
        """Fetch and cache the dimension reported by the service."""

        if self._dimension is not None:
            return self._dimension

        data = await self._request_json("GET", "/dimension")
        dimension = data.get("dimension")

        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
        ):
            raise RuntimeError("Embedding service returned invalid dimension.")

        self._update_service_metadata(
            dimension=dimension,
            model=data.get("model"),
        )
        return dimension

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts through the service's /embed endpoint."""

        if not texts:
            return []

        data = await self._request_json(
            "POST",
            "/embed",
            json={"texts": texts},
        )
        embeddings = data.get("embeddings")

        normalized = self._normalize_embeddings(
            embeddings=embeddings,
            expected_count=len(texts),
        )
        actual_dimension = len(normalized[0])
        reported_dimension = data.get("dimension")

        if (
            not isinstance(reported_dimension, int)
            or isinstance(reported_dimension, bool)
            or reported_dimension != actual_dimension
        ):
            raise RuntimeError(
                "Embedding service returned inconsistent dimension metadata: "
                f"reported {reported_dimension!r}, got {actual_dimension}."
            )

        self._update_service_metadata(
            dimension=actual_dimension,
            model=data.get("model"),
        )
        return normalized

    async def _request_json(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Request one endpoint and normalize transport/protocol failures."""

        try:
            response = await self.client.request(
                method,
                f"{self.base_url}{path}",
                **kwargs,
            )
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Embedding service request failed for {path}: {exc}"
            ) from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip().replace("\n", " ")[:300]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                "Embedding service returned "
                f"HTTP {response.status_code} for {path}{suffix}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Embedding service returned invalid JSON for {path}."
            ) from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Embedding service returned a non-object response for {path}."
            )

        return data

    def _update_service_metadata(
        self,
        *,
        dimension: int,
        model: Any,
    ) -> None:
        """Keep model and dimension stable for this provider lifetime."""

        if self._dimension is not None and self._dimension != dimension:
            raise RuntimeError(
                "Embedding service dimension changed while the client was running: "
                f"expected {self._dimension}, got {dimension}."
            )

        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise RuntimeError(
                    "Embedding service returned an invalid model identity."
                )
            if self._model is not None and self._model != model:
                raise RuntimeError(
                    "Embedding service model changed while the client was running: "
                    f"expected {self._model!r}, got {model!r}."
                )
            self._model = model

        self._dimension = dimension

    @staticmethod
    def _normalize_embeddings(
        *,
        embeddings: Any,
        expected_count: int,
    ) -> list[list[float]]:
        """Validate and coerce embedding service output into float vectors."""

        if not isinstance(embeddings, list):
            raise RuntimeError("Embedding service returned invalid response.")

        if len(embeddings) != expected_count:
            raise RuntimeError(
                "Embedding service returned unexpected embedding count: "
                f"expected {expected_count}, got {len(embeddings)}."
            )

        normalized: list[list[float]] = []
        dimension: int | None = None

        for index, vector in enumerate(embeddings):
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(
                    "Embedding service returned invalid vector at "
                    f"index {index}."
                )

            try:
                normalized_vector = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Embedding service returned non-numeric vector value at "
                    f"index {index}."
                ) from exc

            if not all(math.isfinite(value) for value in normalized_vector):
                raise RuntimeError(
                    "Embedding service returned non-finite vector value at "
                    f"index {index}."
                )

            if dimension is None:
                dimension = len(normalized_vector)
            elif len(normalized_vector) != dimension:
                raise RuntimeError(
                    "Embedding service returned vectors with inconsistent "
                    "dimensions."
                )

            normalized.append(normalized_vector)

        return normalized

    async def aclose(self) -> None:
        """Close the underlying asynchronous HTTP client."""

        await self.client.aclose()
