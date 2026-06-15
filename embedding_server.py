"""Standalone FastAPI service for sentence-transformers embeddings."""

import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from memory_agent.config import env_int, env_str


class EmbeddingRequest(BaseModel):
    """Request body for embedding one or more texts."""

    texts: list[str] = Field(default_factory=list, max_length=256)


class EmbeddingResponse(BaseModel):
    """Response body containing embeddings and their vector dimension."""

    embeddings: list[list[float]]
    dimension: int


class EmbeddingModelService:
    """In-process sentence-transformers wrapper used by the HTTP service."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "cpu",
        concurrency: int = 1,
        batch_size: int = 32,
    ) -> None:
        """Load the embedding model and configure local inference limits."""

        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
        )
        self.batch_size = max(1, int(batch_size))
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Normalize and validate the configured inference device."""

        normalized = device.strip().lower()

        if normalized in {"", "auto", "cpu"}:
            return "cpu"

        raise ValueError(
            "Only EMBEDDING_DEVICE=cpu is supported for now. "
            "Other runtimes can be added later."
        )

    async def adimension(self) -> int:
        """Return the embedding dimension by running a small probe."""

        vectors = await self.aembed_batch(["dimension probe"])
        return len(vectors[0])

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts without blocking the event loop."""

        if not texts:
            return []

        async with self._semaphore:
            return await asyncio.to_thread(self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Run sentence-transformers encoding in a worker thread."""

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return vectors.astype(float).tolist()

    async def aclose(self) -> None:
        """Release resources held by the model service."""

        return None


provider: EmbeddingModelService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the model once for the FastAPI application lifecycle."""

    global provider

    load_dotenv(dotenv_path=".env", override=False)

    provider = EmbeddingModelService(
        model_name_or_path=env_str("EMBEDDING_MODEL", "./models/bge-m3")
        or "./models/bge-m3",
        device=env_str("EMBEDDING_DEVICE", "cpu") or "cpu",
        concurrency=env_int("EMBEDDING_CONCURRENCY", 1),
        batch_size=env_int("EMBEDDING_BATCH_SIZE", 32),
    )

    try:
        yield
    finally:
        if provider is not None:
            await provider.aclose()
            provider = None


app = FastAPI(
    title="Memory Agent Embedding Service",
    lifespan=lifespan,
)


def get_provider() -> EmbeddingModelService:
    """Return the initialized provider or raise a service-unavailable error."""

    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding provider is not initialized.",
        )

    return provider


@app.get("/")
async def root() -> dict[str, str]:
    """Return a basic service identity payload."""

    return {"service": "memory-agent-embedding", "status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a lightweight health check response."""

    return {"status": "ok"}


@app.get("/dimension")
async def dimension() -> dict[str, int]:
    """Return the embedding vector dimension."""

    current_provider = get_provider()
    return {"dimension": await current_provider.adimension()}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest) -> EmbeddingResponse:
    """Embed the requested texts and return normalized vectors."""

    current_provider = get_provider()
    embeddings = await current_provider.aembed_batch(req.texts)
    dimension = len(embeddings[0]) if embeddings else 0

    return EmbeddingResponse(
        embeddings=embeddings,
        dimension=dimension,
    )
