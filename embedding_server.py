from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memory_agent.config import env_int, env_str
from memory_agent.embedding.local import LocalEmbeddingProvider


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=256)


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimension: int


provider: LocalEmbeddingProvider | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global provider

    load_dotenv()

    provider = LocalEmbeddingProvider(
        model_name_or_path=env_str("EMBEDDING_MODEL", "./models/bge-m3")
        or "./models/bge-m3",
        device=env_str("EMBEDDING_DEVICE", "auto") or "auto",
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


def get_provider() -> LocalEmbeddingProvider:
    if provider is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding provider is not initialized.",
        )

    return provider


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "memory-agent-embedding", "status": "ok"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/dimension")
async def dimension() -> dict[str, int]:
    current_provider = get_provider()
    return {"dimension": await current_provider.adimension()}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest) -> EmbeddingResponse:
    current_provider = get_provider()
    embeddings = await current_provider.aembed_batch(req.texts)
    dimension = len(embeddings[0]) if embeddings else 0

    return EmbeddingResponse(
        embeddings=embeddings,
        dimension=dimension,
    )
