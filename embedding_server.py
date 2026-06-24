"""Standalone FastAPI service for sentence-transformers embeddings."""

import asyncio
import math
import re
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer

from memory_agent.config import env_int, env_str


MAX_EMBEDDING_BATCH_SIZE = 256
MAX_EMBEDDING_TEXT_LENGTH = 32_768


class EmbeddingRequest(BaseModel):
    """Request body for embedding one or more texts."""

    texts: list[str] = Field(
        min_length=1,
        max_length=MAX_EMBEDDING_BATCH_SIZE,
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        """Reject blank or excessively large texts before model inference."""

        for index, text in enumerate(texts):
            if not text.strip():
                raise ValueError(f"texts[{index}] must not be blank")
            if len(text) > MAX_EMBEDDING_TEXT_LENGTH:
                raise ValueError(
                    f"texts[{index}] exceeds the maximum length of "
                    f"{MAX_EMBEDDING_TEXT_LENGTH} characters"
                )

        return texts


class EmbeddingResponse(BaseModel):
    """Response body containing embeddings and their vector dimension."""

    embeddings: list[list[float]]
    dimension: int
    model: str


class EmbeddingServiceInfo(BaseModel):
    """Ready-state metadata for the loaded embedding model."""

    status: str = "ok"
    model: str
    device: str
    dimension: int


class EmbeddingDimensionResponse(BaseModel):
    """Embedding dimension and model identity returned to clients."""

    dimension: int
    model: str


class EmbeddingModelService:
    """In-process sentence-transformers wrapper used by the HTTP service."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "auto",
        concurrency: int = 1,
        batch_size: int = 32,
    ) -> None:
        """Load the embedding model and configure local inference limits."""

        self.model_name_or_path = model_name_or_path
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
        )
        self.batch_size = max(1, int(batch_size))
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))
        self._dimension_lock = asyncio.Lock()
        self._dimension = self._read_model_dimension()

    @staticmethod
    def _cuda_is_available() -> bool:
        """Return whether the current PyTorch runtime can use CUDA."""

        return bool(torch.cuda.is_available())

    @staticmethod
    def _xpu_is_available() -> bool:
        """Return whether the current PyTorch runtime can use Intel XPU."""

        xpu = getattr(torch, "xpu", None)
        is_available = getattr(xpu, "is_available", None)
        return bool(callable(is_available) and is_available())

    @staticmethod
    def _mps_is_available() -> bool:
        """Return whether the current PyTorch runtime can use Apple MPS."""

        mps = getattr(torch.backends, "mps", None)
        is_available = getattr(mps, "is_available", None)
        return bool(callable(is_available) and is_available())

    @classmethod
    def _detect_device(cls) -> str:
        """Choose the best accelerator exposed by the PyTorch runtime."""

        if cls._cuda_is_available():
            return "cuda"
        if cls._xpu_is_available():
            return "xpu"
        if cls._mps_is_available():
            return "mps"
        return "cpu"

    @classmethod
    def _resolve_device(cls, device: str) -> str:
        """Resolve auto selection or validate an explicitly requested device."""

        normalized = device.strip().lower()

        if normalized in {"", "auto"}:
            return cls._detect_device()

        if normalized == "cpu":
            return "cpu"

        if normalized == "gpu":
            detected = cls._detect_device()
            if detected == "cpu":
                raise ValueError(
                    "EMBEDDING_DEVICE=gpu was requested, but no supported GPU "
                    "backend is available in the current PyTorch runtime."
                )
            return detected

        cuda_match = re.fullmatch(r"cuda(?::(\d+))?", normalized)
        if cuda_match:
            if not cls._cuda_is_available():
                raise ValueError(
                    f"EMBEDDING_DEVICE={device!r} was requested, but CUDA is "
                    "not available. Install a CUDA-enabled PyTorch build and "
                    "check the GPU driver/runtime."
                )

            index = cuda_match.group(1)
            if index is not None and int(index) >= torch.cuda.device_count():
                raise ValueError(
                    f"EMBEDDING_DEVICE={device!r} refers to an unavailable "
                    f"CUDA device; detected {torch.cuda.device_count()} device(s)."
                )
            return normalized

        xpu_match = re.fullmatch(r"xpu(?::(\d+))?", normalized)
        if xpu_match:
            if not cls._xpu_is_available():
                raise ValueError(
                    f"EMBEDDING_DEVICE={device!r} was requested, but Intel XPU "
                    "is not available in the current PyTorch runtime."
                )

            index = xpu_match.group(1)
            xpu = getattr(torch, "xpu")
            if index is not None and int(index) >= xpu.device_count():
                raise ValueError(
                    f"EMBEDDING_DEVICE={device!r} refers to an unavailable "
                    f"XPU device; detected {xpu.device_count()} device(s)."
                )
            return normalized

        if normalized == "mps":
            if not cls._mps_is_available():
                raise ValueError(
                    "EMBEDDING_DEVICE='mps' was requested, but Apple MPS is "
                    "not available in the current PyTorch runtime."
                )
            return "mps"

        raise ValueError(
            "EMBEDDING_DEVICE must be one of auto, cpu, gpu, cuda, cuda:N, "
            "xpu, xpu:N, or mps."
        )

    def _read_model_dimension(self) -> int | None:
        """Read the model's declared dimension without running inference."""

        get_dimension = getattr(self.model, "get_embedding_dimension", None)

        dimension = get_dimension()
        if dimension is None:
            return None

        dimension = int(dimension)
        if dimension <= 0:
            raise RuntimeError(
                "Embedding model reported a non-positive vector dimension."
            )

        return dimension

    async def adimension(self) -> int:
        """Return the cached embedding dimension, probing only as fallback."""

        if self._dimension is not None:
            return self._dimension

        async with self._dimension_lock:
            if self._dimension is not None:
                return self._dimension

            vectors = await self.aembed_batch(["dimension probe"])
            return len(vectors[0])

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts without blocking the event loop."""

        if not texts:
            return []

        async with self._semaphore:
            return await asyncio.to_thread(self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        """Run sentence-transformers encoding and validate its output."""

        raw_vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )

        try:
            vectors = raw_vectors.astype(float).tolist()
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Embedding model returned an unsupported output value."
            ) from exc

        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError(
                "Embedding model returned an unexpected number of vectors."
            )

        actual_dimension: int | None = None
        normalized: list[list[float]] = []

        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or not vector:
                raise RuntimeError(
                    f"Embedding model returned an invalid vector at index {index}."
                )

            if actual_dimension is None:
                actual_dimension = len(vector)
            elif len(vector) != actual_dimension:
                raise RuntimeError(
                    "Embedding model returned vectors with inconsistent dimensions."
                )

            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError(
                    f"Embedding model returned a non-finite value at index {index}."
                )

            normalized.append(vector)

        if actual_dimension is None:
            raise RuntimeError("Embedding model returned no vectors.")

        if self._dimension is not None and actual_dimension != self._dimension:
            raise RuntimeError(
                "Embedding model output dimension changed: "
                f"expected {self._dimension}, got {actual_dimension}."
            )

        self._dimension = actual_dimension
        return normalized

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


@app.get("/health", response_model=EmbeddingServiceInfo)
async def health() -> EmbeddingServiceInfo:
    """Return readiness metadata for the initialized model service."""

    current_provider = get_provider()
    return EmbeddingServiceInfo(
        model=current_provider.model_name_or_path,
        device=current_provider.device,
        dimension=await current_provider.adimension(),
    )


@app.get("/dimension", response_model=EmbeddingDimensionResponse)
async def dimension() -> EmbeddingDimensionResponse:
    """Return the embedding vector dimension."""

    current_provider = get_provider()
    return EmbeddingDimensionResponse(
        dimension=await current_provider.adimension(),
        model=current_provider.model_name_or_path,
    )


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(req: EmbeddingRequest) -> EmbeddingResponse:
    """Embed the requested texts and return normalized vectors."""

    current_provider = get_provider()
    embeddings = await current_provider.aembed_batch(req.texts)
    dimension = len(embeddings[0]) if embeddings else 0

    return EmbeddingResponse(
        embeddings=embeddings,
        dimension=dimension,
        model=current_provider.model_name_or_path,
    )
