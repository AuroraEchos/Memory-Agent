import asyncio

from sentence_transformers import SentenceTransformer


class LocalEmbeddingProvider:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: str = "auto",
        concurrency: int = 1,
        batch_size: int = 32,
    ) -> None:
        self.device = self._resolve_device(device)
        self.model = SentenceTransformer(
            model_name_or_path,
            device=self.device,
        )
        self.batch_size = max(1, int(batch_size))
        self._semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = device.strip().lower()

        if normalized != "auto":
            return normalized

        try:
            import torch
        except ImportError:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"

    async def aembed(self, text: str) -> list[float]:
        vectors = await self.aembed_batch([text])
        return vectors[0]

    async def adimension(self) -> int:
        vector = await self.aembed("dimension probe")
        return len(vector)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        async with self._semaphore:
            return await asyncio.to_thread(self._embed_batch_sync, texts)

    def _embed_batch_sync(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
        )
        return vectors.astype(float).tolist()

    async def aclose(self) -> None:
        return None
