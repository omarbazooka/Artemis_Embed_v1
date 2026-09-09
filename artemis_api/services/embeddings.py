from __future__ import annotations

from typing import List
import httpx

from artemis_api.config import settings
from artemis_api.services.vector_math import truncate_and_normalize


class EmbeddingServiceError(RuntimeError):
    pass


class HuggingFaceEmbeddingService:
    """Call a hosted Artemis Sentence Transformers endpoint.

    The hosted model returns full 768D sentence embeddings. Matryoshka output
    selection is applied here by slicing the prefix and L2-normalizing again.
    """

    def __init__(self) -> None:
        self.url = settings.hf_url
        self.token = settings.hf_token
        self.timeout = settings.hf_request_timeout_seconds

    async def _request_full_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.url:
            raise EmbeddingServiceError(
                "Hugging Face inference is not configured. Set HF_MODEL_ID or HF_EMBEDDING_URL."
            )

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise EmbeddingServiceError(
                f"Hugging Face inference returned {response.status_code}: {response.text[:1000]}"
            )

        data = response.json()
        if isinstance(data, dict):
            for key in ("embeddings", "data", "output"):
                if key in data:
                    data = data[key]
                    break

        if not isinstance(data, list) or not data:
            raise EmbeddingServiceError("Unexpected Hugging Face response: expected a non-empty list")

        if data and isinstance(data[0], (int, float)):
            data = [data]

        if len(data) != len(texts):
            raise EmbeddingServiceError(f"Expected {len(texts)} embeddings, received {len(data)}")

        vectors: List[List[float]] = []
        for item in data:
            if not isinstance(item, list) or not item:
                raise EmbeddingServiceError("Unexpected embedding item in Hugging Face response")
            if isinstance(item[0], list):
                raise EmbeddingServiceError(
                    "Endpoint returned token-level embeddings. Deploy the Artemis Sentence Transformers export."
                )
            vectors.append([float(x) for x in item])
        return vectors

    async def embed(self, texts: List[str], dimension: int = 768) -> List[List[float]]:
        full = await self._request_full_embeddings(texts)
        return [truncate_and_normalize(vector, dimension) for vector in full]


embedding_service = HuggingFaceEmbeddingService()
