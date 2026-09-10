from __future__ import annotations

from typing import List
import httpx

from artemis_api.config import settings
from artemis_api.services.vector_math import truncate_and_normalize


class EmbeddingServiceError(RuntimeError):
    pass


class RemoteEmbeddingService:
    """Call a hosted Artemis sentence-embedding endpoint.

    The remote runtime returns full 768D sentence embeddings. For the current
    development deployment that runtime may be a Kaggle-hosted Artemis process
    exposed through a temporary tunnel. Hugging Face remains the model registry.

    Matryoshka output selection stays inside this API: slice the leading D
    dimensions and L2-normalize again after truncation.
    """

    def __init__(self) -> None:
        self.url = settings.hf_url
        self.token = settings.hf_token
        self.timeout = settings.hf_request_timeout_seconds

    async def _request_full_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.url:
            raise EmbeddingServiceError(
                "Artemis remote inference is not configured. Set HF_EMBEDDING_URL "
                "or HF_MODEL_ID."
            )

        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=payload)

        if response.status_code >= 400:
            raise EmbeddingServiceError(
                f"Remote Artemis inference returned {response.status_code}: "
                f"{response.text[:1000]}"
            )

        data = response.json()
        if isinstance(data, dict):
            for key in ("embeddings", "data", "output"):
                if key in data:
                    data = data[key]
                    break

        if not isinstance(data, list) or not data:
            raise EmbeddingServiceError(
                "Unexpected remote inference response: expected a non-empty list"
            )

        if isinstance(data[0], (int, float)):
            data = [data]

        if len(data) != len(texts):
            raise EmbeddingServiceError(
                f"Expected {len(texts)} embeddings, received {len(data)}"
            )

        vectors: List[List[float]] = []
        for item in data:
            if not isinstance(item, list) or not item:
                raise EmbeddingServiceError(
                    "Unexpected embedding item in remote inference response"
                )
            if isinstance(item[0], list):
                raise EmbeddingServiceError(
                    "Endpoint returned token-level embeddings; Artemis requires "
                    "one sentence vector per input."
                )
            vectors.append([float(x) for x in item])
        return vectors

    async def embed(self, texts: List[str], dimension: int = 768) -> List[List[float]]:
        full = await self._request_full_embeddings(texts)
        return [truncate_and_normalize(vector, dimension) for vector in full]


embedding_service = RemoteEmbeddingService()
