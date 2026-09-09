from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import quote
import httpx

from artemis_api.config import settings


class SupabaseServiceError(RuntimeError):
    pass


class SupabaseStore:
    def __init__(self) -> None:
        self.base_url = settings.supabase_url
        self.key = settings.supabase_service_role_key
        self.bucket = settings.supabase_bucket
        self.timeout = 30.0

    def _headers(self, *, json_content: bool = True) -> Dict[str, str]:
        if not self.base_url or not self.key:
            raise SupabaseServiceError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
        }
        if json_content:
            headers["Content-Type"] = "application/json"
        return headers

    async def _json_request(self, method: str, path: str, *, payload: Any = None, prefer: str = "") -> Any:
        headers = self._headers()
        if prefer:
            headers["Prefer"] = prefer
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.request(
                method, f"{self.base_url}{path}", headers=headers, json=payload
            )
        if response.status_code >= 400:
            raise SupabaseServiceError(
                f"Supabase returned {response.status_code}: {response.text[:1000]}"
            )
        if not response.content:
            return None
        return response.json()

    async def create_document(self, name: str, storage_path: str, mime_type: str, size_bytes: int) -> Dict[str, Any]:
        rows = await self._json_request(
            "POST",
            "/rest/v1/documents",
            payload={
                "name": name,
                "storage_path": storage_path,
                "mime_type": mime_type,
                "size_bytes": size_bytes,
                "status": "processing",
            },
            prefer="return=representation",
        )
        return rows[0]

    async def update_document(self, document_id: str, **fields: Any) -> None:
        await self._json_request(
            "PATCH",
            f"/rest/v1/documents?id=eq.{quote(document_id)}",
            payload=fields,
            prefer="return=minimal",
        )

    async def insert_chunks(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        await self._json_request(
            "POST", "/rest/v1/document_chunks", payload=rows, prefer="return=minimal"
        )

    async def match_chunks(self, query_embedding: List[float], top_k: int, document_id: Optional[str]) -> List[Dict[str, Any]]:
        result = await self._json_request(
            "POST",
            "/rest/v1/rpc/match_document_chunks",
            payload={
                "query_embedding": query_embedding,
                "match_count": top_k,
                "filter_document_id": document_id,
            },
        )
        return result or []

    async def upload_file(self, storage_path: str, content: bytes, content_type: str) -> None:
        headers = self._headers(json_content=False)
        headers["Content-Type"] = content_type or "application/octet-stream"
        headers["x-upsert"] = "false"
        path = quote(storage_path, safe="/")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/storage/v1/object/{quote(self.bucket)}/{path}",
                headers=headers,
                content=content,
            )
        if response.status_code >= 400:
            raise SupabaseServiceError(
                f"Supabase Storage returned {response.status_code}: {response.text[:1000]}"
            )

    async def delete_file(self, storage_path: str) -> None:
        await self._json_request(
            "DELETE",
            f"/storage/v1/object/{quote(self.bucket)}",
            payload={"prefixes": [storage_path]},
        )

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        rows = await self._json_request(
            "GET",
            f"/rest/v1/documents?id=eq.{quote(document_id)}&select=*",
        )
        return rows[0] if rows else None

    async def delete_document(self, document_id: str) -> None:
        await self._json_request(
            "DELETE",
            f"/rest/v1/documents?id=eq.{quote(document_id)}",
            prefer="return=minimal",
        )


supabase_store = SupabaseStore()
