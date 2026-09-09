from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from artemis_api import __version__
from artemis_api.config import settings
from artemis_api.schemas import EmbedRequest, SimilarityRequest, SearchRequest, DocumentQueryRequest
from artemis_api.services.embeddings import embedding_service, EmbeddingServiceError
from artemis_api.services.vector_math import cosine_similarity
from artemis_api.services.chunking import chunk_text
from artemis_api.services.document_parser import extract_text, SUPPORTED_EXTENSIONS
from artemis_api.services.supabase_store import supabase_store, SupabaseServiceError

app = FastAPI(
    title="Artemis Embed v1 API",
    version=__version__,
    description="Dense embedding, semantic similarity, semantic search, and document retrieval API.",
)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "Artemis Embed v1",
        "version": __version__,
        "model_configured": settings.model_configured,
        "documents_enabled": settings.supabase_configured and settings.model_configured,
    }


@app.get("/api/model")
async def model_info():
    return {
        "name": "Artemis Embed v1",
        "backbone": "answerdotai/ModernBERT-base",
        "pooling": "masked_mean",
        "dimensions": [768, 512, 256, 128],
        "normalization": "L2 after Matryoshka truncation",
        "training_recipe": "LoRA + contrastive + hard negatives + Matryoshka",
        "hf_model_id": settings.hf_model_id or None,
        "model_configured": settings.model_configured,
        "documents_enabled": settings.supabase_configured and settings.model_configured,
        "document_dimension": settings.document_embedding_dimension,
    }


@app.post("/api/embed")
async def embed(request: EmbedRequest):
    try:
        vectors = await embedding_service.embed(request.texts, request.dimension)
        return {"dimension": request.dimension, "count": len(vectors), "embeddings": vectors}
    except EmbeddingServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/similarity")
async def similarity(request: SimilarityRequest):
    try:
        vectors = await embedding_service.embed(
            [request.text_a.strip(), request.text_b.strip()], request.dimension
        )
        return {
            "similarity": cosine_similarity(vectors[0], vectors[1]),
            "dimension": request.dimension,
        }
    except (EmbeddingServiceError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/search")
async def search(request: SearchRequest):
    documents = [text.strip() for text in request.documents if text.strip()]
    if not documents:
        raise HTTPException(status_code=422, detail="at least one non-empty document is required")
    top_k = min(request.top_k, len(documents))
    try:
        vectors = await embedding_service.embed(
            [request.query.strip(), *documents], request.dimension
        )
        query_vector = vectors[0]
        scored = [
            {
                "index": i,
                "score": cosine_similarity(query_vector, vector),
                "text": documents[i],
            }
            for i, vector in enumerate(vectors[1:])
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        results = [{"rank": rank, **item} for rank, item in enumerate(scored[:top_k], start=1)]
        return {"dimension": request.dimension, "query": request.query, "results": results}
    except (EmbeddingServiceError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return name[:160] or "document"


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not settings.supabase_configured:
        raise HTTPException(status_code=503, detail="Supabase document storage is not configured")
    if not settings.model_configured:
        raise HTTPException(status_code=503, detail="Hugging Face model inference is not configured")

    filename = _safe_filename(file.filename or "document")
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="supported document types are PDF, TXT, and MD")

    content = await file.read(settings.max_document_bytes + 1)
    if len(content) > settings.max_document_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"document exceeds the {settings.max_document_bytes} byte limit",
        )

    storage_path = f"{uuid.uuid4()}/{filename}"
    document = None
    try:
        await supabase_store.upload_file(
            storage_path, content, file.content_type or "application/octet-stream"
        )
        document = await supabase_store.create_document(
            filename, storage_path, file.content_type or "application/octet-stream", len(content)
        )

        text = extract_text(filename, content)
        chunks = chunk_text(text, settings.chunk_words, settings.chunk_overlap_words)
        if not chunks:
            raise ValueError("the document did not produce any text chunks")
        chunks = chunks[: settings.max_document_chunks]

        rows = []
        batch_size = 24
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = await embedding_service.embed(
                [chunk.text for chunk in batch], settings.document_embedding_dimension
            )
            for chunk, vector in zip(batch, vectors):
                rows.append(
                    {
                        "document_id": document["id"],
                        "chunk_index": chunk.index,
                        "content": chunk.text,
                        "embedding": vector,
                    }
                )

        for start in range(0, len(rows), 25):
            await supabase_store.insert_chunks(rows[start : start + 25])

        await supabase_store.update_document(
            document["id"], status="ready", chunk_count=len(rows)
        )
        return {
            "document_id": document["id"],
            "name": filename,
            "status": "ready",
            "chunk_count": len(rows),
            "dimension": settings.document_embedding_dimension,
        }
    except (ValueError, EmbeddingServiceError, SupabaseServiceError) as exc:
        if document:
            try:
                await supabase_store.update_document(document["id"], status="error")
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/documents/query")
async def query_document(request: DocumentQueryRequest):
    if not settings.supabase_configured:
        raise HTTPException(status_code=503, detail="Supabase document storage is not configured")
    try:
        query_vector = (
            await embedding_service.embed(
                [request.query.strip()], settings.document_embedding_dimension
            )
        )[0]
        rows = await supabase_store.match_chunks(
            query_vector, request.top_k, request.document_id
        )
        return {
            "query": request.query,
            "document_id": request.document_id,
            "dimension": settings.document_embedding_dimension,
            "results": rows,
        }
    except (EmbeddingServiceError, SupabaseServiceError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    if not settings.supabase_configured:
        raise HTTPException(status_code=503, detail="Supabase document storage is not configured")
    try:
        document = await supabase_store.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="document not found")
        await supabase_store.delete_document(document_id)
        try:
            await supabase_store.delete_file(document["storage_path"])
        except SupabaseServiceError:
            pass
        return {"deleted": True, "document_id": document_id}
    except SupabaseServiceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


PUBLIC_DIR = Path(__file__).resolve().parent / "public"
if PUBLIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="frontend")
