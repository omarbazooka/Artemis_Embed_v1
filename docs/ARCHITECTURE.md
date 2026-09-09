# Product Architecture

## Goal

Serve Artemis Embed v1 as a real web/API system without loading the 149M-parameter PyTorch model inside Vercel serverless functions.

## Runtime path

```text
Browser
  │
  ├── sentence similarity ──────┐
  ├── semantic search ──────────┤
  └── document retrieval ───────┤
                                ▼
                         FastAPI / Vercel
                         │              │
                         │              └─ Supabase Storage
                         │                 + Postgres / pgvector
                         ▼
                  Hugging Face endpoint
                  Artemis Embed v1
```

## Embedding shapes

The hosted Sentence Transformers model returns:

```text
texts: List[str], B texts
    ↓
full embedding: [B, 768]
```

For a requested Matryoshka dimension `D`:

```text
[B,768] -> [:,:D] -> [B,D] -> L2 normalize -> [B,D]
```

For normalized embeddings `q` and `d`, cosine similarity is the dot product:

```text
cos(q,d) = q · d
```

because `||q||₂ = ||d||₂ = 1`.

## Document retrieval

The persistent document index uses `D=256`:

```text
file
 -> text extraction
 -> word chunks (~120 words, ~20 overlap)
 -> Artemis 256D embeddings
 -> Supabase document_chunks.embedding vector(256)
 -> HNSW cosine index
```

A query follows:

```text
query -> Artemis [256] -> match_document_chunks() -> top-k chunks
```

The current synchronous ingestion route is intentionally limited to 5 MB and 100 chunks. Larger production workloads should move extraction/embedding into a queue/background worker rather than extending a single Vercel request indefinitely.

## Security boundary

The browser never receives:

- `HF_TOKEN`
- `SUPABASE_SERVICE_ROLE_KEY`

Those variables exist only in the FastAPI server environment. Supabase tables have RLS enabled with no anonymous policies; the server-side service-role request is the access boundary for the current v1 application.
