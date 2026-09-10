# Product Architecture

## Goal

Serve Artemis Embed v1 as a real web/API system without loading the 149M-parameter PyTorch model inside Vercel serverless functions.

Hugging Face is the canonical model registry. The current development runtime loads that published model inside Kaggle and exposes a temporary HTTP embedding endpoint through a Cloudflare quick tunnel. Vercel calls that endpoint through `HF_EMBEDDING_URL`.

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
                  HF_EMBEDDING_URL
                         │
                         ▼
                Cloudflare quick tunnel
                         │
                         ▼
                    Kaggle runtime
                         │
                         ▼
          Omarbm52/Artemis-Embed-v1
              from Hugging Face Hub
```

The direct `hf-inference` fallback is retained in configuration for compatible hosted models, but the current Artemis release is not supported by that provider. For the current deployment, `HF_EMBEDDING_URL` must therefore point to the Kaggle-hosted `/embed` route.

## Embedding shapes

The Kaggle Artemis runtime returns:

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

The Kaggle runtime already returns normalized 768D vectors. Vercel still re-normalizes after truncation because the norm of the truncated prefix is generally no longer 1.

## Document retrieval

The persistent document index uses `D=256`:

```text
file
 -> text extraction
 -> word chunks (~120 words, ~20 overlap)
 -> Artemis full embedding [768]
 -> truncate to [256]
 -> L2 normalize
 -> Supabase document_chunks.embedding vector(256)
 -> HNSW cosine index
```

A query follows:

```text
query -> Artemis [768] -> truncate+normalize [256]
      -> match_document_chunks() -> top-k chunks
```

The current synchronous ingestion route is intentionally limited to 5 MB and 100 chunks. Larger production workloads should move extraction/embedding into a queue/background worker rather than extending a single Vercel request indefinitely.

## Runtime configuration

```text
HF_MODEL_ID=Omarbm52/Artemis-Embed-v1
HF_EMBEDDING_URL=https://<temporary-tunnel>.trycloudflare.com/embed
```

`HF_EMBEDDING_URL` takes precedence over the Hugging Face router URL derived from `HF_MODEL_ID`.

The current Kaggle endpoint does not require `HF_TOKEN`; the model itself is public. Keep `HF_TOKEN` only if a future remote inference endpoint requires authentication.

## Security boundary

The browser never receives:

- `HF_TOKEN`
- `SUPABASE_SERVICE_ROLE_KEY`

Those variables exist only in the FastAPI server environment. Supabase tables have RLS enabled with no anonymous policies; the server-side service-role request is the access boundary for the current v1 application.

The temporary Cloudflare URL is not treated as a secret, but the Kaggle/Cloudflare runtime is not a hardened production service and has no uptime guarantee.

## Reproducible Kaggle runtime

- `scripts/kaggle_inference_server.py` contains the `/health` and `/embed` server.
- `requirements-kaggle-inference.txt` contains the runtime dependencies.
- `docs/KAGGLE_INFERENCE.md` contains the full startup and tunnel procedure.
