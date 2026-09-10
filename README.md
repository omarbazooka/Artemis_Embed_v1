# Artemis Embed v1

**Artemis Embed v1** is an English general-purpose dense embedding model built on `answerdotai/ModernBERT-base`. It produces one L2-normalized vector per text for semantic similarity, semantic retrieval, RAG retrieval, paraphrase matching, clustering, and classification features.

The current release uses **masked mean pooling**, **LoRA fine-tuning**, **contrastive training with hard negatives**, and **Matryoshka Representation Learning** with output dimensions **768 / 512 / 256 / 128**.

- Hugging Face model: `Omarbm52/Artemis-Embed-v1`
- Live Vercel deployment: `https://artemis-embed-v1-kappa.vercel.app`
- FastAPI docs: `https://artemis-embed-v1-kappa.vercel.app/docs`

## Current model

| Property | Value |
|---|---|
| Backbone | `answerdotai/ModernBERT-base` |
| Parameters | ~149M |
| Hidden size | 768 |
| Pooling | Masked mean |
| Training update | LoRA |
| Training objective | Contrastive + hard-negative refinement + Matryoshka |
| Output dimensions | 768 / 512 / 256 / 128 |
| Normalization | L2 after truncation |
| Language | English |
| Tested max length | 128 tokens |

## Development results

These are **internal development metrics from the current training run**, not final MTEB results.

| Experiment | R@1 | R@5 | MRR | STS Spearman | Classification | Clustering NMI | Aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LoRA + MRL + Hard Negatives** | **0.8675** | **0.9700** | **0.9095** | **0.3450** | 0.872 | **0.5897** | **0.7024** |
| LoRA + MRL + GOR + Teacher | 0.8575 | 0.9600 | 0.9018 | 0.3247 | **0.874** | 0.5492 | 0.6880 |
| LoRA + MRL + GOR | 0.8425 | 0.9600 | 0.8931 | 0.3218 | 0.866 | 0.5661 | 0.6832 |
| QLoRA + MRL + GOR | 0.8325 | 0.9500 | 0.8860 | 0.3004 | 0.868 | 0.5240 | 0.6683 |
| Full FT pilot | 0.7675 | 0.8900 | 0.8234 | 0.2803 | 0.872 | 0.5567 | 0.6433 |

The strongest tested recipe in this development run is **LoRA + Matryoshka + hard negatives**. The training arms used different recipes and step budgets, so this table does not establish that LoRA is inherently better than full fine-tuning.

## Matryoshka quality

| Dimension | R@1 | R@5 | MRR | STS Spearman | float32 bytes/vector |
|---:|---:|---:|---:|---:|---:|
| 768 | 0.8675 | 0.9700 | 0.9095 | 0.3450 | 3072 |
| 512 | 0.8550 | 0.9650 | 0.9015 | 0.3441 | 2048 |
| 256 | 0.8275 | 0.9525 | 0.8833 | 0.3309 | 1024 |
| 128 | 0.8250 | 0.9450 | 0.8763 | 0.3299 | 512 |

After truncating a Matryoshka embedding, the vector must be **L2-normalized again**. The document index uses 256D vectors, reducing raw float32 vector storage by about 66.7% relative to 768D. Truncation reduces downstream vector storage, bandwidth, similarity-computation cost, and index size; it does not reduce ModernBERT backbone FLOPs.

## Current application architecture

```text
Browser
HTML / CSS / JavaScript
        │
        │ HTTP / JSON
        ▼
FastAPI on Vercel
    │            │
    │            └──────────────► Supabase
    │                             Storage + PostgreSQL + pgvector
    ▼
HF_EMBEDDING_URL
    │
    ▼
Cloudflare quick tunnel
    │
    ▼
Kaggle inference runtime
    │
    ▼
Omarbm52/Artemis-Embed-v1
Hugging Face model registry
```

The Vercel function does **not** load PyTorch or the 149M-parameter ModernBERT model. The current development runtime loads the published Sentence Transformers model on Kaggle and exposes a temporary `/embed` endpoint through a Cloudflare quick tunnel. Vercel calls that endpoint through `HF_EMBEDDING_URL`, then performs Matryoshka truncation, L2 normalization, cosine similarity, ranking, and Supabase orchestration.

The free Hugging Face `hf-inference` provider does not currently support this custom Artemis model. Hugging Face therefore remains the canonical model registry while Kaggle is the current development inference runtime. The Kaggle/Cloudflare path is temporary and has no uptime guarantee.

## Hugging Face release

The standalone model is published at:

`Omarbm52/Artemis-Embed-v1`

The release contains merged `model.safetensors` weights plus Sentence Transformers configuration and tokenizer files. Release flow:

```text
ModernBERT-base
  + LoRA adapter
  -> merge_and_unload()
  -> masked mean pooling
  -> Normalize()
  -> standalone SentenceTransformer
```

The merge/export code is in `scripts/export_huggingface.py`; `scripts/verify_hf_parity.py` compares the adapter path against the merged export before publication.

## Kaggle inference runtime

The repository includes a reproducible remote inference server:

```text
scripts/kaggle_inference_server.py
requirements-kaggle-inference.txt
docs/KAGGLE_INFERENCE.md
```

The runtime contract is:

```text
List[str]
  -> Artemis SentenceTransformer
  -> normalized [B, 768]
  -> JSON List[List[float]]
```

Vercel receives the full vector and applies the selected Matryoshka dimension:

```text
[B,768] -> [:,:D] -> [B,D] -> L2 normalize
D ∈ {768,512,256,128}
```

## Web application

The application supports three model workflows:

1. **Sentence Similarity** — compare two texts with cosine similarity at 768/512/256/128 dimensions.
2. **Semantic Search** — rank user-provided texts against a query.
3. **Document Retrieval** — upload PDF/TXT/MD files, chunk them, embed chunks at 256D, and retrieve the top semantic chunks with pgvector.

Artemis Embed v1 is an embedding model, not a generative LLM. Document retrieval returns relevant passages; it does not generate an answer.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Service health and inference mode |
| GET | `/api/model` | Model/deployment metadata |
| POST | `/api/embed` | Create embeddings |
| POST | `/api/similarity` | Cosine similarity between two texts |
| POST | `/api/search` | Semantic search over supplied texts |
| POST | `/api/documents/upload` | Upload, chunk, embed, and index a document |
| POST | `/api/documents/query` | Retrieve top chunks from an indexed document |
| DELETE | `/api/documents/{document_id}` | Delete an indexed document |

The production routing has been smoke-tested with HTTP 200 responses for `/`, `/api/health`, `/api/model`, and `/docs`.

## Environment variables

```bash
HF_MODEL_ID=Omarbm52/Artemis-Embed-v1
HF_EMBEDDING_URL=https://<temporary-tunnel>.trycloudflare.com/embed
HF_TOKEN=

SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_BUCKET=artemis-documents

MAX_DOCUMENT_BYTES=5242880
MAX_DOCUMENT_CHUNKS=100
DOCUMENT_EMBEDDING_DIMENSION=256
```

`HF_EMBEDDING_URL` takes precedence over the Hugging Face router URL derived from `HF_MODEL_ID`. The current public Kaggle endpoint does not require `HF_TOKEN`; keep a token only when a future remote endpoint requires authentication.

`SUPABASE_SERVICE_ROLE_KEY` is a server-only secret and must never be committed or exposed to browser JavaScript.

## Supabase retrieval backend

The Supabase project is configured with:

- `documents` and `document_chunks` tables;
- `extensions.vector(256)` embeddings;
- HNSW cosine index;
- private `artemis-documents` Storage bucket;
- `match_document_chunks` RPC;
- RLS on document tables;
- server-side-only RPC execution through `service_role`.

Apply migrations in order:

```text
supabase/migrations/001_documents.sql
supabase/migrations/002_secure_match_rpc.sql
```

A deterministic pgvector smoke test was run with a unit-aligned vector and an orthogonal vector: the aligned chunk returned cosine score 1.0 and ranked above the orthogonal chunk at 0.0.

## Local development

Python 3.12+ is recommended for the current repository configuration.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload
```

## Reports and documentation

- `reports/experiment_results.csv` — experiment comparison.
- `reports/matryoshka_quality.csv` — dimension-specific evaluation.
- `reports/data_manifest.json` — data sources included in the current development run.
- `reports/training_config.json` — exported training configuration.
- `docs/ARCHITECTURE.md` — application architecture.
- `docs/KAGGLE_INFERENCE.md` — current inference-runtime procedure.
- `docs/PROGRESS.md` — current exact implementation state.

## Research status and limitations

The published model is the current **v1 development release**, not a final research-grade benchmark release. A strict train/evaluation overlap audit and held-out `MTEB(eng, v2)` evaluation remain required before making broad benchmark or SOTA claims.

## Project status

- [x] ModernBERT embedding architecture
- [x] pooling ablation
- [x] contrastive training
- [x] hard-negative refinement
- [x] LoRA / QLoRA comparison
- [x] GOR ablation
- [x] teacher-alignment experiment
- [x] Matryoshka dimensions 768/512/256/128
- [x] standalone Sentence Transformers model published to Hugging Face
- [x] reproducible Kaggle inference server
- [x] FastAPI application
- [x] Supabase Storage + pgvector retrieval backend
- [x] Vercel production deployment and routing smoke test
- [ ] complete live Vercel -> Kaggle embedding/similarity/search smoke tests
- [ ] complete live document upload/query smoke test
- [ ] run final MTEB English v2 evaluation
- [ ] complete final data-license/leakage audit

---

Artemis Embed v1 is Mission 01 of the broader Artemis research program.
