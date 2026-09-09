# Artemis Embed v1

**Artemis Embed v1** is an English general-purpose dense embedding model built on `answerdotai/ModernBERT-base`. It produces one L2-normalized vector per text and is designed for semantic similarity, semantic retrieval, RAG retrieval, paraphrase matching, clustering, and classification features.

The current checkpoint uses **masked mean pooling**, **LoRA fine-tuning**, **contrastive training with hard negatives**, and **Matryoshka Representation Learning** with output dimensions **768 / 512 / 256 / 128**.

## Current model

| Property | Value |
|---|---|
| Backbone | `answerdotai/ModernBERT-base` |
| Hidden size | 768 |
| Pooling | Masked mean |
| Training update | LoRA |
| Training objective | Contrastive + hard-negative refinement + Matryoshka |
| Output dimensions | 768 / 512 / 256 / 128 |
| Normalization | L2 after truncation |
| Language | English |

### Development results

These are **internal development metrics from the current training run**, not a final MTEB claim.

| Experiment | R@1 | R@5 | MRR | STS Spearman | Classification | Clustering NMI | Aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|
| **LoRA + MRL + Hard Negatives** | **0.8675** | **0.9700** | **0.9095** | **0.3450** | 0.872 | **0.5897** | **0.7024** |
| LoRA + MRL + GOR + Teacher | 0.8575 | 0.9600 | 0.9018 | 0.3247 | **0.874** | 0.5492 | 0.6880 |
| LoRA + MRL + GOR | 0.8425 | 0.9600 | 0.8931 | 0.3218 | 0.866 | 0.5661 | 0.6832 |
| QLoRA + MRL + GOR | 0.8325 | 0.9500 | 0.8860 | 0.3004 | 0.868 | 0.5240 | 0.6683 |
| Full FT pilot | 0.7675 | 0.8900 | 0.8234 | 0.2803 | 0.872 | 0.5567 | 0.6433 |

The strongest tested recipe is therefore the **LoRA + Matryoshka + hard-negative** checkpoint. GOR and QLoRA were tested and were not kept in the current release path. Teacher alignment showed a mixed improvement but was not a clean enough ablation to replace the winning checkpoint.

### Matryoshka quality

| Dimension | R@1 | R@5 | MRR | STS Spearman |
|---:|---:|---:|---:|---:|
| 768 | 0.8675 | 0.9700 | 0.9095 | 0.3450 |
| 512 | 0.8550 | 0.9650 | 0.9015 | 0.3441 |
| 256 | 0.8275 | 0.9525 | 0.8833 | 0.3309 |
| 128 | 0.8250 | 0.9450 | 0.8763 | 0.3299 |

The web application's document index uses **256-dimensional vectors** by default. Relative to 768D float32 vectors, this reduces raw vector storage by about **66.7%** while retaining useful retrieval quality in the development suite.

## Repository architecture

```text
Browser (HTML / CSS / JavaScript)
              │
              │ HTTP + JSON
              ▼
        FastAPI on Vercel
          │          │
          │          └──────────────► Supabase
          │                           Storage + Postgres + pgvector
          ▼
Hugging Face inference
Artemis Embed v1
```

The Vercel function **does not load PyTorch or ModernBERT weights**. It calls the deployed Artemis embedding endpoint on Hugging Face, then performs dimension truncation, L2 normalization, cosine similarity, ranking, and Supabase orchestration. This keeps the Vercel deployment lightweight and avoids loading a 149M-parameter model on every serverless cold start.

## Web application

The application contains three model labs:

1. **Sentence Similarity** — compare two texts with cosine similarity at 768/512/256/128 dimensions.
2. **Semantic Search** — rank a small set of user-provided documents against a query.
3. **Document Retrieval** — upload PDF/TXT/MD files to Supabase, chunk them, embed each chunk at 256D, and retrieve the top semantic chunks with pgvector.

Artemis Embed v1 is an embedding model, not a generative LLM. The document feature returns the most relevant chunks; it does not invent or generate an answer.

## API

When running, FastAPI exposes interactive documentation at `/docs` and `/redoc`.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/health` | Service health |
| GET | `/api/model` | Model/deployment metadata |
| POST | `/api/embed` | Create embeddings |
| POST | `/api/similarity` | Cosine similarity between two texts |
| POST | `/api/search` | Semantic search over supplied documents |
| POST | `/api/documents/upload` | Upload, chunk, embed, and index a document |
| POST | `/api/documents/query` | Retrieve top chunks from an indexed document |
| DELETE | `/api/documents/{document_id}` | Delete an indexed document |

## Local development

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000`.

## Environment variables

```bash
HF_MODEL_ID=your-hf-username/artemis-embed-v1
HF_EMBEDDING_URL=
HF_TOKEN=hf_...

SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_BUCKET=artemis-documents

MAX_DOCUMENT_BYTES=5242880
MAX_DOCUMENT_CHUNKS=100
DOCUMENT_EMBEDDING_DIMENSION=256
```

`HF_EMBEDDING_URL` is optional. If it is set, the API uses it directly (for example a dedicated Hugging Face Inference Endpoint). Otherwise the service constructs the Hugging Face Inference API URL from `HF_MODEL_ID`.

Never expose `HF_TOKEN` or `SUPABASE_SERVICE_ROLE_KEY` to browser JavaScript. Store them only in local `.env` files and Vercel Environment Variables.

## Supabase setup

Run:

```text
supabase/migrations/001_documents.sql
```

in the Supabase SQL editor or through the Supabase CLI. It enables `pgvector`, creates document/chunk tables, creates a 256D HNSW cosine index, and defines the `match_document_chunks` RPC used by FastAPI.

## Hugging Face release

The Kaggle export contains the LoRA adapter and tokenizer. To create a standalone Sentence Transformers model, merge the adapter into ModernBERT and publish the result:

```bash
pip install -r requirements-model.txt
export HF_TOKEN=hf_...
export HF_REPO_ID=your-hf-username/artemis-embed-v1
python scripts/export_huggingface.py --artifact-dir /path/to/final_export/model --push
```

The export script builds:

```text
ModernBERT-base
  + LoRA adapter
  -> merge_and_unload()
  -> masked mean pooling
  -> Normalize()
  -> Sentence Transformers package
```

After publishing, configure the Hugging Face endpoint URL/token in Vercel.

## Vercel deployment

The repository is configured as a FastAPI Vercel project. `vercel.json` gives the FastAPI entrypoint a longer function duration for document ingestion.

After importing this GitHub repository into Vercel, set the environment variables above and deploy. The same deployment serves both the FastAPI routes and the static frontend.

## Training notebook and reports

- `notebooks/Artemis_Embed_v1_Training.ipynb` — end-to-end training/evaluation notebook used for the current checkpoint.
- `reports/experiment_results.csv` — experiment comparison.
- `reports/matryoshka_quality.csv` — dimension-specific evaluation.
- `reports/data_manifest.json` — data sources included in the current training run.

## Research status and limitations

The current checkpoint is a strong **v1 development checkpoint**, but the internal dev suite is not the final research benchmark. The research specification still requires a strict train/eval overlap audit and held-out `MTEB(eng, v2)` evaluation before making broad benchmark or SOTA claims.

The Full Fine-Tuning and LoRA experiments also used different training recipes/step budgets, so the current experiment table does **not** prove that LoRA is inherently better than full fine-tuning. It only identifies the strongest configuration tested in this run.

## Project status

- [x] ModernBERT embedding architecture
- [x] pooling ablation
- [x] contrastive training
- [x] hard-negative refinement
- [x] LoRA / QLoRA comparison
- [x] GOR ablation
- [x] teacher-alignment experiment
- [x] Matryoshka dimensions 768/512/256/128
- [x] FastAPI application
- [x] semantic similarity/search frontend
- [x] Supabase document retrieval integration
- [x] Vercel deployment configuration
- [ ] publish standalone model to Hugging Face
- [ ] configure production Hugging Face inference endpoint
- [ ] run final MTEB English v2 evaluation
- [ ] complete final data-license/leakage audit

---

Artemis Embed v1 is Mission 01 of the broader Artemis research program.
