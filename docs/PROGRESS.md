# Progress Log — Productization

## Current state
- Current milestone: Milestone 11 — productization and live integration
- Current task: Vercel + Kaggle inference + Hugging Face registry + Supabase integration
- Status: standalone model published, Kaggle inference verified at 768D, Supabase retrieval backend active, and Vercel routing live; final Vercel-to-Kaggle and document end-to-end smoke tests remain
- Current checkpoint: `lora_mrl_hardneg`
- Current config: masked mean, LoRA, MRL [768,512,256,128], hard negatives
- Current dataset version: current development mix; see `reports/data_manifest.json`
- Current phase location: productization / release engineering

## Completed
- Development experiment table reviewed and winner frozen for the current v1 application.
- FastAPI embedding/similarity/search API implemented.
- Static HTML/CSS/JavaScript application implemented.
- Hugging Face merge/export and parity-verification scripts implemented.
- Standalone Sentence Transformers release published at `Omarbm52/Artemis-Embed-v1`.
- Remote Hugging Face repository verified to contain merged `model.safetensors`, `modules.json`, Sentence Transformers configuration, tokenizer files, Artemis metadata, and model card.
- Hugging Face metadata resolves the release as a 149M-parameter ModernBERT Sentence Transformers feature-extraction model.
- Hugging Face `hf-inference` authentication was fixed, then the provider reported that the custom Artemis model is not supported by that provider.
- Kaggle was selected as the current development inference runtime to avoid a paid dedicated Hugging Face endpoint; this changes hosting only, not the model architecture or training decisions.
- Kaggle loaded `Omarbm52/Artemis-Embed-v1` successfully and returned two normalized 768D embeddings in the local smoke test.
- A temporary Cloudflare quick tunnel exposed the Kaggle `/embed` endpoint for Vercel integration.
- `scripts/kaggle_inference_server.py`, `requirements-kaggle-inference.txt`, and `docs/KAGGLE_INFERENCE.md` added for reproducibility.
- Vercel embedding-service messages and model status made provider-agnostic; `/api/health` and `/api/model` now expose the selected inference mode without exposing the endpoint URL.
- Supabase project `Artemis Embed v1` confirmed `ACTIVE_HEALTHY`.
- Supabase `vector` + `pgcrypto` extensions enabled.
- `documents` and `document_chunks` tables created with RLS enabled.
- Private `artemis-documents` Storage bucket created.
- 256D HNSW cosine index created for document chunks.
- `match_document_chunks` RPC created and restricted to the server-side `service_role`.
- `002_secure_match_rpc.sql` added to set an explicit function search path and restrict RPC execution.
- pgvector retrieval smoke test passed: a unit-aligned 256D query ranked the aligned chunk with cosine score 1.0 above an orthogonal chunk with score 0.0; smoke-test rows were removed afterward.
- Vercel project `artemis-embed-v1` created in the Artemis team and production build completed as a Python FastAPI Lambda.
- Live routing verified with HTTP 200 for `/`, `/api/health`, `/api/model`, and `/docs` before the Kaggle integration deploy.

## Runtime contract

```text
Kaggle Artemis runtime: List[str] -> normalized [B,768]
Vercel Matryoshka layer: [B,768] -> [:,:D] -> L2 normalize -> [B,D]
D in {768,512,256,128}
Supabase document index: D=256
```

## Next exact task
1. Redeploy Vercel after setting the current `HF_EMBEDDING_URL` to the live Kaggle `/embed` tunnel URL.
2. Verify `/api/model` reports `inference_mode=custom_endpoint`.
3. Run live `/api/embed`, `/api/similarity`, and `/api/search` smoke tests through Vercel.
4. Upload a small TXT/PDF and verify 256D pgvector document retrieval end-to-end.
5. If the Kaggle session/tunnel restarts, update `HF_EMBEDDING_URL` in Vercel and redeploy; do not retrain or re-export the model.

## Integration note
The Vercel connector can deploy and inspect projects but does not expose an environment-variable write action. The temporary Kaggle tunnel URL therefore has to be entered in the Vercel project settings as `HF_EMBEDDING_URL`; secrets must not be sent through chat or committed to source control.

Kaggle sessions and Cloudflare quick tunnels are ephemeral and have no uptime guarantee. This is the current development/live-demo inference path, not the final persistent hosting design.

## Research limitation
The current development checkpoint is not the final research-grade MTEB English v2 release. Final benchmark and leakage/license audit remain open.
