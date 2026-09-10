# Progress Log — Productization

## Current state
- Current milestone: Milestone 11 — productization and live integration
- Current task: FastAPI + Vercel + Hugging Face + Supabase integration
- Status: standalone model published, Supabase retrieval backend active, and Vercel routing live; production secret wiring and end-to-end inference smoke tests remain
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
- Supabase project `Artemis Embed v1` confirmed `ACTIVE_HEALTHY`.
- Supabase `vector` + `pgcrypto` extensions enabled.
- `documents` and `document_chunks` tables created with RLS enabled.
- Private `artemis-documents` Storage bucket created.
- 256D HNSW cosine index created for document chunks.
- `match_document_chunks` RPC created and restricted to the server-side `service_role`.
- `002_secure_match_rpc.sql` added to set an explicit function search path and restrict RPC execution.
- pgvector retrieval smoke test passed: a unit-aligned 256D query ranked the aligned chunk with cosine score 1.0 above an orthogonal chunk with score 0.0; smoke-test rows were removed afterward.
- Supabase security/performance checks were run after the Artemis schema changes.
- Vercel project `artemis-embed-v1` created in the Artemis team and production build completed as a Python FastAPI Lambda.
- Live routing verified with HTTP 200 for `/`, `/api/health`, `/api/model`, and `/docs`.

## Next exact task
1. Configure Vercel `HF_MODEL_ID=Omarbm52/Artemis-Embed-v1` and a server-only `HF_TOKEN`, or configure a dedicated `HF_EMBEDDING_URL`.
2. Configure Vercel `SUPABASE_URL=https://tansaxtdjkdtmewtmuvu.supabase.co` and server-only `SUPABASE_SERVICE_ROLE_KEY`.
3. Redeploy after environment-variable changes.
4. Run live `/api/embed`, `/api/similarity`, and `/api/search` smoke tests.
5. Upload a small TXT/PDF and verify 256D pgvector document retrieval end-to-end.

## Integration note
The current Vercel connector can deploy and inspect projects but does not expose an environment-variable write action. The Supabase connector intentionally does not expose the service-role secret. Production secrets therefore need to be entered directly in the Vercel project settings rather than sent through chat or committed to source control.

## Research limitation
The current development checkpoint is not the final research-grade MTEB English v2 release. Final benchmark and leakage/license audit remain open.
