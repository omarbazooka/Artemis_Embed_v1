# Progress Log — Productization

## Current state
- Current milestone: Milestone 11 — Local productization and product demos
- Current task: FastAPI + Vercel + Hugging Face + Supabase integration
- Status: Supabase backend active and smoke-tested; model publication and final Vercel wiring remain
- Current checkpoint: `lora_mrl_hardneg`
- Current config: masked mean, LoRA, MRL [768,512,256,128], hard negatives
- Current dataset version: current development mix; see `reports/data_manifest.json`
- Current phase location: productization / release engineering

## Completed
- Development experiment table reviewed and winner frozen for the current v1 application.
- FastAPI embedding/similarity/search API implemented.
- Static HTML/CSS/JavaScript application implemented.
- Hugging Face merge/export and parity-verification scripts implemented.
- Vercel configuration implemented.
- Supabase project `Artemis Embed v1` created and confirmed `ACTIVE_HEALTHY`.
- Supabase `vector` + `pgcrypto` extensions enabled.
- `documents` and `document_chunks` tables created with RLS enabled.
- Private `artemis-documents` Storage bucket created.
- 256D HNSW cosine index created for document chunks.
- `match_document_chunks` RPC created and restricted to the server-side `service_role`.
- Security advisor re-run after hardening; Artemis RPC search-path warning resolved.
- pgvector retrieval smoke test passed: a unit-aligned 256D query ranked the aligned chunk with cosine score 1.0 above an orthogonal chunk with score 0.0; smoke-test rows were removed afterward.
- Supabase performance advisor reports no current performance lints.
- Repository scan found no `portfolio`, `interview`, or `job fair` wording in the current project source.

## Next exact task
1. Publish the standalone Artemis Sentence Transformers package to Hugging Face.
2. Configure `HF_MODEL_ID` or `HF_EMBEDDING_URL` + `HF_TOKEN` in Vercel.
3. Configure `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` in Vercel.
4. Deploy and run live `/api/health`, similarity, semantic-search, upload, and pgvector retrieval smoke tests.

## External integration note
The connected Hugging Face MCP credential currently exposes read-repository and Jobs scopes, but not Hub write/upload scope; the attempted remote Job path also requires billed HF Jobs. The model export remains ready in `scripts/export_huggingface.py` pending an available Hub write path.

The current Vercel MCP connection is returning authorization failures when listing the project, so production environment-variable wiring cannot be completed through the connector until that connection has project access again.

## Research limitation
The current development checkpoint is not the final research-grade MTEB English v2 release. Final benchmark and leakage/license audit remain open.
