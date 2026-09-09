# Progress Log — Productization

## Current state
- Current milestone: Milestone 11 — Local productization and product demos
- Current task: FastAPI + Vercel + Hugging Face + Supabase integration
- Status: implementation prepared
- Current checkpoint: `lora_mrl_hardneg`
- Current config: masked mean, LoRA, MRL [768,512,256,128], hard negatives
- Current dataset version: current development mix; see `reports/data_manifest.json`
- Current phase location: productization / release engineering

## Completed
- Development experiment table reviewed and winner frozen for the current v1 application.
- FastAPI embedding/similarity/search API implemented.
- Static HTML/CSS/JavaScript application implemented.
- Supabase Storage + pgvector document retrieval schema implemented.
- Hugging Face merge/export script implemented.
- Vercel configuration implemented.

## Next exact task
1. Publish the merged Sentence Transformers package to Hugging Face.
2. Configure `HF_MODEL_ID` or `HF_EMBEDDING_URL` + `HF_TOKEN` in Vercel.
3. Create/apply the Supabase migration and set the Supabase service-role environment variables.
4. Deploy and run live similarity/search/document smoke tests.

## Research limitation
The current development checkpoint is not the final research-grade MTEB English v2 release. Final benchmark and leakage/license audit remain open.
