#!/usr/bin/env python
"""Merge the Artemis LoRA adapter into ModernBERT and export Sentence Transformers.

Expected artifact layout:
  ARTIFACT_DIR/
    backbone/adapter_config.json
    backbone/adapter_model.safetensors
    tokenizer/...
    artemis_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi
from peft import PeftModel
from sentence_transformers import SentenceTransformer, models
from transformers import AutoModel, AutoTokenizer


def build_export(artifact_dir: Path, output_dir: Path) -> None:
    meta = json.loads((artifact_dir / "artemis_config.json").read_text())
    base_model_id = meta.get("base_model", "answerdotai/ModernBERT-base")
    pooling = meta.get("pooling", "mean")
    if pooling != "mean":
        raise ValueError(f"Current release expects masked mean pooling, got {pooling!r}")

    print(f"Loading base model: {base_model_id}")
    base = AutoModel.from_pretrained(base_model_id)
    peft_model = PeftModel.from_pretrained(base, artifact_dir / "backbone")
    merged = peft_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir / "tokenizer")
    merged_dir = output_dir / "_merged_backbone"
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    transformer = models.Transformer(str(merged_dir), max_seq_length=meta.get("max_length", 128))
    pooling_module = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=False,
        pooling_mode_mean_tokens=True,
        pooling_mode_max_tokens=False,
    )
    sentence_model = SentenceTransformer(
        modules=[transformer, pooling_module, models.Normalize()]
    )

    final_dir = output_dir / "sentence_transformers"
    sentence_model.save_pretrained(final_dir)

    project_root = Path(__file__).resolve().parents[1]
    card = project_root / "huggingface" / "README.md"
    if card.exists():
        shutil.copy2(card, final_dir / "README.md")
    shutil.copy2(artifact_dir / "artemis_config.json", final_dir / "artemis_config.json")

    shutil.rmtree(merged_dir)
    print(f"Sentence Transformers export: {final_dir}")


def push_export(export_dir: Path, repo_id: str, token: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        commit_message="Publish Artemis Embed v1",
    )
    print(f"Published: https://huggingface.co/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artemis_hf_export"))
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", ""))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_export(args.artifact_dir, args.output_dir)
    final_dir = args.output_dir / "sentence_transformers"

    if args.push:
        token = os.getenv("HF_TOKEN", "")
        if not token:
            raise SystemExit("HF_TOKEN is required with --push")
        if not args.repo_id:
            raise SystemExit("Set HF_REPO_ID or pass --repo-id with --push")
        push_export(final_dir, args.repo_id, token)


if __name__ == "__main__":
    main()
