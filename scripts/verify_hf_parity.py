#!/usr/bin/env python
"""Compare adapter inference against the merged Sentence Transformers export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

TEXTS = [
    "A dog runs through a green park.",
    "A puppy is running outside in the park.",
    "Vector databases store dense numerical representations.",
]


def masked_mean(hidden, mask):
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    args = parser.parse_args()

    meta = json.loads((args.artifact_dir / "artemis_config.json").read_text())
    base = AutoModel.from_pretrained(meta["base_model"])
    adapter = PeftModel.from_pretrained(base, args.artifact_dir / "backbone").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.artifact_dir / "tokenizer")
    batch = tokenizer(
        TEXTS,
        padding=True,
        truncation=True,
        max_length=meta.get("max_length", 128),
        return_tensors="pt",
    )
    with torch.no_grad():
        raw = adapter(**batch).last_hidden_state
        expected = F.normalize(masked_mean(raw, batch["attention_mask"]), p=2, dim=-1)

    merged = SentenceTransformer(str(args.export_dir))
    actual = torch.tensor(merged.encode(TEXTS, normalize_embeddings=True))
    max_abs = (expected - actual).abs().max().item()
    print(f"max_abs_diff={max_abs:.8f}")
    if max_abs > 1e-4:
        raise SystemExit("Parity check failed")
    print("Parity check passed")


if __name__ == "__main__":
    main()
