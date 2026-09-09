---
language:
- en
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: answerdotai/ModernBERT-base
tags:
- sentence-transformers
- embeddings
- matryoshka
- modernbert
- semantic-search
---

# Artemis Embed v1

Artemis Embed v1 is an English general-purpose dense embedding model built from `answerdotai/ModernBERT-base`.

## Architecture

```text
Text
 -> ModernBERT-base
 -> masked mean pooling
 -> sentence vector [768]
 -> Matryoshka prefix 768/512/256/128
 -> L2 normalization
```

The current checkpoint was selected from a development experiment suite. The winning tested recipe used LoRA, contrastive learning, hard negatives, and Matryoshka Representation Learning.

## Usage

```python
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F

model = SentenceTransformer("YOUR_USERNAME/artemis-embed-v1")
emb = model.encode(["first text", "second text"], normalize_embeddings=True)

emb256 = F.normalize(torch.tensor(emb[:, :256]), p=2, dim=-1).numpy()
```

## Supported dimensions

768, 512, 256, 128. Always normalize again after truncation.

## Development metrics

| Dimension | Retrieval R@1 | MRR | STS Spearman |
|---:|---:|---:|---:|
| 768 | 0.8675 | 0.9095 | 0.3450 |
| 512 | 0.8550 | 0.9015 | 0.3441 |
| 256 | 0.8275 | 0.8833 | 0.3309 |
| 128 | 0.8250 | 0.8763 | 0.3299 |

These are internal development results and must not be interpreted as final MTEB or SOTA claims.

## Limitations

- English-only v1 scope.
- The current internal dev suite is not the final MTEB English v2 benchmark.
- Final training/evaluation overlap audit is still required for research-grade release claims.
- The model produces embeddings, not generated answers.

## Base model

`answerdotai/ModernBERT-base`.
