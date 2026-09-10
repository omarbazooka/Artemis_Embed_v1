from __future__ import annotations

from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = "Omarbm52/Artemis-Embed-v1"
EMBEDDING_DIMENSION = 768

print(f"Loading {MODEL_ID}...")
model = SentenceTransformer(MODEL_ID, device="cpu")
print("Artemis Embed v1 loaded.")

app = FastAPI(title="Artemis Embed v1 — Kaggle Inference")


class InferenceRequest(BaseModel):
    inputs: Union[str, List[str]]
    options: Optional[dict] = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dimension": EMBEDDING_DIMENSION,
    }


@app.post("/embed")
def embed(request: InferenceRequest):
    texts = [request.inputs] if isinstance(request.inputs, str) else request.inputs
    vectors = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
