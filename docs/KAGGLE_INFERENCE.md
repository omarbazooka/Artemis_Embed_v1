# Kaggle Inference Runtime

## Purpose

Hugging Face hosts the standalone `Omarbm52/Artemis-Embed-v1` model repository, but the free `hf-inference` provider does not currently serve this custom model. The current development deployment therefore uses Kaggle as the model runtime and exposes the notebook process through a temporary Cloudflare quick tunnel.

This is a development/live-demo path, not a permanent production hosting guarantee. The tunnel URL changes when the Kaggle session or tunnel restarts.

## Runtime contract

```text
List[str]
  -> Artemis SentenceTransformer
  -> normalized embeddings [B, 768]
  -> JSON List[List[float]]
```

The Vercel FastAPI layer is responsible for Matryoshka truncation and re-normalization:

```text
[B, 768] -> [:, :D] -> [B, D] -> L2 normalize
D in {768, 512, 256, 128}
```

## Start the Kaggle server

Enable Internet in the Kaggle notebook, then install:

```bash
pip install -r requirements-kaggle-inference.txt
```

If the repository is not cloned in the notebook, install the same packages directly and copy `scripts/kaggle_inference_server.py` into `/kaggle/working`.

Start the server:

```bash
python scripts/kaggle_inference_server.py
```

The local health endpoint is:

```text
http://127.0.0.1:8000/health
```

The embedding endpoint is:

```text
http://127.0.0.1:8000/embed
```

## Expose through a temporary tunnel

Download Cloudflare `cloudflared`, then run:

```bash
./cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

Cloudflare prints a temporary URL similar to:

```text
https://example.trycloudflare.com
```

Configure Vercel with:

```text
HF_EMBEDDING_URL=https://example.trycloudflare.com/embed
```

`HF_EMBEDDING_URL` takes precedence over the fallback Hugging Face router URL built from `HF_MODEL_ID`.

## Smoke test

```python
import httpx

response = httpx.post(
    "http://127.0.0.1:8000/embed",
    json={"inputs": ["A dog runs in a park.", "A puppy is outside."]},
    timeout=120,
)
response.raise_for_status()
vectors = response.json()
assert len(vectors) == 2
assert len(vectors[0]) == 768
```

## Limitations

- Kaggle sessions are ephemeral and can stop.
- Cloudflare quick tunnels have no uptime guarantee.
- Every restarted tunnel can produce a new URL, requiring `HF_EMBEDDING_URL` to be updated in Vercel and the deployment redeployed.
- This path is suitable for development and demonstration. A permanent service should later move to a persistent inference runtime without changing the Artemis model architecture.
