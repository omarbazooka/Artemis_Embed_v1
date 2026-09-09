from dataclasses import dataclass
import os


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    hf_model_id: str = os.getenv("HF_MODEL_ID", "")
    hf_embedding_url: str = os.getenv("HF_EMBEDDING_URL", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_request_timeout_seconds: float = _float("HF_REQUEST_TIMEOUT_SECONDS", 45.0)

    supabase_url: str = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_service_role_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    supabase_bucket: str = os.getenv("SUPABASE_BUCKET", "artemis-documents")

    max_document_bytes: int = _int("MAX_DOCUMENT_BYTES", 5 * 1024 * 1024)
    max_document_chunks: int = _int("MAX_DOCUMENT_CHUNKS", 100)
    document_embedding_dimension: int = _int("DOCUMENT_EMBEDDING_DIMENSION", 256)
    chunk_words: int = _int("CHUNK_WORDS", 120)
    chunk_overlap_words: int = _int("CHUNK_OVERLAP_WORDS", 20)

    @property
    def hf_url(self) -> str:
        if self.hf_embedding_url:
            return self.hf_embedding_url.rstrip("/")
        if self.hf_model_id:
            return f"https://router.huggingface.co/hf-inference/models/{self.hf_model_id}"
        return ""

    @property
    def model_configured(self) -> bool:
        return bool(self.hf_url)

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)


settings = Settings()
