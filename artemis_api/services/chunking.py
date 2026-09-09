from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str


def chunk_text(text: str, chunk_words: int = 120, overlap_words: int = 20) -> List[TextChunk]:
    words = text.split()
    if not words:
        return []
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be >= 0 and smaller than chunk_words")

    chunks: List[TextChunk] = []
    step = chunk_words - overlap_words
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(TextChunk(index=idx, text=piece))
            idx += 1
        if end >= len(words):
            break
        start += step
    return chunks
