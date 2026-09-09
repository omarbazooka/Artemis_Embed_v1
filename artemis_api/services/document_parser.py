from io import BytesIO
from pathlib import Path
from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("supported document types are PDF, TXT, and MD")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)
    else:
        text = content.decode("utf-8", errors="replace")

    text = "\n".join(line.rstrip() for line in text.splitlines())
    if not text.strip():
        raise ValueError("no extractable text was found in the document")
    return text.strip()
