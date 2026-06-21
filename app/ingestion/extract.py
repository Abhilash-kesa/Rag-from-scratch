"""
extract.py — turn a source file into (raw_text, metadata).

This is the only file that needs to know about file formats.
Everything downstream just sees plain text + metadata, regardless
of whether it came from a .pdf, .txt, or (in production) a DB row.
"""
import hashlib
from pathlib import Path
from pypdf import PdfReader


def extract_text(file_path: str) -> str:
    """Dispatch by extension. Add more branches as you support more formats."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def content_hash(text: str) -> str:
    """Used for idempotent re-ingestion: only re-embed if this hash changed."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_doc_metadata(file_path: str) -> dict:
    path = Path(file_path)
    return {
        "doc_id": path.stem,
        "source": str(path),
        "filename": path.name,
    }
