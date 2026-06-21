"""
chunk.py — split raw text into overlapping chunks.

Strategy used here: recursive splitting on natural boundaries
(paragraph -> sentence -> word) until each chunk is under the
target size, with a sliding overlap so an answer that straddles
two chunks doesn't get cut in half.

This is the "RecursiveCharacterTextSplitter" pattern, written from
scratch so you can explain every line of it in an interview.
"""
import re
from app.config import settings
from app.ingestion.extract import content_hash

SEPARATORS = ["\n\n", "\n", ". ", " "]  # try paragraph first, fall back to finer splits


def _split_on_separator(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split_on_separator(text, rest)
    return [piece for piece in text.split(sep) if piece.strip()]


def chunk_text(
    text: str,
    doc_metadata: dict,
    chunk_size: int = None,
    overlap: int = None,
) -> list[dict]:
    """
    Returns a list of chunk dicts:
    {chunk_id, text, doc_id, source, chunk_index, content_hash}
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    # Normalize whitespace first
    text = re.sub(r"[ \t]+", " ", text).strip()

    pieces = _split_on_separator(text, SEPARATORS)

    # Greedily pack pieces into chunks up to chunk_size (measured in words,
    # simple stand-in for tokens — swap for a real tokenizer in production)
    chunks: list[str] = []
    current_words: list[str] = []

    for piece in pieces:
        piece_words = piece.split()
        if len(current_words) + len(piece_words) > chunk_size and current_words:
            chunks.append(" ".join(current_words))
            # carry the overlap forward into the next chunk
            current_words = current_words[-overlap:] if overlap else []
        current_words.extend(piece_words)

    if current_words:
        chunks.append(" ".join(current_words))

    doc_id = doc_metadata["doc_id"]
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            "chunk_id": f"{doc_id}::chunk_{i}",
            "text": chunk,
            "doc_id": doc_id,
            "source": doc_metadata["source"],
            "chunk_index": i,
            "content_hash": content_hash(chunk),
        })
    return results
