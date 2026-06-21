"""
embed.py — turn chunks into vectors.

Idempotency: each chunk carries a content_hash (set in chunk.py).
We keep a manifest {content_hash: embedding} so re-running ingestion
on unchanged documents costs ~0 extra embedding calls — only new or
edited chunks get re-embedded. This is the "diff, not full rebuild"
pattern mentioned in the ingestion design.

Batching: sentence-transformers' .encode() natively batches, which is
where the real speedup comes from at scale (batch of 64 >> 64 single calls).
"""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def _load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.write_text(json.dumps(manifest))


def embed_chunks(chunks: list[dict], manifest_path: Path, batch_size: int = 64) -> list[dict]:
    """
    Returns chunks with an added 'embedding' field (as a list of floats).
    Skips re-embedding chunks whose content_hash is already in the manifest
    and reuses the cached vector instead.
    """
    manifest = _load_manifest(manifest_path)
    model = get_embedding_model()

    to_embed_idx = []
    to_embed_text = []

    for i, chunk in enumerate(chunks):
        cached = manifest.get(chunk["content_hash"])
        if cached is not None:
            chunk["embedding"] = cached
        else:
            to_embed_idx.append(i)
            to_embed_text.append(chunk["text"])

    if to_embed_text:
        vectors = model.encode(
            to_embed_text,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # so dot product == cosine similarity
        )
        for idx, vec in zip(to_embed_idx, vectors):
            vec_list = vec.tolist()
            chunks[idx]["embedding"] = vec_list
            manifest[chunks[idx]["content_hash"]] = vec_list

    _save_manifest(manifest_path, manifest)
    return chunks


def embed_query(query: str) -> np.ndarray:
    model = get_embedding_model()
    vec = model.encode([query], normalize_embeddings=True)[0]
    return vec
