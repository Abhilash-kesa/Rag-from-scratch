"""
retriever.py — query -> candidate chunks.

This is stage 1 of the two-stage retrieve-then-rerank pattern:
fast, approximate, over-fetches (top_k=20) so the reranker has
enough candidates to find the genuinely best ones from.
"""
from app.config import settings
from app.ingestion.embed import embed_query
from app.retrieval.vector_store import VectorStore


def retrieve(query: str, store: VectorStore, top_k: int = None) -> list[dict]:
    top_k = top_k or settings.TOP_K_RETRIEVE
    query_vec = embed_query(query)
    return store.search(query_vec, top_k=top_k)
