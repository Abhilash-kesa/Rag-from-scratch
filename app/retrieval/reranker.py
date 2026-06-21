"""
reranker.py — stage 2: cross-encoder reranking.

Why this exists: the retriever's bi-encoder embeds query and chunk
SEPARATELY then compares vectors — fast, but loses interaction signal.
A cross-encoder feeds (query, chunk) into the model TOGETHER, so it can
attend to both at once and score relevance far more accurately. It's
too slow to run over the whole corpus, so we only run it on the
retriever's top_k candidates.
"""
from sentence_transformers import CrossEncoder
from app.config import settings

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(settings.RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_n: int = None) -> list[dict]:
    top_n = top_n or settings.TOP_N_RERANK
    if not candidates:
        return []

    model = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_n]
