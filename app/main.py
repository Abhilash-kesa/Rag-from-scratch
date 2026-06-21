"""
main.py — the API surface. This is what gets containerized and deployed.

Flow per request:
  query -> retrieve (FAISS, top_k=20) -> rerank (cross-encoder, top_n=5)
        -> generate (LLM, structured JSON) -> return answer + citations
"""
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank
from app.generation.generator import generate_answer
from app.ingestion.embed import get_embedding_model

app = FastAPI(title="RAG Service")
app.mount("/static", StaticFiles(directory="static"), name="static")

_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        dim = get_embedding_model().get_sentence_embedding_dimension()
        index_path = Path(settings.INDEX_DIR) / "index.faiss"
        if not index_path.exists():
            raise HTTPException(
                status_code=503,
                detail="Index not built yet. Run `python scripts/ingest.py` first.",
            )
        _store = VectorStore.load(settings.INDEX_DIR, dim)
    return _store


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None
    top_n: int | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    retrieved_chunk_count: int


@app.get("/")
def root():
    return RedirectResponse(url="/static/chat.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    store = get_store()

    candidates = retrieve(req.question, store, top_k=req.top_k)
    top_chunks = rerank(req.question, candidates, top_n=req.top_n)
    result = generate_answer(req.question, top_chunks)

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        retrieved_chunk_count=len(candidates),
    )
