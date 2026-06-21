"""
vector_store.py — thin wrapper around FAISS.

In production you'd swap this class for a Pinecone/Qdrant/Milvus client
(same interface: add(), search()) to get metadata filtering, sharding,
and managed scaling without touching any other file in the project.
"""
import json
from pathlib import Path
import numpy as np
import faiss


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product == cosine, since vectors are normalized
        self.metadata: list[dict] = []  # parallel array: metadata[i] describes index vector i

    def add(self, chunks: list[dict]):
        vectors = np.array([c["embedding"] for c in chunks], dtype="float32")
        self.index.add(vectors)
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            self.metadata.append(meta)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[dict]:
        query_vector = np.array([query_vector], dtype="float32")
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            result = dict(self.metadata[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def save(self, index_dir: str):
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, f"{index_dir}/index.faiss")
        Path(f"{index_dir}/metadata.json").write_text(json.dumps(self.metadata))

    @classmethod
    def load(cls, index_dir: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(f"{index_dir}/index.faiss")
        store.metadata = json.loads(Path(f"{index_dir}/metadata.json").read_text())
        return store
