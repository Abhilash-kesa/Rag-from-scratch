"""
scripts/ingest.py — the batch ingestion entry point.

This is the single-process stand-in for what would be an Airflow DAG
or Kafka consumer at real scale (see project discussion). Run it
whenever you add/change files in data/uploads/.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.ingestion.extract import extract_text, build_doc_metadata
from app.ingestion.chunk import chunk_text
from app.ingestion.embed import embed_chunks, get_embedding_model
from app.retrieval.vector_store import VectorStore


def main():
    data_dir = Path(settings.DATA_DIR)
    files = [f for f in data_dir.glob("*") if f.suffix.lower() in (".pdf", ".txt", ".md")]

    if not files:
        print(f"No files found in {data_dir}. Add .pdf/.txt/.md files and re-run.")
        return

    print(f"Found {len(files)} files to ingest.")

    all_chunks = []
    for file_path in files:
        print(f"  extracting: {file_path.name}")
        text = extract_text(str(file_path))
        metadata = build_doc_metadata(str(file_path))
        chunks = chunk_text(text, metadata)
        print(f"    -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"Embedding {len(all_chunks)} chunks (cached chunks are skipped automatically)...")
    manifest_path = Path(settings.INDEX_DIR) / "embedding_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    all_chunks = embed_chunks(all_chunks, manifest_path)

    print("Building FAISS index...")
    dim = get_embedding_model().get_sentence_embedding_dimension()
    store = VectorStore(dim)
    store.add(all_chunks)
    store.save(settings.INDEX_DIR)

    print(f"Done. Index saved to {settings.INDEX_DIR}/ ({len(all_chunks)} vectors).")


if __name__ == "__main__":
    main()
