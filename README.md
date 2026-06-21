# RAG Service — End to End

A minimal but production-shaped Retrieval-Augmented Generation service:
ingest documents → chunk → embed → index → retrieve → rerank → generate a
cited answer. Includes a RAGAS evaluation harness and a Docker setup.

---

## 1. Architecture / Flow

```
INGESTION (offline, run via scripts/ingest.py)
─────────────────────────────────────────────
Documents (.pdf/.txt/.md in data/uploads/)
   │
   ▼
extract.py        file → raw text (+ doc metadata)
   │
   ▼
chunk.py           raw text → overlapping chunks (recursive splitting)
   │
   ▼
embed.py            chunks → vectors (sentence-transformers, hash-cached)
   │
   ▼
vector_store.py       vectors + metadata → FAISS index, saved to data/index/


QUERY (online, served via app/main.py)
─────────────────────────────────────────────
User question (POST /query)
   │
   ▼
retriever.py        embed query → ANN search FAISS → top-K candidates (e.g. 20)
   │
   ▼
reranker.py           cross-encoder scores (query, chunk) pairs → top-N (e.g. 5)
   │
   ▼
prompt.py               builds prompt: numbered chunks + JSON-answer contract
   │
   ▼
generator.py               calls the LLM → parses {answer, citations} →
                            resolves citation numbers back to real chunk_ids


EVALUATION (offline, run via eval/evaluate.py)
─────────────────────────────────────────────
eval/golden_dataset.json (question + ground_truth pairs)
   │
   ▼
Runs each question through the REAL retrieve → rerank → generate pipeline
   │
   ▼
RAGAS scores: faithfulness, answer_relevancy, context_precision, context_recall
```

---

## 2. Project Structure

```
rag-project/
│
├── app/                          # the deployable service
│   ├── main.py                    # FastAPI app — exposes /query and /health
│   ├── config.py                   # all settings, loaded from .env
│   │
│   ├── ingestion/                  # offline: turns files into indexed vectors
│   │   ├── extract.py               # file → raw text (pdf/txt/md), content hashing
│   │   ├── chunk.py                  # raw text → overlapping chunks
│   │   └── embed.py                   # chunks → embeddings, with hash-based caching
│   │                                    so re-ingestion only embeds NEW/CHANGED chunks
│   │
│   ├── retrieval/                  # online: question → relevant chunks
│   │   ├── vector_store.py          # FAISS wrapper: add / search / save / load
│   │   ├── retriever.py              # stage 1 — fast ANN search, over-fetches
│   │   └── reranker.py                # stage 2 — cross-encoder, precise re-scoring
│   │
│   └── generation/                 # online: chunks → final answer
│       ├── prompt.py                # builds the prompt (numbered chunks, JSON contract)
│       └── generator.py              # calls the LLM, parses + verifies citations
│
├── eval/                          # evaluation harness
│   ├── golden_dataset.json         # hand-written question/ground_truth pairs
│   ├── evaluate.py                  # runs pipeline on golden set, scores with RAGAS
│   └── last_run_results.json        # (generated) most recent eval run's scores
│
├── scripts/
│   └── ingest.py                  # CLI entry point: runs the full ingestion pipeline
│                                     (stand-in for an Airflow DAG / Kafka consumer at scale)
│
├── data/
│   ├── uploads/                   # drop source documents here (sample.txt included)
│   └── index/                      # (generated) FAISS index + embedding manifest live here
│
├── tests/                         # (placeholder for unit tests)
│
├── requirements.txt               # Python dependencies
├── Dockerfile                      # container image for app/main.py
├── docker-compose.yml               # runs ingest.py then starts the API, with a volume
│                                      so the index persists across container restarts
├── .env.example                     # template for required environment variables
└── README.md                        # this file
```

**Why it's organized this way:** `ingestion/` and `retrieval/` and `generation/`
are separated because they run at different times and scale independently —
ingestion is a batch job, retrieval+generation is a live request path. In a
real deployment you could swap `scripts/ingest.py` for an Airflow DAG without
touching anything in `app/retrieval/` or `app/generation/` at all.

---

## 3. Prerequisites

- Python 3.11+ (or Docker, if you don't want to install Python deps locally)
- An OpenAI API key (for the generation step) — get one at platform.openai.com
- ~2GB free disk space (for the embedding + reranker models, downloaded on first run)
- Internet access on first run (to download the sentence-transformers models from
  Hugging Face — after that, they're cached locally and no network is needed for
  embedding/reranking)

---

## 4. Setup & Run — Option A: Local (no Docker)

**Step 1 — clone/unzip and enter the project**
```bash
cd rag-project
```

**Step 2 — create a virtual environment and install dependencies**
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3 — configure environment variables**
```bash
cp .env.example .env
```
Open `.env` and set `OPENAI_API_KEY=your-actual-key`. The other values
(chunk size, top_k, model names) have sensible defaults — leave them as-is
for the first run.

**Step 4 — add documents (optional — a sample is already included)**
```bash
# data/uploads/sample.txt is already there (refund/shipping policy text).
# To add your own: drop .pdf, .txt, or .md files into data/uploads/
cp /path/to/your/document.pdf data/uploads/
```

**Step 5 — run ingestion**
```bash
python scripts/ingest.py
```
This extracts text, chunks it, embeds the chunks, and builds a FAISS index
under `data/index/`. First run downloads the embedding model (~80MB) and
takes a minute or two; subsequent runs are fast and skip already-embedded
chunks.

**Step 6 — start the API server**
```bash
uvicorn app.main:app --reload --port 8000
```

**Step 7 — query it**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days do I have to request a refund?"}'
```

Expected response shape:
```json
{
  "answer": "You have 30 days from the date of purchase to request a refund. [1]",
  "citations": [
    {"marker": 1, "chunk_id": "sample::chunk_0", "source": "data/uploads/sample.txt", "text_snippet": "Refund Policy\n\nCustomers may request..."}
  ],
  "retrieved_chunk_count": 1
}
```

You can also open `http://localhost:8000/docs` for the interactive FastAPI
Swagger UI to try queries without curl.

---

## 5. Setup & Run — Option B: Docker (recommended for a clean run)

**Step 1 — configure environment**
```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

**Step 2 — build and run**
```bash
docker compose up --build
```
This single command builds the image, runs `scripts/ingest.py` automatically
on container start (re-indexing `data/uploads/`), then starts the API on
`http://localhost:8000`. The `data/` directory is mounted as a volume, so the
index persists across container restarts — you don't re-embed from scratch
every time.

**Step 3 — query it** (same as above)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about shipping times?"}'
```

**To add new documents and re-index:**
```bash
cp new_doc.pdf data/uploads/
docker compose restart rag-api
```

---

## 6. Running Evaluation (RAGAS)

After you've run ingestion at least once (Step 5 above):
```bash
python eval/evaluate.py
```
This runs every question in `eval/golden_dataset.json` through the live
retrieve → rerank → generate pipeline and scores it on four axes
(`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`).
Results print to the console and are saved to `eval/last_run_results.json`.
See the comments at the top of `eval/evaluate.py` for what each metric
diagnoses and how to read them. To evaluate your own documents, edit
`eval/golden_dataset.json` with questions/answers relevant to what you
ingested.

---

## 7. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `503 Index not built yet` on `/query` | Run `python scripts/ingest.py` before starting the server |
| `OSError: couldn't connect to huggingface.co` | No internet on first run, or a corporate firewall blocking it — the embedding/reranker models need to download once |
| Empty/irrelevant answers | Check `data/uploads/` actually has files, and that `scripts/ingest.py` reported a non-zero chunk count |
| `OPENAI_API_KEY` errors | Make sure `.env` is populated and you ran `cp .env.example .env` (not just edited the example) |
| Docker build is slow | The Dockerfile pre-downloads embedding/reranker models at build time — this is a one-time cost baked into the image layer |

- **Grounding guardrail**: the system prompt explicitly instructs the model to say it doesn't know if the context doesn't contain the answer — and the golden eval set has a question specifically designed to test this.
- **Evaluation is split by stage**: RAGAS metrics separate retrieval quality (`context_precision`/`recall`) from generation quality (`faithfulness`/`answer_relevancy`) so regressions can be localized.
- **Swap points for scale**: FAISS → Pinecone/Qdrant/Milvus for sharding + metadata filtering; single-process `ingest.py` → Airflow DAG or Kafka consumer for millions of docs; local sentence-transformers → hosted embedding API for throughput.
