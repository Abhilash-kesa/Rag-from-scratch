"""
eval/evaluate.py — RAG evaluation using RAGAS.

What this measures and why each metric matters:

  faithfulness        Does the generated answer only contain claims that are
                       actually supported by the retrieved context? This is
                       THE hallucination metric — low faithfulness means the
                       LLM is making things up beyond what it was given.

  answer_relevancy     Does the answer actually address the question asked
                       (as opposed to being faithful to context but off-topic
                       or evasive)? Faithfulness and relevancy are deliberately
                       separate axes — an answer can be perfectly grounded in
                       context and still fail to answer the question.

  context_precision    Of the chunks that were retrieved, how many were
                       actually relevant/useful? Low precision = retriever is
                       pulling in noise that dilutes the LLM's context window.

  context_recall       Of the information needed to answer correctly (per the
                       ground_truth), how much did retrieval actually surface?
                       Low recall = the right chunk never made it into context,
                       no matter how good the LLM or reranker are downstream.

Together: faithfulness/answer_relevancy score the GENERATION step,
context_precision/context_recall score the RETRIEVAL step. Splitting eval
this way is what lets you debug "is my retriever bad or is my LLM bad?"
instead of staring at one end-to-end accuracy number.

Usage:
    python eval/evaluate.py
Requires: an index already built via `python scripts/ingest.py`.
"""
import json
import sys
import time
from pathlib import Path

from tenacity import retry, wait_exponential, stop_after_attempt

sys.path.append(str(Path(__file__).resolve().parent.parent))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import retrieve
from app.retrieval.reranker import rerank
from app.generation.generator import generate_answer
from app.ingestion.embed import get_embedding_model


@retry(wait=wait_exponential(multiplier=2, min=4, max=60), stop=stop_after_attempt(5))
def safe_generate_answer(q, chunks):
    return generate_answer(q, chunks)


def run_pipeline_on_dataset(golden: list[dict]) -> dict:
    """Runs each golden question through the real retrieve->rerank->generate
    pipeline and collects what RAGAS needs: question, answer, retrieved
    contexts, and ground_truth."""
    dim = get_embedding_model().get_sentence_embedding_dimension()
    store = VectorStore.load(settings.INDEX_DIR, dim)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden:
        q = item["question"]
        candidates = retrieve(q, store)
        top_chunks = rerank(q, candidates)
        result = safe_generate_answer(q, top_chunks)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([c["text"] for c in top_chunks])
        ground_truths.append(item["ground_truth"])

        time.sleep(5)

    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def main():
    golden_path = Path(__file__).parent / "golden_dataset.json"
    golden = json.loads(golden_path.read_text())

    print(f"Running pipeline on {len(golden)} golden questions...")
    results = run_pipeline_on_dataset(golden)
    dataset = Dataset.from_dict(results)

    # RAGAS needs an LLM (as judge) and embeddings — wired to the same
    # providers the rest of this project already uses, so no extra API keys.
    judge_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-4o",
        api_key=settings.OPENAI_API_KEY,
    ))
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)
    )

    print("Scoring with RAGAS (this calls the judge LLM several times per row)...")
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    df = scores.to_pandas()
    question_col = "user_input" if "user_input" in df.columns else "question"
    print("\n=== Per-question scores ===")
    print(df[[question_col, "faithfulness", "answer_relevancy", "context_precision", "context_recall"]]
          .to_string(index=False))

    print("\n=== Aggregate scores ===")
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric:20s}: {df[metric].mean():.3f}")

    out_path = Path(__file__).parent / "last_run_results.json"
    df.to_json(out_path, orient="records", indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
