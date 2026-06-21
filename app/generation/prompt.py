"""
prompt.py — builds the final prompt sent to the LLM.

Key design choices:
1. Each chunk gets a numeric marker [1], [2]... so the model can
   refer to specific sources instead of vague paraphrasing.
2. The model is asked to return STRUCTURED JSON (answer + citations),
   not free-text citations — this makes citations machine-verifiable
   downstream instead of trusting the model's inline formatting.
3. Explicit grounding instruction: if the context doesn't contain the
   answer, the model must say so rather than guessing.
"""

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context chunks.

Rules:
- Base your answer strictly on the numbered context chunks below. Do not use outside knowledge.
- If the context does not contain enough information to answer, say so explicitly — do not guess.
- Every factual claim in your answer must be traceable to at least one chunk.
- Respond ONLY with valid JSON in this exact shape, no markdown fences, no preamble:
{"answer": "<your answer text, with inline [n] markers next to claims>", "citations": [<chunk numbers used, as integers>]}
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(f"[{i}] (source: {chunk['source']})\n{chunk['text']}")
    context = "\n\n".join(context_blocks)

    return f"""Context:
{context}

Question: {question}

Answer the question using only the context above, following the JSON response rules."""
