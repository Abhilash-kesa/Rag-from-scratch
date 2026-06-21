"""
generator.py — calls the LLM and resolves citations back to real chunks.

Citation verification step: the model returns chunk NUMBERS (1, 2, 3...)
referring to the order chunks were given in the prompt. We map those
back to actual chunk_ids/sources here, and drop any citation number
that's out of range (a basic hallucination guard — the model can't
cite a source that doesn't exist).
"""
import json
import re
from openai import OpenAI
from app.config import settings
from app.generation.prompt import SYSTEM_PROMPT, build_prompt

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return _client


def _parse_llm_json(raw: str) -> dict:
    # Strip accidental markdown fences if the model adds them anyway
    cleaned = re.sub(r"^```json|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "answer": "I don't have enough information in the knowledge base to answer that.",
            "citations": [],
        }

    prompt = build_prompt(question, chunks)
    client = get_client()

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw_text = response.choices[0].message.content

    try:
        parsed = _parse_llm_json(raw_text)
    except (json.JSONDecodeError, IndexError):
        # Fallback: model didn't follow the JSON contract — degrade gracefully
        return {"answer": raw_text, "citations": []}

    # Resolve numeric citation markers back to real source metadata,
    # dropping any citation number the model hallucinated out of range.
    resolved_citations = []
    for n in parsed.get("citations", []):
        if isinstance(n, int) and 1 <= n <= len(chunks):
            chunk = chunks[n - 1]
            resolved_citations.append({
                "marker": n,
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text_snippet": chunk["text"][:200],
            })

    return {
        "answer": parsed.get("answer", ""),
        "citations": resolved_citations,
    }
