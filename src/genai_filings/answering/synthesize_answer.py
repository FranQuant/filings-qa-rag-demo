import os
from typing import Dict, List

from openai import OpenAI

from ..retrieval import retrieve

SYSTEM_PROMPT = (
    "You are a financial research assistant.\n"
    "Answer ONLY using the provided source excerpts.\n"
    "If the sources are insufficient, say so explicitly.\n"
    "Do NOT hallucinate facts.\n"
    "Use professional investment-research tone.\n"
    "Cite sources by number."
)

MAX_CONTEXT_TOKENS = 12000


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _format_source(index: int, item: Dict[str, object]) -> str:
    return (
        f"[SOURCE {index}]\n"
        f"Issuer: {item['issuer']}\n"
        f"Period: {item['period']}\n"
        f"Document: {item['doc_type']}\n"
        f"Source file: {item['source_file']}\n"
        f"Chunk ID: {item['chunk_id']}\n"
        "Text:\n"
        f"{item['text']}"
    )


def _build_context(chunks: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted(chunks, key=lambda item: item["rank"])
    total_tokens = sum(_estimate_tokens(item["text"]) for item in ordered)
    if total_tokens <= MAX_CONTEXT_TOKENS:
        return ordered
    trimmed = list(ordered)
    while trimmed and total_tokens > MAX_CONTEXT_TOKENS:
        removed = trimmed.pop()
        total_tokens -= _estimate_tokens(removed["text"])
    return trimmed


def synthesize_answer(
    query: str,
    issuer: str,
    period: str,
    k: int,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for answer synthesis.")

    retrieved = retrieve(query=query, issuer=issuer, period=period, k=k)
    context_chunks = _build_context(retrieved)

    context_block = "\n\n".join(
        _format_source(index + 1, item) for index, item in enumerate(context_chunks)
    )
    user_prompt = (
        f"{context_block}\n\n"
        "User Question:\n"
        f"{query}\n\n"
        "Return Markdown in this structure:\n"
        "### Answer\n"
        "<concise synthesized answer>\n\n"
        "### Key Points\n"
        "- Bullet\n"
        "- Bullet\n"
        "- Bullet\n\n"
        "### Sources\n"
        "- [1] <doc_type> | file=<source_file> | chunk=<chunk_id>\n"
        "- [2] ...\n"
    )

    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    answer_text = response.output_text if hasattr(response, "output_text") else ""

    return {
        "query": query,
        "issuer": issuer,
        "period": period,
        "model": model,
        "used_chunks": len(context_chunks),
        "answer_markdown": answer_text,
    }
