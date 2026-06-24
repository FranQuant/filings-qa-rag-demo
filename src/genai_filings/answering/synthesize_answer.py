import os
import re
from typing import Dict, List, Optional, Tuple

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

# Supported generation providers. Embeddings always stay on OpenAI
# (the stored vectors were built with an OpenAI model per the manifest),
# so only the *generator* is switchable here.
SUPPORTED_PROVIDERS = ("openai", "anthropic")

# Convenience defaults so callers can pass a provider without a model.
DEFAULT_MODELS = {
    "openai": "gpt-5.5",
    "anthropic": "claude-opus-4-8",
}


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


def _validate_citations(
    answer_text: str, num_sources: int
) -> Tuple[List[int], List[int]]:
    """Extract [N] citations from the answer and split into valid / invalid.

    Valid means 1 <= N <= num_sources, i.e. the citation maps to a source we
    actually provided. Invalid citations indicate the model referenced a
    source number that does not exist -- a grounding failure worth surfacing.
    """
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer_text)})
    valid = [n for n in cited if 1 <= n <= num_sources]
    invalid = [n for n in cited if n < 1 or n > num_sources]
    return valid, invalid


# --- Provider-specific call sites -------------------------------------------


# Anthropic models that DEPRECATED the `temperature` parameter (sending it
# returns a 400). Opus 4.8 and the Fable/Mythos 5 generation use adaptive
# reasoning and reject `temperature`. Older models (Sonnet 4.6, Haiku 4.5,
# and the 4.x Opus line before 4.8) still accept it. Match by prefix so we
# don't have to enumerate every dated snapshot.
_ANTHROPIC_NO_TEMPERATURE_PREFIXES = (
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
    "claude-mythos-preview",
)


def _anthropic_accepts_temperature(model: str) -> bool:
    return not any(
        model.startswith(prefix) for prefix in _ANTHROPIC_NO_TEMPERATURE_PREFIXES
    )


# OpenAI GPT-5 generation are reasoning models that DROPPED the `temperature`
# parameter (sending it returns a 400, like Anthropic's newest models). The
# GPT-4.x line still accepts it. Match by prefix. When temperature is not
# accepted, the model is internally deterministic-ish for our purposes; we
# simply omit the knob rather than failing.
_OPENAI_NO_TEMPERATURE_PREFIXES = (
    "gpt-5",
    "o1",
    "o3",
    "o4",
)


def _openai_accepts_temperature(model: str) -> bool:
    return not any(
        model.startswith(prefix) for prefix in _OPENAI_NO_TEMPERATURE_PREFIXES
    )


def _call_openai(
    model: str, system: str, user: str, max_tokens: int, temperature: float
) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_output_tokens": max_tokens,
    }
    if _openai_accepts_temperature(model):
        kwargs["temperature"] = temperature
    response = client.responses.create(**kwargs)
    return response.output_text if hasattr(response, "output_text") else ""


def _call_anthropic(
    model: str, system: str, user: str, max_tokens: int, temperature: float
) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    # Anthropic differences vs OpenAI Responses API:
    #   - `system` is a top-level kwarg, not a message role
    #   - token cap is `max_tokens` (not `max_output_tokens`)
    #   - text lives in message.content[] blocks, not `output_text`
    #   - newer models (e.g. Opus 4.8) DEPRECATED `temperature`; sending it
    #     returns a 400. We omit it for those and only pass it to models that
    #     still accept it. Determinism on the OpenAI side still uses temp=0.0.
    kwargs = {
        "model": model,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
    }
    if _anthropic_accepts_temperature(model):
        kwargs["temperature"] = temperature
    message = client.messages.create(**kwargs)
    parts = [
        block.text
        for block in message.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts)


def _call_llm(
    provider: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if provider == "openai":
        return _call_openai(model, system, user, max_tokens, temperature)
    if provider == "anthropic":
        return _call_anthropic(model, system, user, max_tokens, temperature)
    raise ValueError(
        f"Unsupported provider '{provider}'. "
        f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
    )


def _require_api_key(provider: str) -> None:
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for answer synthesis.")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is required for Anthropic answer synthesis."
        )


# --- Public entrypoint -------------------------------------------------------


def synthesize_answer(
    query: str,
    issuer: str,
    period: str,
    k: int,
    max_tokens: int,
    temperature: float,
    model: Optional[str] = None,
    provider: str = "openai",
) -> dict:
    """Synthesize a cited answer from retrieved filings chunks.

    Args:
        provider: "openai" (default) or "anthropic". Selects the *generator*
            only; query embeddings always use the OpenAI model recorded in the
            embeddings manifest, so retrieval is unaffected by this choice.
        model: Explicit model id. If omitted, a sensible per-provider default
            is used (see DEFAULT_MODELS).
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}."
        )

    # retrieve() needs OPENAI_API_KEY regardless of generator (embeddings).
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for retrieval.")
    _require_api_key(provider)

    resolved_model = model or DEFAULT_MODELS[provider]

    retrieved = retrieve(query=query, issuer=issuer, period=period, k=k)
    context_chunks = _build_context(retrieved)

    # Guard: if trimming or empty retrieval left us with no context, do not
    # invite the model to cite sources that do not exist. Fail closed.
    if not context_chunks:
        return {
            "query": query,
            "issuer": issuer,
            "period": period,
            "provider": provider,
            "model": resolved_model,
            "used_chunks": 0,
            "answer_markdown": (
                "### Answer\n"
                "No source excerpts were available for this query, so no "
                "grounded answer can be produced.\n"
            ),
            "citations_valid": [],
            "citations_invalid": [],
            "error": None,
        }

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

    error: Optional[str] = None
    answer_text = ""
    try:
        answer_text = _call_llm(
            provider=provider,
            model=resolved_model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:  # surface, don't crash the pipeline
        error = f"{type(exc).__name__}: {exc}"

    valid, invalid = _validate_citations(answer_text, len(context_chunks))

    return {
        "query": query,
        "issuer": issuer,
        "period": period,
        "provider": provider,
        "model": resolved_model,
        "used_chunks": len(context_chunks),
        "answer_markdown": answer_text,
        "citations_valid": valid,
        "citations_invalid": invalid,
        "error": error,
    }
