import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import pyarrow.parquet as pq

from ..common import _chunk_uid, _normalize_text, _sha256_text

EMBEDDINGS_NAME = "embeddings.parquet"
MANIFEST_NAME = "embeddings_manifest.json"
CHUNKS_NAME = "chunks.parquet"


def _load_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_parquet_rows(path: Path) -> List[Dict[str, object]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _get_openai_client() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for retrieval. Install it with `pip install openai`."
        ) from exc
    return OpenAI()


def _embed_query(client: object, model: str, query: str) -> List[float]:
    response = client.embeddings.create(model=model, input=[query])
    return response.data[0].embedding


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Embedding dimension mismatch.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _build_text_map(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        normalized = _normalize_text(text)
        text_sha256 = _sha256_text(normalized)
        chunk_uid = _chunk_uid(
            str(row.get("issuer", "")).upper(),
            str(row.get("period", "")).upper(),
            str(row.get("doc_type", "")),
            str(row.get("source_file", "")),
            int(row.get("chunk_id", 0)),
            text_sha256,
        )
        mapping[chunk_uid] = {
            "text": normalized,
            "text_preview": normalized[:200],
        }
    return mapping


def retrieve(
    query: str,
    issuer: str,
    period: str,
    k: int = 5,
    doc_type: Optional[str] = None,
) -> List[Dict[str, object]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for retrieval.")

    issuer_key = issuer.upper()
    period_key = period.upper()

    base_dir = Path("data/processed/filings") / issuer_key / period_key
    embeddings_path = base_dir / EMBEDDINGS_NAME
    manifest_path = base_dir / MANIFEST_NAME
    chunks_path = base_dir / CHUNKS_NAME

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Embeddings manifest not found: {manifest_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    manifest = _load_manifest(manifest_path)
    embedding_model = manifest.get("embedding_model")
    if not embedding_model:
        raise ValueError("Embedding model missing from manifest.")

    embeddings_rows = _load_parquet_rows(embeddings_path)
    filtered_rows = [
        row
        for row in embeddings_rows
        if str(row.get("issuer", "")).upper() == issuer_key
        and str(row.get("period", "")).upper() == period_key
        and str(row.get("embedding_model", "")) == embedding_model
    ]
    if doc_type:
        filtered_rows = [
            row
            for row in filtered_rows
            if str(row.get("doc_type", "")) == doc_type
        ]

    mismatched = [
        row
        for row in embeddings_rows
        if str(row.get("issuer", "")).upper() == issuer_key
        and str(row.get("period", "")).upper() == period_key
        and str(row.get("embedding_model", "")) != embedding_model
    ]
    if mismatched:
        raise ValueError("Embedding model mismatch detected in embeddings.parquet.")

    chunks_rows = _load_parquet_rows(chunks_path)
    text_map = _build_text_map(chunks_rows)

    client = _get_openai_client()
    query_embedding = _embed_query(client, embedding_model, query)

    scored: List[Dict[str, object]] = []
    for row in filtered_rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = _cosine_similarity(query_embedding, embedding)
        chunk_uid = str(row.get("chunk_uid", ""))
        text_entry = text_map.get(chunk_uid, {})
        scored.append(
            {
                "score": score,
                "chunk_uid": chunk_uid,
                "issuer": str(row.get("issuer", "")),
                "period": str(row.get("period", "")),
                "doc_type": str(row.get("doc_type", "")),
                "source_file": str(row.get("source_file", "")),
                "chunk_id": int(row.get("chunk_id", 0)),
                "text": text_entry.get("text", ""),
                "text_preview": str(row.get("text_preview", "")),
            }
        )

    scored_sorted = sorted(
        scored,
        key=lambda r: (-r["score"], r["chunk_uid"]),
    )
    results = []
    for idx, row in enumerate(scored_sorted[:k], start=1):
        results.append(
            {
                "rank": idx,
                "score": row["score"],
                "chunk_uid": row["chunk_uid"],
                "issuer": row["issuer"],
                "period": row["period"],
                "doc_type": row["doc_type"],
                "source_file": row["source_file"],
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "text_preview": row["text_preview"],
            }
        )
    return results


def build_retrieve_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parser.add_argument("--period", required=True, help="Period label (e.g., Q2).")
    parser.add_argument("--query", required=True, help="Query string.")
    parser.add_argument("--k", type=int, default=5, help="Top-K results.")
    parser.add_argument("--doc-type", dest="doc_type", help="Optional doc type filter.")
