import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

CHUNKS_NAME = "chunks.parquet"
EMBEDDINGS_NAME = "embeddings.parquet"
MANIFEST_NAME = "embeddings_manifest.json"

from ..common import _chunk_uid, _normalize_text, _sha256_text


def _load_chunks(path: Path) -> List[Dict[str, object]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _load_embeddings(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    table = pq.read_table(path)
    return table.to_pylist()


def _write_parquet(rows: List[Dict[str, object]], output_path: Path) -> None:
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def _write_manifest(path: Path, manifest: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _relativize_path(path: Path) -> str:
    """Render a path relative to the current working directory when possible.

    Manifests are committed artifacts, so they must not leak absolute,
    machine-specific paths (e.g. a user's home directory). When the path lives
    under the repo root (the cwd for the pipeline), store it relative; otherwise
    fall back to the path as given.
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _get_openai_client() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "openai is required for embeddings. Install it with `pip install openai`."
        ) from exc
    return OpenAI()


def _embed_batch(
    client: object, model: str, texts: List[str], retries: int = 3, pause: float = 1.0
) -> List[List[float]]:
    attempt = 0
    while True:
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt > retries:
                raise exc
            time.sleep(pause * (2**(attempt - 1)))


def run_embed(
    issuer: str,
    period: str,
    input_root: Path,
    output_root: Path,
    model: str,
    batch_size: int,
    force: bool,
) -> Tuple[Path, Path]:
    started_at = time.time()
    load_dotenv(".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is required for embeddings.")

    input_dir = input_root / issuer.upper() / period.upper()
    chunks_path = input_dir / CHUNKS_NAME
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    output_dir = output_root / issuer.upper() / period.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / EMBEDDINGS_NAME
    manifest_path = output_dir / MANIFEST_NAME

    chunks = _load_chunks(chunks_path)
    existing_embeddings = _load_embeddings(embeddings_path)
    existing_by_model = [
        row for row in existing_embeddings if row.get("embedding_model") == model
    ]
    existing_map = {
        row["chunk_uid"]: row for row in existing_by_model if row.get("chunk_uid")
    }

    prepared: List[Dict[str, object]] = []
    missing_text_rows = 0
    for row in chunks:
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            missing_text_rows += 1
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
        prepared.append(
            {
                "issuer": str(row.get("issuer", "")).upper(),
                "period": str(row.get("period", "")).upper(),
                "doc_type": str(row.get("doc_type", "")),
                "source_file": str(row.get("source_file", "")),
                "chunk_id": int(row.get("chunk_id", 0)),
                "chunk_uid": chunk_uid,
                "text_sha256": text_sha256,
                "text": normalized,
                "text_preview": normalized[:200],
                "tokens_est": len(normalized.split()),
            }
        )

    if force:
        to_embed = prepared
    else:
        to_embed = [row for row in prepared if row["chunk_uid"] not in existing_map]

    client = _get_openai_client()
    embedded_at = datetime.now(timezone.utc).isoformat()
    new_rows: List[Dict[str, object]] = []
    errors: List[str] = []

    for start in range(0, len(to_embed), batch_size):
        batch = to_embed[start : start + batch_size]
        texts = [row["text"] for row in batch]
        try:
            embeddings = _embed_batch(client, model, texts)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue

        for row, vector in zip(batch, embeddings):
            new_rows.append(
                {
                    "issuer": row["issuer"],
                    "period": row["period"],
                    "doc_type": row["doc_type"],
                    "source_file": row["source_file"],
                    "chunk_id": row["chunk_id"],
                    "chunk_uid": row["chunk_uid"],
                    "text_sha256": row["text_sha256"],
                    "embedding_model": model,
                    "embedding_dim": len(vector),
                    "embedded_at": embedded_at,
                    "embedding": vector,
                    "text_preview": row.get("text_preview"),
                    "tokens_est": row.get("tokens_est"),
                }
            )

    combined = existing_embeddings + new_rows
    deduped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in sorted(
        combined,
        key=lambda r: (
            str(r.get("chunk_uid", "")),
            str(r.get("embedding_model", "")),
            str(r.get("embedded_at", "")),
        ),
    ):
        key = (str(row.get("chunk_uid", "")), str(row.get("embedding_model", "")))
        deduped[key] = row

    final_rows = sorted(
        deduped.values(),
        key=lambda r: (str(r.get("chunk_uid", "")), str(r.get("embedding_model", ""))),
    )
    _write_parquet(final_rows, embeddings_path)

    embedding_dim = None
    if new_rows:
        embedding_dim = new_rows[0].get("embedding_dim")
    elif existing_by_model:
        embedding_dim = existing_by_model[0].get("embedding_dim")

    manifest = {
        "issuer": issuer.upper(),
        "period": period.upper(),
        "source_input": _relativize_path(chunks_path),
        "output_file": _relativize_path(embeddings_path),
        "embedding_model": model,
        "embedding_dim": embedding_dim,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "chunks_total": len(prepared),
            "embedded_existing": len(existing_by_model),
            "embedded_new": len(new_rows),
            "embedded_total": len([row for row in final_rows if row.get("embedding_model") == model]),
        },
        "missing_text_rows": missing_text_rows,
        "errors": errors,
        "runtime": {"seconds": round(time.time() - started_at, 3)},
    }
    _write_manifest(manifest_path, manifest)
    return embeddings_path, manifest_path


def build_embed_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
    parser.add_argument("--model", required=True, help="Embedding model name.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--force", action="store_true", help="Re-embed all chunks.")
