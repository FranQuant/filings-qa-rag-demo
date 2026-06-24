"""Shared helpers for filings workflows."""

import hashlib


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_uid(
    issuer: str,
    period: str,
    doc_type: str,
    source_file: str,
    chunk_id: int,
    text_sha256: str,
) -> str:
    payload = f"{issuer}|{period}|{doc_type}|{source_file}|{chunk_id}|{text_sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
