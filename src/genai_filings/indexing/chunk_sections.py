import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
SECTIONS_NAME = "sections.parquet"
CHUNKS_NAME = "chunks.parquet"


def _tokenize(text: str) -> List[str]:
    return text.split()


def _detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def _chunk_tokens(tokens: List[str]) -> Iterable[List[str]]:
    if not tokens:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
    chunks: List[List[str]] = []
    for start in range(0, len(tokens), step):
        chunk = tokens[start : start + CHUNK_SIZE]
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(tokens):
            break
    return chunks


def _load_sections(path: Path) -> List[Dict[str, object]]:
    table = pq.read_table(path)
    return table.to_pylist()


def _group_rows(rows: List[Dict[str, object]]) -> Dict[Tuple[str, str, str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row.get("issuer", "")),
            str(row.get("period", "")),
            str(row.get("doc_type", "")),
            str(row.get("source_file", "")),
        )
        grouped.setdefault(key, []).append(row)
    for key in grouped:
        grouped[key].sort(key=lambda r: int(r.get("page", 0)))
    return grouped


def run_index(issuer: str, period: str, input_root: Path, output_root: Path) -> Path:
    input_dir = input_root / issuer.upper() / period.upper()
    sections_path = input_dir / SECTIONS_NAME
    if not sections_path.exists():
        raise FileNotFoundError(f"Sections file not found: {sections_path}")

    rows = _load_sections(sections_path)
    grouped = _group_rows(rows)
    chunks: List[Dict[str, object]] = []
    chunk_counters: Dict[str, int] = {}

    for (row_issuer, row_period, doc_type, source_file), entries in grouped.items():
        combined_text = " ".join(entry.get("text", "") for entry in entries).strip()
        tokens = _tokenize(combined_text)
        for token_chunk in _chunk_tokens(tokens):
            chunk_text = _detokenize(token_chunk)
            chunk_counters[source_file] = chunk_counters.get(source_file, 0) + 1
            chunk_id = chunk_counters[source_file]
            chunks.append(
                {
                    "issuer": row_issuer,
                    "period": row_period,
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "chunk_id": chunk_id,
                    "char_len": len(chunk_text),
                    "text": chunk_text,
                }
            )

    output_dir = output_root / issuer.upper() / period.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / CHUNKS_NAME
    table = pa.Table.from_pylist(chunks)
    pq.write_table(table, output_path)
    return output_path


def build_index_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parser.add_argument("--period", required=True, help="Period label (e.g., Q2).")
