import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

MANIFEST_NAME = "manifest.json"


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_pages(pdf_path: Path) -> Iterable[str]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF parsing. Install it with "
            "`pip install pypdf`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    for page in reader.pages:
        yield page.extract_text() or ""


def _write_parquet(rows: List[Dict[str, object]], output_path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to write parquet output. "
            "Install it with `pip install pyarrow`."
        ) from exc

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def run_parse(issuer: str, period: str, input_root: Path, output_root: Path) -> Path:
    input_dir = input_root / issuer.upper() / period.upper()
    manifest_path = input_dir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    files = manifest.get("files", [])
    rows: List[Dict[str, object]] = []

    for entry in files:
        source_file = entry.get("filename")
        doc_type = entry.get("doc_type", "unknown")
        if not source_file:
            continue
        pdf_path = input_dir / source_file
        for page_index, text in enumerate(_extract_pages(pdf_path), start=1):
            rows.append(
                {
                    "issuer": issuer.upper(),
                    "period": period.upper(),
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "page": page_index,
                    "text": _clean_whitespace(text),
                }
            )

    output_dir = output_root / issuer.upper() / period.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sections.parquet"
    _write_parquet(rows, output_path)
    return output_path


def build_parse_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parser.add_argument("--period", required=True, help="Period label (e.g., 2025Q2).")
