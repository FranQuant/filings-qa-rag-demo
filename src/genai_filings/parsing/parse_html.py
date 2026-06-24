"""Parse EDGAR HTML filings (inline XBRL) into clean section text.

``parse_pdfs.py`` is PDF-only (one row per page). EDGAR primary documents are
HTML — typically inline XBRL with thousands of ``<ix:...>`` tags and hundreds
of financial tables. This module strips that markup to clean, ordered text and
writes ``sections.parquet`` in the exact schema the chunker expects:
``issuer, period, doc_type, source_file, page, text``.

There are no real "pages" in HTML, so ``page`` is a synthetic, monotonically
increasing ordering key. The downstream chunker concatenates all rows for a
document in ``page`` order and re-chunks into fixed token windows, so the only
thing that matters is that text is clean and in document order. We pack the
document-order text into ~``PAGE_TARGET_CHARS`` blocks purely so the rows stay
inspectable.

Table handling: financial tables are rendered row by row with ``" | "`` between
cells. Whitespace is collapsed downstream, so this delimiter is what keeps a
number from smushing into the adjacent label (e.g. ``... | $ | 25,479 | 2.50``).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

from bs4 import BeautifulSoup

MANIFEST_NAME = "manifest.json"
SECTIONS_NAME = "sections.parquet"

# Target size of a synthetic "page" (ordering block) in characters.
PAGE_TARGET_CHARS = 3000

# HTML parser backend. bs4's built-in "html.parser" needs no extra deps; it
# handles inline-XBRL documents fine for text extraction.
_PARSER = "html.parser"

# Inline-XBRL wrappers whose contents must NOT be rendered as visible text.
# ``ix:hidden`` holds non-displayed facts; ``ix:header`` holds context/unit
# metadata. Both would otherwise inject noise into the text stream.
_DROP_TAG_RE = re.compile(r"ix:(hidden|header)", re.I)


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _render_table(table) -> str:
    """Render an HTML table row by row with cell delimiters.

    Empty cells (common in financial tables used for spacing/indentation) are
    dropped so the delimiters mark real value boundaries.
    """
    lines: List[str] = []
    for row in table.find_all("tr"):
        cells = [
            _clean_whitespace(cell.get_text(" ", strip=True))
            for cell in row.find_all(["td", "th"])
        ]
        cells = [cell for cell in cells if cell]
        if cells:
            lines.append(" | ".join(cells))
    return " \n".join(lines)


def html_to_pages(html: str, target_chars: int = PAGE_TARGET_CHARS) -> List[str]:
    """Convert filing HTML into ordered, clean text blocks ("pages")."""
    soup = BeautifulSoup(html, _PARSER)
    for tag in soup(["script", "style"]):
        tag.decompose()
    for tag in soup.find_all(_DROP_TAG_RE):
        tag.decompose()

    # Replace each table with a rendered-text node so it keeps its document
    # position when we flatten to text.
    for table in soup.find_all("table"):
        table.replace_with(soup.new_string(" " + _render_table(table) + " "))

    full_text = _clean_whitespace(soup.get_text(" ", strip=True))
    if not full_text:
        return []

    words = full_text.split(" ")
    pages: List[str] = []
    buffer: List[str] = []
    size = 0
    for word in words:
        buffer.append(word)
        size += len(word) + 1
        if size >= target_chars:
            pages.append(" ".join(buffer))
            buffer = []
            size = 0
    if buffer:
        pages.append(" ".join(buffer))
    return pages


def _load_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_parquet(rows: List[Dict[str, object]], output_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def run_parse_html(issuer: str, period: str, input_root: Path, output_root: Path) -> Path:
    """Parse the HTML filing(s) listed in the raw manifest into sections.parquet."""
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
        html_path = input_dir / source_file
        html = html_path.read_text(encoding="utf-8", errors="replace")
        for page_index, page_text in enumerate(html_to_pages(html), start=1):
            rows.append(
                {
                    "issuer": issuer.upper(),
                    "period": period.upper(),
                    "doc_type": doc_type,
                    "source_file": source_file,
                    "page": page_index,
                    "text": page_text,
                }
            )

    output_dir = output_root / issuer.upper() / period.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / SECTIONS_NAME
    _write_parquet(rows, output_path)
    return output_path


def build_parse_html_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., JPM).")
    parser.add_argument("--period", required=True, help="Period label (e.g., 2026Q1).")
