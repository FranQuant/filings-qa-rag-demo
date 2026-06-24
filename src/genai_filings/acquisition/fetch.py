import argparse
import hashlib
import json
import mimetypes
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .sources import SOURCES

RESULTS_CENTER_URL = "https://www.investidores.nu/en/financials/results-center/"

DOC_KEYWORDS = {
    "earnings_release": ("earnings release", "results release", "earnings"),
    "financial_statements": ("financial statements", "financial statement", "financials"),
    "conference_call_transcript": ("conference call transcript", "transcript", "conference call"),
}


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _period_tokens(period: str) -> Tuple[str, ...]:
    period_norm = period.lower().strip()
    tokens = {period_norm}
    if period_norm.startswith("q") and period_norm[1:].isdigit():
        tokens.add(f"{period_norm[1:]}q")
    if period_norm.startswith("q2"):
        tokens.update({"second quarter", "2nd quarter"})
    return tuple(tokens)


def _fetch_results_center(session: requests.Session) -> BeautifulSoup:
    response = session.get(RESULTS_CENTER_URL, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _candidate_links(soup: BeautifulSoup) -> Iterable[Tuple[str, str]]:
    for anchor in soup.find_all("a", href=True):
        text = _normalize_text(anchor.get_text(" ", strip=True))
        href = anchor["href"].strip()
        if not href:
            continue
        yield text, href


def _matches_period(text: str, href: str, period_tokens: Tuple[str, ...]) -> bool:
    haystack = f"{text} {href}".lower()
    return any(token in haystack for token in period_tokens)


def _select_documents(
    links: Iterable[Tuple[str, str]], period: str
) -> Dict[str, str]:
    period_tokens = _period_tokens(period)
    selections: Dict[str, str] = {}
    for text, href in links:
        if not _matches_period(text, href, period_tokens):
            continue
        for doc_type, keywords in DOC_KEYWORDS.items():
            if doc_type in selections:
                continue
            if any(keyword in text for keyword in keywords) or any(
                keyword in href.lower() for keyword in keywords
            ):
                selections[doc_type] = urljoin(RESULTS_CENTER_URL, href)
    return selections


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(
    session: requests.Session, url: str, dest: Path
) -> Tuple[str, int, Optional[str]]:
    response = session.get(url, stream=True, timeout=30)
    response.raise_for_status()

    hasher = hashlib.sha256()
    size = 0
    temp_path = dest.with_suffix(dest.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            hasher.update(chunk)
            size += len(chunk)
    temp_path.replace(dest)
    content_type = response.headers.get("Content-Type")
    return hasher.hexdigest(), size, content_type


def _load_manifest(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_manifest(path: Path, manifest: Dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def _resolve_filename(url: str, issuer: str, period: str, doc_type: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        name = f"{issuer.lower()}_{period.lower()}_{doc_type}.bin"
    return name


def _guess_content_type(path: Path) -> Optional[str]:
    content_type, _ = mimetypes.guess_type(path.name)
    return content_type


def run_fetch(issuer: str, period: str, output_root: Path) -> Dict[str, Dict]:
    output_dir = output_root / issuer.upper() / period.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    existing_entries = {entry["url"]: entry for entry in manifest.get("files", [])}
    existing_hashes = {entry["sha256"] for entry in manifest.get("files", [])}

    issuer_key = issuer.upper()
    period_key = period.upper()
    documents = (
        SOURCES.get(issuer_key, {}).get(period_key, {}).copy()
        if SOURCES.get(issuer_key, {}).get(period_key)
        else {}
    )
    session = requests.Session()
    if not documents:
        soup = _fetch_results_center(session)
        links = list(_candidate_links(soup))
        documents = _select_documents(links, period)

    missing = [doc for doc in DOC_KEYWORDS if doc not in documents]
    if missing:
        warnings.warn(
            f"Missing documents for {period}: {', '.join(missing)}", RuntimeWarning
        )

    files: List[Dict[str, str]] = []
    retrieved_at = datetime.now(timezone.utc).isoformat()

    for doc_type, url in documents.items():
        filename = _resolve_filename(url, issuer, period, doc_type)
        dest_path = output_dir / filename
        if dest_path.exists():
            existing_hash = _hash_file(dest_path)
            entry = existing_entries.get(url)
            if (entry and entry.get("sha256") == existing_hash) or (
                existing_hash in existing_hashes
            ):
                size = dest_path.stat().st_size
                content_type = entry.get("content_type") if entry else None
                if not content_type:
                    content_type = _guess_content_type(dest_path)
                files.append(
                    {
                        "doc_type": doc_type,
                        "filename": filename,
                        "url": url,
                        "sha256": existing_hash,
                        "bytes": size,
                        "content_type": content_type,
                        "retrieved_at": retrieved_at,
                    }
                )
                continue

        sha256, size, content_type = _download_file(session, url, dest_path)
        if not content_type:
            content_type = _guess_content_type(dest_path)
        files.append(
            {
                "doc_type": doc_type,
                "filename": filename,
                "url": url,
                "sha256": sha256,
                "bytes": size,
                "content_type": content_type,
                "retrieved_at": retrieved_at,
            }
        )

    manifest = {
        "issuer": issuer.upper(),
        "period": period.upper(),
        "source_page": RESULTS_CENTER_URL,
        "retrieved_at": retrieved_at,
        "missing_doc_types": missing,
        "files": files,
    }
    _write_manifest(manifest_path, manifest)
    return {entry["filename"]: entry for entry in files}


def build_fetch_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., NU).")
    parser.add_argument("--period", required=True, help="Period label (e.g., Q2).")
