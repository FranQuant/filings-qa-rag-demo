"""Acquire filings live from SEC EDGAR.

This is the live counterpart to ``fetch.py`` (which targets a single
issuer's investor-relations site). EDGAR exposes every US filer's documents
through a small set of JSON endpoints and an Archives host. No API key is
required, but SEC mandates a descriptive ``User-Agent`` that identifies the
caller, and asks clients to stay under 10 requests/second.

Pipeline position: this module writes the same ``manifest.json`` schema that
``parsing`` consumes, so the downstream chunk -> embed steps are unchanged.
The selected primary document is HTML (inline XBRL), not PDF, so it is parsed
by ``parse_html.py`` rather than ``parse_pdfs.py``.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# SEC endpoints.
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
DOCUMENT_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_document}"
)

# SEC asks clients to identify themselves and stay <= 10 requests/second.
# A conservative spacing keeps us comfortably under the ceiling.
MIN_REQUEST_INTERVAL = 0.15  # seconds between requests (~6.7 req/s)

# Quarter label -> the calendar month that a 10-Q reporting period ends in.
# 10-Qs cover Q1-Q3; Q4 is folded into the annual 10-K, so there is no "Q4"
# quarterly filing to select here.
_QUARTER_END_MONTH = {1: 3, 2: 6, 3: 9}


def build_user_agent(contact: str, app: str = "filings-qa-rag-demo") -> str:
    """Build a SEC-compliant User-Agent string identifying the caller.

    SEC requires a descriptive agent that includes a contact (an email or a
    company name). Example: ``filings-qa-rag-demo jane@example.com``.
    """
    contact = contact.strip()
    if not contact:
        raise ValueError(
            "A contact (email or company name) is required for the SEC "
            "User-Agent header. Set it before fetching from EDGAR."
        )
    return f"{app} {contact}"


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return session


class _RateLimitedGetter:
    """Spaces successive GETs to respect SEC's rate guidance."""

    def __init__(self, session: requests.Session, min_interval: float = MIN_REQUEST_INTERVAL):
        self._session = session
        self._min_interval = min_interval
        self._last = 0.0

    def get(self, url: str, **kwargs) -> requests.Response:
        elapsed = time.monotonic() - self._last
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        response = self._session.get(url, timeout=60, **kwargs)
        self._last = time.monotonic()
        response.raise_for_status()
        return response


def ticker_to_cik(ticker: str, getter: _RateLimitedGetter) -> Tuple[int, str]:
    """Resolve a ticker to its integer CIK and zero-padded 10-digit CIK.

    Returns ``(cik_int, cik10)``. ``cik_int`` is used in the Archives URL;
    ``cik10`` is the zero-padded form the submissions endpoint requires.
    """
    ticker_key = ticker.strip().upper()
    data = getter.get(COMPANY_TICKERS_URL).json()
    for entry in data.values():
        if str(entry.get("ticker", "")).upper() == ticker_key:
            cik_int = int(entry["cik_str"])
            return cik_int, str(cik_int).zfill(10)
    raise ValueError(f"Ticker {ticker_key!r} not found in SEC company_tickers.json.")


def get_submissions(cik10: str, getter: _RateLimitedGetter) -> Dict[str, object]:
    return getter.get(SUBMISSIONS_URL.format(cik10=cik10)).json()


def _parse_period(period: str) -> Tuple[int, Optional[int]]:
    """Parse a period label into ``(quarter, year_or_None)``.

    Accepts forms like ``"Q1"``, ``"2026Q1"``, ``"Q1-2026"``, ``"Q1 2026"``.
    A missing year means "the most recent available year for that quarter".
    """
    text = period.strip().upper()
    qmatch = re.search(r"Q([1-4])", text)
    if not qmatch:
        raise ValueError(
            f"Could not find a quarter (Q1-Q4) in period {period!r}. "
            "Use a label like 'Q1' or '2026Q1'."
        )
    quarter = int(qmatch.group(1))
    if quarter == 4:
        raise ValueError(
            "Q4 is reported in the annual 10-K, not a 10-Q. "
            "Choose Q1, Q2, or Q3 for a 10-Q."
        )
    ymatch = re.search(r"(19|20)\d{2}", text)
    year = int(ymatch.group(0)) if ymatch else None
    return quarter, year


def select_filing(
    submissions: Dict[str, object],
    period: str,
    form: str = "10-Q",
) -> Dict[str, str]:
    """Select the target filing for a requested form and period.

    ``filings.recent`` stores fields as parallel arrays (form[i], reportDate[i],
    accessionNumber[i], ...). We zip them into rows, keep the requested form,
    match the period by the report period's end month (and year if given), and
    return the most recent match.
    """
    quarter, year = _parse_period(period)
    target_month = _QUARTER_END_MONTH[quarter]

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    rows = [
        {
            "form": recent["form"][i],
            "filingDate": recent["filingDate"][i],
            "reportDate": recent["reportDate"][i],
            "accessionNumber": recent["accessionNumber"][i],
            "primaryDocument": recent["primaryDocument"][i],
            "primaryDocDescription": recent.get("primaryDocDescription", [None] * len(forms))[i],
        }
        for i in range(len(forms))
        if recent["form"][i] == form
    ]
    if not rows:
        raise ValueError(f"No {form} filings found for this issuer in recent submissions.")

    def _matches(row: Dict[str, str]) -> bool:
        report_date = row.get("reportDate") or ""
        parts = report_date.split("-")
        if len(parts) != 3:
            return False
        r_year, r_month = int(parts[0]), int(parts[1])
        if r_month != target_month:
            return False
        if year is not None and r_year != year:
            return False
        return True

    matches = [row for row in rows if _matches(row)]
    if not matches:
        available = ", ".join(sorted({r["reportDate"] for r in rows})[:8])
        raise ValueError(
            f"No {form} found for period {period!r} "
            f"(report-period end month {target_month:02d}"
            + (f", year {year}" if year else "")
            + f"). Available {form} report dates include: {available}."
        )
    # Most recent report period first.
    matches.sort(key=lambda r: r["reportDate"], reverse=True)
    return matches[0]


def build_document_url(cik_int: int, accession_number: str, primary_document: str) -> str:
    accession_nodash = accession_number.replace("-", "")
    return DOCUMENT_URL.format(
        cik_int=cik_int,
        accession_nodash=accession_nodash,
        primary_document=primary_document,
    )


def canonical_period(report_date: str) -> str:
    """Derive a canonical ``YYYYQN`` period label from a filing's report date.

    The report date is the period-end (e.g. ``2026-03-31`` -> ``2026Q1``). Using
    the actual filing's report date guarantees a year-qualified period even when
    the caller passed a bare quarter like ``"Q1"`` — so all ingestion emits the
    standard ``YYYYQN`` form, never a bare ``QN``.
    """
    parts = report_date.split("-")
    if len(parts) < 2:
        raise ValueError(f"Unexpected report date format: {report_date!r}.")
    year = int(parts[0])
    month = int(parts[1])
    quarter = (month - 1) // 3 + 1
    return f"{year}Q{quarter}"


def run_edgar_fetch(
    issuer: str,
    period: str,
    output_root: Path,
    user_agent: str,
    form: str = "10-Q",
    doc_type: Optional[str] = None,
) -> Dict[str, object]:
    """Fetch the target filing's primary document from EDGAR and write a manifest.

    Writes the HTML document plus a ``manifest.json`` to
    ``output_root/{ISSUER}/{PERIOD}/`` where ``PERIOD`` is the canonical
    ``YYYYQN`` label derived from the selected filing's report date — so the
    output folder is always year-qualified even if the caller passed a bare
    quarter (e.g. ``"Q1"``). ``doc_type`` defaults to the form name
    (e.g. ``"10-Q"``). Returns the manifest dict; its ``period`` field is the
    canonical label callers should use for the downstream stages.
    """
    doc_type = doc_type or form

    getter = _RateLimitedGetter(make_session(user_agent))

    cik_int, cik10 = ticker_to_cik(issuer, getter)
    submissions = get_submissions(cik10, getter)
    filing = select_filing(submissions, period, form=form)

    # Normalize to canonical YYYYQN from the filing's actual report date.
    period_canonical = canonical_period(filing["reportDate"])
    output_dir = output_root / issuer.upper() / period_canonical
    output_dir.mkdir(parents=True, exist_ok=True)

    url = build_document_url(cik_int, filing["accessionNumber"], filing["primaryDocument"])

    response = getter.get(url)
    content = response.content
    sha256 = hashlib.sha256(content).hexdigest()
    filename = filing["primaryDocument"]
    dest_path = output_dir / filename
    dest_path.write_bytes(content)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    file_entry = {
        "doc_type": doc_type,
        "filename": filename,
        "url": url,
        "sha256": sha256,
        "bytes": len(content),
        "content_type": response.headers.get("Content-Type"),
        "retrieved_at": retrieved_at,
        # EDGAR provenance for audit.
        "form": filing["form"],
        "report_date": filing["reportDate"],
        "filing_date": filing["filingDate"],
        "accession_number": filing["accessionNumber"],
        "primary_document": filing["primaryDocument"],
        "cik": cik_int,
    }
    manifest = {
        "issuer": issuer.upper(),
        "period": period_canonical,
        "period_requested": period.upper(),
        "source": "sec_edgar",
        "source_page": SUBMISSIONS_URL.format(cik10=cik10),
        "cik": cik_int,
        "cik10": cik10,
        "user_agent": user_agent,
        "retrieved_at": retrieved_at,
        "missing_doc_types": [],
        "files": [file_entry],
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def build_edgar_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--issuer", required=True, help="Issuer ticker (e.g., JPM).")
    parser.add_argument("--period", required=True, help="Period label (e.g., 2026Q1).")
    parser.add_argument("--user-agent", required=True, help="SEC User-Agent (include a contact).")
    parser.add_argument("--form", default="10-Q", help="Filing form to select (default 10-Q).")
