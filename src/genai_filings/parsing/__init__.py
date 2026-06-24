"""Parsing utilities for filings."""

from .parse_pdfs import run_parse
from .parse_html import run_parse_html

__all__ = ["run_parse", "run_parse_html"]
