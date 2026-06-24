"""Data acquisition for filings."""

from .fetch import run_fetch
from .edgar import run_edgar_fetch

__all__ = ["run_fetch", "run_edgar_fetch"]
