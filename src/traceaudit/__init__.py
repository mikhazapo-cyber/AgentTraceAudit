"""Audit AI agent traces for errors and inefficiencies."""

from .pipeline import analyze_trace
from .schemas import CanonicalTrace, Finding, TraceResult

__version__ = "0.2.0"
__all__ = ["CanonicalTrace", "Finding", "TraceResult", "analyze_trace", "__version__"]
