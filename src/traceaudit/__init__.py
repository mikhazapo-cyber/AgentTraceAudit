"""Detect and explain errors and inefficiencies in AI agent traces."""

from .pipeline import analyze_trace
from .schemas import CanonicalTrace, Finding, TraceResult

__version__ = "0.1.0"
__all__ = ["CanonicalTrace", "Finding", "TraceResult", "analyze_trace", "__version__"]
