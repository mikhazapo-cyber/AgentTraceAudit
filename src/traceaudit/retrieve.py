from __future__ import annotations

import re

from .evidence import render_step
from .schemas import CanonicalTrace, DerivedCheck
from .structural import pair_calls_results

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_\-]{3,}")
_STOP = {
    "that",
    "this",
    "with",
    "from",
    "have",
    "been",
    "were",
    "when",
    "what",
    "which",
    "into",
    "your",
    "their",
    "them",
    "then",
    "than",
    "only",
    "must",
    "should",
    "shall",
    "does",
    "call",
    "called",
    "using",
    "used",
    "tool",
    "tools",
    "step",
    "steps",
    "trace",
    "agent",
    "check",
    "verify",
}


def _keywords(check: DerivedCheck) -> set[str]:
    text = " ".join([check.description, check.condition, check.justified_when, *check.source_refs])
    return {w.lower() for w in _WORD.findall(text) if w.lower() not in _STOP}


def neighborhood_indices(
    trace: CanonicalTrace,
    seeds: list[int],
    radius: int = 1,
) -> list[int]:
    """Cited steps plus neighbours and paired call/result rows."""
    if not trace.steps:
        return []
    pairing = pair_calls_results(trace)
    result_to_call = {ri: ci for ci, ri in pairing.items()}
    n = len(trace.steps)
    out: set[int] = set()
    for raw in seeds:
        try:
            idx = int(raw)
        except (TypeError, ValueError):
            continue
        for j in range(idx - radius, idx + radius + 1):
            if 0 <= j < n:
                out.add(trace.steps[j].index)
        partner = pairing.get(idx, result_to_call.get(idx))
        if partner is not None and 0 <= partner < n:
            out.add(trace.steps[partner].index)
    return sorted(out)


def retrieve_step_indices(trace: CanonicalTrace, checks: list[DerivedCheck], max_steps: int = 36) -> list[int]:
    if not trace.steps:
        return []
    scores = {s.index: 0.0 for s in trace.steps}
    pairing = pair_calls_results(trace)
    kws: set[str] = set()
    for check in checks:
        kws |= _keywords(check)

    for s in trace.steps:
        if s.kind == "tool_result" and s.is_error:
            scores[s.index] += 6
        if s.kind == "message" and s.role == "user" and s.index == 0:
            scores[s.index] += 2
        if s.kind == "message" and s.role == "assistant":
            scores[s.index] += 0.4
        hay = (s.text() or "")[:800].lower()
        name = (s.name or "").lower()
        if name and name in kws:
            scores[s.index] += 5
        hits = sum(1 for k in kws if k in hay)
        if hits:
            scores[s.index] += min(4, hits)
        if s.kind == "tool_call" and s.index in pairing:
            scores[pairing[s.index]] += scores[s.index] * 0.3

    # Always keep a short tail of assistant claims (unsupported_success / contradiction).
    assistants = [s.index for s in trace.steps if s.kind == "message" and s.role == "assistant"]
    for idx in assistants[-2:]:
        scores[idx] += 1.5

    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    picked = [i for i in ranked if scores[i] > 0][:max_steps]
    if not picked:
        picked = [s.index for s in trace.steps[:8]]
    return neighborhood_indices(trace, picked, radius=1)


def render_retrieved(
    trace: CanonicalTrace,
    indices: list[int],
    char_budget: int = 16000,
    arg_limit: int = 2400,
    content_limit: int = 3600,
) -> str:
    parts = []
    used = 0
    for idx in indices:
        if not (0 <= idx < len(trace.steps)):
            continue
        block = render_step(trace.steps[idx], arg_limit=arg_limit, content_limit=content_limit)
        if used + len(block) + 1 > char_budget:
            break
        parts.append(block)
        used += len(block) + 1
    return "\n".join(parts) if parts else "(no retrieved excerpts)"
