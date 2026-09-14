from __future__ import annotations

import re

from .schemas import CandidateFinding, CanonicalTrace, Evidence


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def quote_supported(trace: CanonicalTrace, ev: Evidence, min_len: int = 12, overlap: float = 0.9) -> bool:
    if ev.step < 0 or ev.step >= len(trace.steps):
        return False
    quote = _norm(ev.quote)
    if len(quote) < 8:
        return False
    hay = _norm(trace.steps[ev.step].text())
    if quote in hay:
        return True
    if len(quote) < min_len:
        return False
    words = [w for w in re.findall(r"[a-z0-9_]+", quote) if len(w) > 2]
    if len(words) < 4:
        return quote[:32] in hay
    hits = sum(1 for w in words if w in hay)
    return hits / len(words) >= overlap


def verify_candidate(trace: CanonicalTrace, cand: CandidateFinding) -> CandidateFinding:
    if not cand.evidence:
        cand.meta["quote_verified"] = 0
        cand.meta["quote_total"] = 0
        return cand
    ok = sum(1 for e in cand.evidence if quote_supported(trace, e))
    cand.meta["quote_verified"] = ok
    cand.meta["quote_total"] = len(cand.evidence)
    return cand


def locatable(trace: CanonicalTrace, cand: CandidateFinding) -> bool:
    if not cand.steps:
        return False
    return all(0 <= i < len(trace.steps) for i in cand.steps)


def excerpt_in_source(excerpt: str, *blobs: str) -> bool:
    needle = _norm(excerpt)
    if len(needle) < 12:
        return False
    hay = _norm("\n".join(b or "" for b in blobs))
    if needle in hay:
        return True
    words = [w for w in re.findall(r"[a-z0-9_]+", needle) if len(w) > 2]
    if len(words) < 5:
        return needle[:40] in hay
    return sum(1 for w in words if w in hay) / len(words) >= 0.85


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def supporting_instruction(trace: CanonicalTrace, tool_name: str) -> str:
    """First verbatim sentence in the task or instructions that names this tool."""
    name = (tool_name or "").strip()
    if len(name) < 2:
        return ""
    blob = f"{trace.task or ''}\n{trace.instructions or ''}"
    # Word characters only: a trailing period is sentence punctuation, not
    # part of a qualified tool name (those are matched as the full name).
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])"
    for raw in _SENTENCE_SPLIT.split(blob):
        sentence = raw.strip().replace("\n", " ")
        if len(sentence) < 16 or len(sentence) > 420:
            continue
        if re.search(pattern, sentence, re.IGNORECASE):
            return sentence[:400]
    return ""


def classify_evidence_source(trace: CanonicalTrace, ev: Evidence) -> str:
    if ev.source:
        return ev.source
    if excerpt_in_source(ev.quote, trace.task, trace.instructions):
        return "instructions"
    if ev.step < 0 or ev.step >= len(trace.steps):
        return ""
    step = trace.steps[ev.step]
    if step.kind == "tool_result":
        return "tool_result"
    if step.kind == "tool_call":
        return "tool_call"
    if step.kind == "message" and step.role == "user":
        return "user_message"
    if step.kind == "message":
        return "agent_message"
    return step.kind


def annotate_evidence_sources(trace: CanonicalTrace, evidence: list[Evidence]) -> None:
    for ev in evidence:
        if not ev.source:
            ev.source = classify_evidence_source(trace, ev)


def unique_evidence(items: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[int, str]] = set()
    out: list[Evidence] = []
    for ev in items:
        key = (ev.step, (ev.quote or "").strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out
