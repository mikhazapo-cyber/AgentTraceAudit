from __future__ import annotations

import json
import re

from .schemas import CanonicalStep, CanonicalTrace, ToolSpec
from .structural import logical_tool_name, pair_calls_results


def _clip(text: str, limit: int) -> str:
    text = (text or "").replace("\r\n", "\n").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 16] + "\n[...truncated...]"


def render_tools(tools: list[ToolSpec], limit: int = 8000) -> str:
    parts = []
    for t in tools:
        params = t.parameters or {}
        req = params.get("required") or []
        props = params.get("properties") or {}
        fields = []
        for name, spec in list(props.items())[:12]:
            if not isinstance(spec, dict):
                continue
            typ = spec.get("type", "")
            desc = str(spec.get("description") or "")[:80]
            mark = " required" if name in req else ""
            fields.append(f"    - {name}: {typ}{mark} {desc}".rstrip())
        block = f"- {t.name}: {_clip(t.description, 180)}"
        if fields:
            block += "\n" + "\n".join(fields)
        elif req:
            block += f"\n    required: {req}"
        parts.append(block)
    text = "\n".join(parts) if parts else "(no declared tools)"
    return _clip(text, limit)


def render_step(step: CanonicalStep, arg_limit: int = 720, content_limit: int = 900) -> str:
    if step.kind == "tool_call":
        name = logical_tool_name(step) or step.name or "?"
        try:
            args = json.dumps(step.arguments or {}, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            args = str(step.arguments)
        return f"#{step.index} [tool_call] {name}({_clip(args, arg_limit)})"
    if step.kind == "tool_result":
        err = " ERROR" if step.is_error else ""
        return f"#{step.index} [tool_result{err}] {step.name}: {_clip(step.content, content_limit)}"
    role = step.role or "message"
    return f"#{step.index} [{step.kind} {role}] {_clip(step.content, content_limit)}"


def evidence_index(trace: CanonicalTrace, char_budget: int = 160000) -> str:
    """Step-addressable view of what the agent could see.

    Prefers the largest per-step clips that still fit the char budget. Older
    packs started at 960/1400 and left most of a 160k budget unused, so
    auditors never saw the tool results that could settle a check.
    """
    if char_budget <= 0:
        return ""
    tiers = (
        (12000, 18000),
        (6000, 9000),
        (3000, 4500),
        (1600, 2400),
        (960, 1400),
        (640, 900),
        (360, 480),
        (160, 200),
    )
    text = ""
    for arg_limit, content_limit in tiers:
        lines = [render_step(s, arg_limit=arg_limit, content_limit=content_limit) for s in trace.steps]
        text = "\n".join(lines)
        if len(text) <= char_budget:
            return text
    keep = max(64, char_budget - 40)
    head = keep * 2 // 3
    tail = keep - head
    return text[:head] + "\n[...index truncated...]\n" + text[-tail:]


def step_outline(trace: CanonicalTrace, limit: int = 4000) -> str:
    rows = []
    for s in trace.steps:
        if s.kind == "tool_call":
            rows.append(f"#{s.index} call {logical_tool_name(s) or s.name}")
        elif s.kind == "tool_result":
            tag = "err" if s.is_error else "ok"
            rows.append(f"#{s.index} result/{tag} {s.name}")
        else:
            rows.append(f"#{s.index} {s.kind}/{s.role or '-'}")
    return _clip("\n".join(rows), limit)


def info_available_at(trace: CanonicalTrace, step_index: int, limit: int = 2400) -> str:
    """What the agent had already seen when it took `step_index`."""
    pairing = pair_calls_results(trace)
    bits = []
    for s in trace.steps:
        if s.index >= step_index:
            break
        if s.kind == "tool_result":
            bits.append(f"#{s.index} {s.name}: {_clip(s.content, 480)}")
        elif s.kind == "message" and s.role in ("user", "system", "tool"):
            bits.append(f"#{s.index} {s.role}: {_clip(s.content, 280)}")
        elif s.kind == "tool_call":
            bits.append(f"#{s.index} call {logical_tool_name(s) or s.name}")
    # include the paired result of earlier calls
    _ = pairing
    return _clip("\n".join(bits[-24:]), limit)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
_OBLIGATION = re.compile(
    r"\b(must|should|shall|only|never|always|required|require|before|after|"
    r"exactly one|at most one|do not|don't|ensure|cannot|may not|please only|"
    r"not before|only after|only if|only use)\b",
    re.IGNORECASE,
)


def obligation_sentences(text: str, limit: int = 16) -> list[str]:
    if not (text or "").strip():
        return []
    units = _SENTENCE_SPLIT.split(text)
    out: list[str] = []
    seen: set[str] = set()
    for u in units:
        s = u.strip().replace("\n", " ")
        if len(s) < 24 or len(s) > 420:
            continue
        if not _OBLIGATION.search(s):
            continue
        key = s[:90].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out
