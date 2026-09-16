from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _close_truncated(snippet: str) -> Any | None:
    """O(n) close of a truncated JSON object/array. No per-character rescan."""
    s = snippet.rstrip()
    if not s:
        return None
    in_str = False
    esc = False
    stack: list[str] = []
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack:
                stack.pop()
    if in_str:
        s += '"'
    s = s.rstrip()
    if s.endswith(","):
        s = s[:-1]
    s += "".join(reversed(stack))
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def extract_json(text: str) -> Any:
    """Best-effort JSON object/array extraction from a model reply."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    raw = text.strip()
    candidates = _FENCE.findall(raw) + [raw]
    decoder = json.JSONDecoder()
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        start_obj, start_arr = cand.find("{"), cand.find("[")
        starts = [i for i in (start_obj, start_arr) if i >= 0]
        if not starts:
            continue
        snippet = cand[min(starts) :]
        try:
            value, _end = decoder.raw_decode(snippet)
            return value
        except json.JSONDecodeError:
            repaired = _close_truncated(snippet)
            if repaired is not None:
                return repaired
    raise ValueError("no JSON object in model output")
