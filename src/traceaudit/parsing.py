from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Best-effort JSON object/array extraction from a model reply."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    raw = text.strip()
    fences = _FENCE.findall(raw)
    candidates = fences + [raw]
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
        start = min(starts)
        snippet = cand[start:]
        decoder = json.JSONDecoder()
        try:
            value, _end = decoder.raw_decode(snippet)
            return value
        except json.JSONDecodeError:
            for end in range(len(snippet), start, -1):
                try:
                    return json.loads(snippet[:end])
                except json.JSONDecodeError:
                    continue
    raise ValueError("no JSON object in model output")
