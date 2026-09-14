from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalStep, ToolSpec

TRUNC_MARK = '\n[...TRUNCATED {} chars...]\n'

def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return (text, False)
    head = int(max_chars * 0.7)
    tail = max_chars - head - 60
    omitted = len(text) - head - tail
    return (text[:head] + TRUNC_MARK.format(omitted) + text[-tail:], True)

def _block_to_text(block: Any, depth: int=0) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, (int, float, bool)) or block is None:
        return '' if block is None else str(block)
    if isinstance(block, bytes):
        return block.decode('utf-8', errors='replace')
    if isinstance(block, list):
        return '\n'.join((p for p in (_block_to_text(b, depth + 1) for b in block) if p))
    if isinstance(block, dict) and depth < 6:
        btype = str(block.get('type', '')).lower()
        if btype == 'text' and isinstance(block.get('text'), str):
            return block['text']
        if btype in ('image', 'image_url', 'audio', 'file', 'video'):
            return f'[{btype} omitted]'
        if 'text' in block and isinstance(block['text'], str):
            return block['text']
        if 'content' in block:
            inner = _block_to_text(block['content'], depth + 1)
            if inner:
                return inner
        if btype in ('tool_use', 'tool_call') or 'input' in block:
            name = block.get('name', '')
            try:
                args = json.dumps(block.get('input', block.get('arguments', {})), ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                args = str(block.get('input', ''))
            return f'{name} {args}'.strip()
        try:
            return json.dumps(block, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(block)
    try:
        return json.dumps(block, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(block)

def content_to_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _block_to_text(value)

def maybe_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t[:1] in ('[', '{'):
        try:
            return json.loads(t)
        except (json.JSONDecodeError, ValueError):
            return text
    return text
_ERROR_STATUS = {'failed', 'error', 'failure', 'aborted', 'abort', 'timeout', 'exception', 'cancelled', 'canceled', 'denied', 'rejected'}
_TEXT_HINTS = ('"error"', 'error:', 'exception', 'traceback', 'failed', 'failure', 'invalid', 'not found', 'permission denied', 'unauthorized', 'cannot be executed', 'no such', 'does not exist', 'timed out')
_NEGATIONS = ('no error', '0 error', 'without error', 'error-free', 'no errors')

def _text_has_error(s: str) -> bool:
    low = s[:800].lower()
    if any((neg in low for neg in _NEGATIONS)):
        return False
    return any((h in low for h in _TEXT_HINTS))

def detect_error(value: Any, _depth: int=0) -> bool:
    if _depth > 5:
        return False
    obj = value
    if isinstance(value, str):
        parsed = maybe_json(value)
        obj = parsed if isinstance(parsed, (dict, list)) else value
    if isinstance(obj, dict):
        if obj.get('is_error') is True or obj.get('isError') is True:
            return True
        if obj.get('success') is False or obj.get('ok') is False:
            return True
        err = obj.get('error')
        if err is True or (isinstance(err, (str, dict, list)) and err):
            return True
        st = str(obj.get('status') or obj.get('state') or '').strip().lower()
        if st in _ERROR_STATUS:
            return True
        code = obj.get('code') or obj.get('status_code') or obj.get('http_status')
        if isinstance(code, int) and code >= 400:
            return True
        for v in obj.values():
            if isinstance(v, str):
                if _text_has_error(v):
                    return True
            elif isinstance(v, (dict, list)):
                if detect_error(v, _depth + 1):
                    return True
        return False
    if isinstance(obj, list):
        return any((detect_error(x, _depth + 1) for x in obj))
    if isinstance(obj, str):
        return _text_has_error(obj)
    return False

def make_step(index: int, kind: str, *, role: str='', name: str='', content: str='', arguments: dict | None=None, call_id: str='', is_error: bool=False, source_ref: str='', time: str='', max_chars: int=6000, meta: dict | None=None) -> CanonicalStep:
    raw_len = len(content)
    body, truncated = truncate(content, max_chars)
    return CanonicalStep(index=index, kind=kind, role=role, name=name, content=body, arguments=arguments, call_id=call_id, is_error=is_error, truncated=truncated, raw_len=raw_len, source_ref=source_ref, time=time, meta=meta or {})

def normalize_tools(raw: Any) -> list[ToolSpec]:
    raw = maybe_json(raw)
    specs: list[ToolSpec] = []
    if not isinstance(raw, list):
        return specs
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get('function') if isinstance(item.get('function'), dict) else item
        name = fn.get('name') or ''
        if not name:
            continue
        params = fn.get('parameters') or fn.get('input_schema') or {}
        if not isinstance(params, dict):
            params = {}
        specs.append(ToolSpec(name=str(name), description=str(fn.get('description') or ''), parameters=params))
    return specs

def parse_arguments(raw: Any) -> dict | None:
    raw = maybe_json(raw)
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == '':
        return {}
    return {'_raw': raw}
