from __future__ import annotations

import json

from ..schemas import CanonicalTrace, ToolSpec
from .base import (
    content_to_text,
    detect_error,
    make_step,
    maybe_json,
    normalize_tools,
    parse_arguments,
)

_ROLE_MAP = {'user': 'user', 'human': 'user', 'customer': 'user', 'assistant': 'assistant', 'gpt': 'assistant', 'bot': 'assistant', 'agent': 'assistant', 'system': 'system', 'developer': 'system', 'tool': 'tool', 'function': 'tool', 'observation': 'tool'}
_MSG_KEYS = ('messages', 'conversations', 'turns', 'dialogue', 'history', 'trace', 'steps')

def _looks_like_message(v) -> bool:
    return isinstance(v, dict) and any((k in v for k in ('role', 'from', 'content', 'value', 'text', 'tool_calls', 'function_call')))

def _read_doc(path: str, gaps: list[str]):
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as exc:
        gaps.append(f'{path}: unreadable ({exc})')
        return (None, False)
    if not text.strip():
        return (None, True)
    if path.endswith('.jsonl'):
        recs = []
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                gaps.append(f'{path}:{i}: unparsable JSONL record skipped')
        return (recs, True)
    try:
        return (json.loads(text), True)
    except json.JSONDecodeError:
        recs = []
        ok = True
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                ok = False
                break
        if ok and recs:
            return (recs, True)
        gaps.append(f'{path}: unparsable JSON')
        return (None, True)

def _find_message_list(doc) -> list:
    if isinstance(doc, list):
        if all((_looks_like_message(v) for v in doc[:5] if isinstance(v, dict))):
            return [v for v in doc if isinstance(v, dict)]
        for v in doc:
            if isinstance(v, dict):
                for key in _MSG_KEYS:
                    inner = maybe_json(v.get(key))
                    if isinstance(inner, list) and inner:
                        return inner
        return [v for v in doc if isinstance(v, dict)]
    if isinstance(doc, dict):
        for key in _MSG_KEYS:
            val = maybe_json(doc.get(key))
            if isinstance(val, list) and val:
                return val
        for val in doc.values():
            val = maybe_json(val)
            if isinstance(val, list) and val and all((isinstance(v, dict) for v in val[:3])):
                return val
    return []

def _looks_canonical(doc) -> bool:
    if not isinstance(doc, dict):
        return False
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    first = steps[0]
    return isinstance(first, dict) and "kind" in first


def load_canonical(trace_id: str, doc: dict, max_step_chars: int) -> CanonicalTrace:
    from ..schemas import CanonicalStep
    from ..schemas import CanonicalTrace as CT

    raw_steps = doc.get("steps") or []
    steps = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue
        payload = dict(s)
        payload.setdefault("index", i)
        payload.setdefault("kind", "other")
        if len(str(payload.get("content") or "")) > max_step_chars:
            payload["content"], payload["truncated"] = (
                str(payload["content"])[:max_step_chars],
                True,
            )
        steps.append(CanonicalStep.model_validate(payload))
    tools = normalize_tools(doc.get("tools") or [])
    return CT(
        trace_id=str(doc.get("trace_id") or trace_id),
        source_format=str(doc.get("source_format") or "canonical"),
        task=str(doc.get("task") or ""),
        instructions=str(doc.get("instructions") or ""),
        tools=tools,
        steps=steps,
        outcome=str(doc.get("outcome") or ""),
        capture_gaps=list(doc.get("capture_gaps") or []),
        meta=dict(doc.get("meta") or {}),
    )


def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps = [f"unknown source format '{source}': generic fallback adapter used"]
    doc = None
    any_readable = False
    for path in files:
        d, readable = _read_doc(path, gaps)
        any_readable = any_readable or readable
        if d is not None:
            doc = d
            break
    if _looks_canonical(doc):
        return load_canonical(trace_id, doc, max_step_chars)
    if files and (not any_readable):
        raise ValueError(f'none of the trace files are readable: {gaps[-1]}')
    messages = _find_message_list(doc)
    if not messages:
        gaps.append('no message-like list found; trace normalized as empty')
    tools: list[ToolSpec] = []
    if isinstance(doc, dict):
        for key in ('tools', 'available_tools', 'functions', 'tool_definitions'):
            if key in doc:
                tools = normalize_tools(doc[key])
                if tools:
                    break
    steps = []
    instructions = ''
    task = ''
    idx = 0
    last_call_name = ''
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        raw_role = str(msg.get('role') or msg.get('from') or '').lower()
        role = _ROLE_MAP.get(raw_role, '')
        ref = f'messages[{i}]'
        text = content_to_text(msg.get('content', msg.get('value', msg.get('text'))))
        if role == 'system' and (not instructions):
            instructions = text
        if role == 'user' and (not task) and text.strip():
            task = text
        for tc in msg.get('tool_calls') or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') or {}
            name = str(fn.get('name') or '')
            steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(fn.get('arguments')), call_id=str(tc.get('id') or ''), source_ref=ref, max_chars=max_step_chars))
            last_call_name = name or last_call_name
            idx += 1
        fc = msg.get('function_call') or msg.get('tool_call')
        if isinstance(fc, dict):
            name = str(fc.get('name') or '')
            steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(fc.get('arguments')), call_id=str(fc.get('id') or ''), source_ref=ref, max_chars=max_step_chars))
            last_call_name = name or last_call_name
            idx += 1
        if text.strip():
            if role == 'tool':
                steps.append(make_step(idx, 'tool_result', role='tool', name=str(msg.get('name') or last_call_name), content=text, call_id=str(msg.get('tool_call_id') or ''), is_error=detect_error(msg.get('content', msg.get('value', msg.get('text')))), source_ref=ref, max_chars=max_step_chars))
            else:
                steps.append(make_step(idx, 'message', role=role or raw_role or 'unknown', content=text, source_ref=ref, max_chars=max_step_chars))
            idx += 1
    return CanonicalTrace(trace_id=trace_id, source_format=f'generic:{source}', task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps)
