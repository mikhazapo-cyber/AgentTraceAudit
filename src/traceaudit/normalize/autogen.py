from __future__ import annotations

import json

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'autogen'

def _read_doc(path: str, gaps: list[str]):
    try:
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    except OSError as exc:
        gaps.append(f'{path}: unreadable ({exc})')
        return None

def _parse(text: str | None, path: str, gaps: list[str]):
    if not text or not text.strip():
        return None
    is_jsonl = path.endswith('.jsonl')
    if is_jsonl:
        recs: list = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                gaps.append(f'{path}:{lineno}: unparsable JSONL record skipped')
        return recs or None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        gaps.append(f'{path}: unparsable JSON')
        return None

def _resolve_messages(doc) -> list[dict]:
    if isinstance(doc, list):
        return [m for m in doc if isinstance(m, dict)]
    if isinstance(doc, dict):
        cm = doc.get('chat_messages')
        if isinstance(cm, list) and cm and isinstance(cm[0], dict):
            return [m for m in cm if isinstance(m, dict)]
        outer = doc.get('config')
        if isinstance(outer, dict) and isinstance(outer.get('messages'), list):
            return [m for m in outer['messages'] if isinstance(m, dict)]
    return []

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"autogen adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        if not text:
            continue
        rec = _parse(text, path, gaps)
        if rec:
            docs.append(rec)
    tools: list[ToolSpec] = []
    if any((isinstance(d, dict) and isinstance(d.get('tools'), list) for d in docs)):
        for d in docs:
            ts = normalize_tools(d.get('tools') or [])
            if ts:
                tools = ts
                break
    task = ''
    instructions = ''
    steps: list = []
    idx = 0
    last_call_name = ''
    n_msgs = 0
    for doc in docs:
        msgs = _resolve_messages(doc)
        for m in msgs:
            n_msgs += 1
            role = str(m.get('role') or m.get('from') or '').lower()
            name = m.get('name') or ''
            content = content_to_text(m.get('content'))
            if role == 'system' and (not instructions) and content.strip():
                instructions = content
                continue
            if role in ('user', 'human') and (not task) and content.strip():
                task = content
            tcs = m.get('tool_calls') or []
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or tc
                tname = str(fn.get('name') or name or '')
                args = fn.get('arguments') or fn.get('args')
                cid = str(tc.get('id') or '')
                last_call_name = tname or last_call_name
                steps.append(make_step(idx, 'tool_call', role='assistant', name=tname, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref='messages[tool_calls]', max_chars=max_step_chars))
                idx += 1
            if role in ('tool', 'function') and content.strip():
                cid_next: str | None = None
                for s in reversed(steps):
                    if s.kind == 'tool_call':
                        cid_next = s.call_id or None
                        break
                steps.append(make_step(idx, 'tool_result', role='tool', name=last_call_name, content=content, call_id=str(m.get('tool_call_id') or cid_next or ''), is_error=detect_error(content), source_ref='messages[tool]', max_chars=max_step_chars))
                idx += 1
            elif content.strip() and role in ('user', 'human', 'assistant', 'ai', 'bot', 'agent'):
                steps.append(make_step(idx, 'message', role='user' if role in ('user', 'human') else 'assistant', content=content, source_ref='messages[content]', max_chars=max_step_chars, meta={'name': name} if name else None))
                idx += 1
    if not docs:
        gaps.append('no AutoGen messages recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_messages': n_msgs})
