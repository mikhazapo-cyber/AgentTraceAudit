from __future__ import annotations

import json

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'langchain'

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
    if path.endswith('.jsonl'):
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

def _find_messages(doc) -> list[dict]:
    if isinstance(doc, list) and all((isinstance(v, dict) for v in doc[:8] or [])):
        return [v for v in doc if isinstance(v, dict)]
    if isinstance(doc, dict):
        for key in ('messages', 'history', 'state', 'values'):
            val = doc.get(key)
            if isinstance(val, dict):
                inner = val.get('messages')
                if isinstance(inner, list):
                    return [m for m in inner if isinstance(m, dict)]
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                return [m for m in val if isinstance(m, dict)]
    return []

def _flatten_kwargs(msg: dict) -> dict:
    if isinstance(msg.get('kwargs'), dict):
        flat = {k: v for k, v in msg['kwargs'].items()}
        if isinstance(flat.get('id'), list) and flat['id']:
            flat['id'] = flat['id'][-1]
        flat.pop('type', None)
        flat.pop('id', None)
        flat['__type__'] = msg.get('type') or flat.get('type') or ''
        return flat
    return msg

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"langchain adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        doc = _parse(text, path, gaps)
        if doc:
            docs.append(doc)
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
        msgs = _find_messages(doc)
        for raw in msgs:
            msg = _flatten_kwargs(raw) if isinstance(raw, dict) else {}
            if not msg:
                continue
            n_msgs += 1
            mtype = (msg.get('__type__') or '').lower()
            role = str(msg.get('role') or msg.get('type') or '').lower()
            if not role and mtype in ('human', 'ai', 'tool', 'system'):
                role = 'user' if mtype == 'human' else 'tool' if mtype == 'tool' else 'system' if mtype == 'system' else 'assistant'
                mtype = mtype + 'message'
            tools_calls = msg.get('tool_calls') or msg.get('tool_call_chunks') or []
            content = msg.get('content') or msg.get('text') or ''
            if isinstance(content, list) or not isinstance(content, str):
                content = content_to_text(content)
            is_system = mtype.endswith('systemmessage') or role == 'system'
            is_human = mtype.endswith('humanmessage') or role in ('user', 'human')
            is_ai = mtype.endswith('aimessage') or role in ('assistant', 'ai')
            is_tool = mtype.endswith('toolmessage') or role == 'tool' or mtype.endswith('toolresultmessage')
            if is_system and (not instructions) and isinstance(content, str) and content.strip():
                instructions = content
                continue
            if is_human and (not task) and isinstance(content, str) and content.strip():
                task = content
            for tc in tools_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or tc
                name = str(fn.get('name') or '')
                args = fn.get('arguments') or fn.get('args') or fn.get('input')
                cid = str(tc.get('id') or tc.get('tool_call_id') or '')
                last_call_name = name or last_call_name
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref='messages[tool_calls]', max_chars=max_step_chars))
                idx += 1
            if is_tool and isinstance(content, str):
                cid = str(msg.get('tool_call_id') or msg.get('call_id') or '')
                steps.append(make_step(idx, 'tool_result', role='tool', name=last_call_name, content=content, call_id=cid, is_error=detect_error(content), source_ref='messages[tool]', max_chars=max_step_chars))
                idx += 1
            elif (is_human or is_ai) and isinstance(content, str) and content.strip():
                steps.append(make_step(idx, 'message', role='user' if is_human else 'assistant', content=content, source_ref='messages[content]', max_chars=max_step_chars, meta={'type': mtype} if mtype else None))
                idx += 1
    if not docs:
        gaps.append('no LangChain messages recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_messages': n_msgs})
