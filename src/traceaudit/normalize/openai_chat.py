from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'openai-chat'

def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, dict):
            d = dst.setdefault(k, {})
            if not isinstance(d, dict):
                dst[k] = {}
                d = dst[k]
            _deep_merge(d, v)
        elif isinstance(v, list):
            existing = dst.setdefault(k, [])
            if not isinstance(existing, list):
                dst[k] = []
                existing = dst[k]
            existing.extend(v)
        else:
            dst[k] = v

def _collapse_streaming(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for rec in records:
        if isinstance(rec, dict) and rec.get('object') == 'thread.message.delta' and out and isinstance(out[-1], dict) and (out[-1].get('object') == 'thread.message'):
            _deep_merge(out[-1], rec.get('delta') or {})
            continue
        out.append(rec)
    return out

def _resolve_record(record: dict) -> list[dict]:
    role = str(record.get('role') or '').lower()
    text = content_to_text(record.get('content'))
    parts: list[Any] = []
    if role == 'system':
        return [('system', record, text, [])]
    if role == 'user':
        parts.append(('message', record, text, []))
        return parts
    if role == 'tool':
        cid = str(record.get('tool_call_id') or '')
        return [('tool_result', record, text, [cid])]
    if role in ('assistant', 'ai', 'bot'):
        tcs = record.get('tool_calls') or []
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get('function') or {}
                name = str(fn.get('name') or '')
                cid = str(tc.get('id') or '')
                parts.append(('tool_call', record, '', [(name, cid)]))
        fc = record.get('function_call')
        if isinstance(fc, dict):
            name = str(fc.get('name') or '')
            cid = str(record.get('id') or '')
            parts.append(('tool_call', record, '', [(name, cid)]))
        if text.strip():
            parts.append(('message', record, text, []))
        return parts or [('message', record, text, [])]
    return [('message', record, text, [])]

def _read_doc(path: str, gaps: list[str]):
    try:
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
    except OSError as exc:
        gaps.append(f'{path}: unreadable ({exc})')
        return None
    if not text.strip():
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
        return recs
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return _collapse_streaming(obj)
        return [obj]
    except json.JSONDecodeError:
        try:
            obj = json.loads(text.split('}]}')[0] + '}]}')
        except json.JSONDecodeError:
            gaps.append(f'{path}: unparsable JSON')
            return None
        if isinstance(obj, list):
            return _collapse_streaming(obj)
        return [obj]

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"openai-chat adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        result = _read_doc(path, gaps)
        if result:
            docs.extend(result)
    if files and (not docs):
        gaps.append('no records recoverable from provided files')
    task = ''
    instructions = ''
    tools: list[ToolSpec] = []
    saw_tools = False
    steps: list = []
    pairing: dict[str, str] = {}
    idx = 0
    last_call_name = ''
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        msgs = doc.get('messages')
        if not isinstance(msgs, list):
            continue
        if 'object' in doc and doc['object'] not in ('thread', 'thread.run', 'chat.completion', 'chat.completion.chunk'):
            pass
        if isinstance(doc.get('tools'), list) and (not saw_tools):
            tools = normalize_tools(doc['tools'])
            if tools:
                saw_tools = True
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or '').lower()
            text = content_to_text(m.get('content'))
            if role == 'system' and (not instructions) and text.strip():
                instructions = text
            elif role == 'user' and (not task) and text.strip():
                task = text
            parts = _resolve_record(m)
            for kind, src, body, extra in parts:
                if kind == 'tool_call':
                    name, cid = extra[0]
                    last_call_name = name or last_call_name
                    fn = (m.get('tool_calls') or [{}])[0].get('function') if m.get('tool_calls') else m.get('function_call') or {}
                    args_raw = fn.get('arguments')
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args_raw) if args_raw is not None else None, call_id=cid, source_ref='messages', max_chars=max_step_chars))
                    if cid:
                        pairing[cid] = src.get('id') or ''
                    idx += 1
                elif kind == 'tool_result':
                    cid = str(m.get('tool_call_id') or '')
                    content = body
                    steps.append(make_step(idx, 'tool_result', role='tool', name=last_call_name, content=content, call_id=cid, is_error=detect_error(m.get('content')), source_ref='messages', max_chars=max_step_chars))
                    idx += 1
                elif kind == 'system':
                    if text.strip() and (not instructions):
                        instructions = text
                elif kind == 'message':
                    if text.strip():
                        steps.append(make_step(idx, 'message', role=role or 'assistant', content=text, source_ref='messages', max_chars=max_step_chars))
                        idx += 1
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_records': len(docs), 'paired_results': len(pairing)})
