from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'mlflow-trace'

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

def _resolve_messages(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ('messages', 'prompt'):
        val = payload.get(key)
        if isinstance(val, list):
            return [m for m in val if isinstance(m, dict)]
    return []

def _response_message(payload: Any) -> dict | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get('choices')
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get('message')
            if isinstance(msg, dict):
                return msg
    if isinstance(payload.get('message'), dict):
        return payload['message']
    return None

def _walk_emit(span: dict, idx_start: int, max_step_chars: int, tools_seen: set[str], tools: list[ToolSpec], request_msgs: list[dict], pending_tool_calls: dict[str, dict], ref_base: str) -> tuple[list, int]:
    steps: list = []
    idx = idx_start
    for msg in request_msgs:
        role = str(msg.get('role') or '').lower()
        content = content_to_text(msg.get('content'))
        steps.append(make_step(idx, 'message', role={'user': 'user', 'assistant': 'assistant', 'ai': 'assistant', 'system': 'system'}.get(role, role or 'user'), content=content, source_ref=f'{ref_base}.events[chat_completion_request]', max_chars=max_step_chars))
        idx += 1
        if role == 'user':
            msg.setdefault('_captured_task', True)
    return (steps, idx)

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"mlflow-trace adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        if not text:
            continue
        rec = _parse(text, path, gaps)
        if rec:
            docs.append(rec)
    task = ''
    instructions = ''
    tools: list[ToolSpec] = []
    seen_tools: set[str] = set()
    steps: list = []
    idx = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get('tools'), list):
            for spec in normalize_tools(doc['tools']):
                if spec.name and spec.name not in seen_tools:
                    tools.append(spec)
                    seen_tools.add(spec.name)
        spans = doc.get('spans')
        if not isinstance(spans, list):
            continue
        if not task:
            for sp in spans:
                if not isinstance(sp, dict):
                    continue
                for ev in sp.get('events') or []:
                    if not isinstance(ev, dict):
                        continue
                    if ev.get('type') == 'chat_completion_request':
                        for msg in _resolve_messages(ev.get('payload')):
                            if str(msg.get('role', '')).lower() == 'user':
                                text = content_to_text(msg.get('content'))
                                if text.strip():
                                    task = text
                                    break
                    if task:
                        break
            if not task and isinstance(doc.get('request'), dict):
                req = doc['request']
                if isinstance(req.get('messages'), list):
                    for msg in req['messages']:
                        if str(msg.get('role', '')).lower() == 'user':
                            text = content_to_text(msg.get('content'))
                            if text.strip():
                                task = text
                                break
                if not task:
                    task = content_to_text(req)
        by_id: dict[str, dict] = {}
        for sp in spans:
            if isinstance(sp, dict) and isinstance(sp.get('span_id'), str):
                by_id[sp['span_id']] = sp
        ordered = sorted([sp for sp in spans if isinstance(sp, dict)], key=lambda s: (str(s.get('start_time_ns') or s.get('start_time') or ''), str(s.get('span_id') or '')))
        pending_calls: dict[str, dict] = {}
        for sp in ordered:
            sid = str(sp.get('span_id') or '')
            sname = str(sp.get('name') or '')
            stype = str(sp.get('span_type') or sp.get('type') or '').upper()
            ref = f'spans[{sid or sname}]'
            events = sp.get('events') or []
            if not isinstance(events, list):
                continue
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                etype = str(ev.get('type') or ev.get('name') or '')
                if etype == 'chat_completion_request':
                    payload = ev.get('payload') or ev.get('request') or ev.get('data') or {}
                elif etype == 'chat_completion_response':
                    payload = ev.get('payload') or ev.get('response') or ev.get('data') or {}
                elif etype in ('tool_request', 'tool_response'):
                    payload = ev.get('payload') or ev.get('request') or ev.get('response') or ev.get('data') or ev
                else:
                    payload = ev.get('payload') or ev.get('data') or {}
                if etype == 'chat_completion_request':
                    if isinstance(payload, dict) and isinstance(payload.get('tools'), list):
                        for spec in normalize_tools(payload['tools']):
                            if spec.name and spec.name not in seen_tools:
                                tools.append(spec)
                                seen_tools.add(spec.name)
                    for msg in _resolve_messages(payload):
                        role = str(msg.get('role') or '').lower()
                        content = content_to_text(msg.get('content'))
                        if role == 'system' and (not instructions) and content.strip():
                            instructions = content
                            continue
                        if content.strip():
                            steps.append(make_step(idx, 'message', role={'user': 'user', 'assistant': 'assistant', 'ai': 'assistant', 'system': 'system', 'tool': 'tool'}.get(role, role or 'user'), content=content, source_ref=f'{ref}.events[chat_completion_request]', max_chars=max_step_chars, meta={'span_type': stype, 'name': sname} if sname or stype else None))
                            idx += 1
                elif etype == 'chat_completion_response':
                    msg = _response_message(payload)
                    if isinstance(msg, dict):
                        content = content_to_text(msg.get('content'))
                        if content.strip():
                            steps.append(make_step(idx, 'message', role='assistant', content=content, source_ref=f'{ref}.events[chat_completion_response]', max_chars=max_step_chars))
                            idx += 1
                        for tc in msg.get('tool_calls') or []:
                            if not isinstance(tc, dict):
                                continue
                            fn = tc.get('function') or tc
                            name = str(fn.get('name') or '')
                            if name and name not in seen_tools:
                                tools.append(ToolSpec(name=name))
                                seen_tools.add(name)
                            cid = str(tc.get('id') or '')
                            args = fn.get('arguments')
                            pending_calls[cid] = {'name': name, 'args': args, 'ref': f'{ref}.events[chat_completion_response]'}
                elif etype == 'tool_request':
                    name = str(payload.get('name') or payload.get('tool_name') or sp.get('name') or '')
                    if name and name not in seen_tools:
                        tools.append(ToolSpec(name=name))
                        seen_tools.add(name)
                    cid = str(payload.get('id') or payload.get('call_id') or sid or '')
                    args = payload.get('arguments') or payload.get('input')
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref=f'{ref}.events[tool_request]', max_chars=max_step_chars, meta={'span_type': stype}))
                    idx += 1
                    pending_calls[cid] = {'name': name, 'args': args, 'ref': ref}
                elif etype == 'tool_response':
                    cid = str(payload.get('id') or payload.get('call_id') or sid or '')
                    info = pending_calls.pop(cid, {})
                    name = str(payload.get('name') or info.get('name') or '')
                    body = content_to_text(payload.get('output') or payload.get('result') or payload)
                    steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=cid, is_error=detect_error(payload.get('output') or payload.get('result')), source_ref=f'{ref}.events[tool_response]', max_chars=max_step_chars, meta={'span_type': stype}))
                    idx += 1
                elif etype == 'span_input' and stype != 'LLM':
                    body = content_to_text(payload)
                    if body.strip():
                        steps.append(make_step(idx, 'message', role='assistant', content=body, source_ref=f'{ref}.events[span_input]', max_chars=max_step_chars, meta={'span_type': stype}))
                        idx += 1
                elif etype == 'span_output' and stype != 'LLM':
                    body = content_to_text(payload)
                    if body.strip():
                        steps.append(make_step(idx, 'message', role='assistant', content=body, source_ref=f'{ref}.events[span_output]', max_chars=max_step_chars, meta={'span_type': stype}))
                        idx += 1
        if pending_calls:
            for cid, info in pending_calls.items():
                gaps.append(f"tool call '{info.get('name')}' ({cid}) unmatched by any tool_response span")
    if not docs:
        gaps.append('no MLflow trace records recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_spans': len(docs) and sum((len(d.get('spans') or []) for d in docs if isinstance(d, dict)))})
