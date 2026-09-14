from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, parse_arguments

_SOURCE_FORMAT = 'mcp-events'

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
        obj = json.loads(text)
    except json.JSONDecodeError:
        gaps.append(f'{path}: unparsable JSON')
        return None
    if isinstance(obj, list):
        return obj
    return [obj]

def _call_argument_blob(params: Any) -> Any:
    if not isinstance(params, dict):
        return params
    if 'arguments' in params:
        return params['arguments']
    return params

def _tool_defs_from_result(result: Any) -> list[dict]:
    if not isinstance(result, dict):
        return []
    inner = result.get('tools')
    if isinstance(inner, list):
        return [t for t in inner if isinstance(t, dict)]
    return []

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"mcp-events adapter: '{source}' (alias mapped)"]
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
    pending: dict[str, dict] = {}
    n_msgs = 0
    for doc in docs:
        for msg in doc:
            if not isinstance(msg, dict):
                continue
            n_msgs += 1
            method = str(msg.get('method') or '')
            direction = 'server' if 'result' in msg or 'error' in msg else 'client' if method else ''
            mid = str(msg.get('id') or '')
            ref = f"jsonrpc:{method or 'response'}{(':' + mid if mid else '')}"
            if method == 'initialize':
                params = msg.get('params') or {}
                client_info = params.get('clientInfo') or {}
                server_info = (msg.get('result') or {}).get('serverInfo') if isinstance(msg.get('result'), dict) else {}
                info = content_to_text(client_info) or content_to_text(server_info)
                if info.strip():
                    steps.append(make_step(idx, 'message', role='system', content=info, source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                    idx += 1
                if not instructions:
                    caps = (msg.get('result') or {}).get('capabilities') if isinstance(msg.get('result'), dict) else {}
                    if isinstance(caps, dict) and caps:
                        instructions = content_to_text(caps)
                continue
            if method == 'tools/list':
                tools_raw = msg.get('params') or {}
                if isinstance(msg.get('result'), dict):
                    tools_raw = msg['result']
                for entry in _tool_defs_from_result(msg.get('result') or tools_raw):
                    name = str(entry.get('name') or '')
                    if not name:
                        continue
                    if name in seen_tools:
                        continue
                    schema = entry.get('inputSchema') or entry.get('schema') or {}
                    if not isinstance(schema, dict):
                        schema = {}
                    tools.append(ToolSpec(name=name, description=str(entry.get('description') or ''), parameters=schema))
                    seen_tools.add(name)
                steps.append(make_step(idx, 'message', role='server' if direction == 'server' else 'client', content=content_to_text(tools_raw), source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                idx += 1
                continue
            if method == 'tools/call':
                params = msg.get('params') or {}
                name = str(params.get('name') or '')
                if name and name not in seen_tools:
                    tools.append(ToolSpec(name=name))
                    seen_tools.add(name)
                args = _call_argument_blob(params)
                steps.append(make_step(idx, 'tool_call', role='client', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=mid, source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                idx += 1
                pending[mid] = {'name': name, 'ref': ref}
                continue
            if method.startswith('notifications/'):
                content = content_to_text(msg.get('params') or msg)
                if content.strip():
                    steps.append(make_step(idx, 'message', role='server', content=content, source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                    idx += 1
                continue
            if mid and mid in pending and ('result' in msg or 'error' in msg):
                info = pending.pop(mid)
                name = info['name']
                if 'error' in msg:
                    err = msg['error']
                    body = content_to_text(err)
                    is_err = True
                else:
                    body = content_to_text(msg.get('result'))
                    is_err = detect_error(msg.get('result'))
                steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=mid, is_error=is_err, source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                idx += 1
                continue
            if 'result' in msg and mid:
                body = content_to_text(msg['result'])
                if body.strip():
                    steps.append(make_step(idx, 'message', role='server', content=body, source_ref=ref, max_chars=max_step_chars, meta={'method': method or 'response'}))
                    idx += 1
                continue
            if 'error' in msg and mid:
                body = content_to_text(msg['error'])
                steps.append(make_step(idx, 'message', role='server', content=body, is_error=True, source_ref=ref, max_chars=max_step_chars, meta={'method': method or 'error'}))
                idx += 1
                continue
            if method and (not mid):
                content = content_to_text(msg.get('params') or msg)
                if content.strip():
                    steps.append(make_step(idx, 'message', role='client', content=content, source_ref=ref, max_chars=max_step_chars, meta={'method': method}))
                    idx += 1
                continue
            if isinstance(msg.get('params'), dict):
                content = content_to_text(msg['params'])
                if content.strip():
                    if not task:
                        task = content
                    steps.append(make_step(idx, 'message', role='client' if direction == 'client' else 'server', content=content, source_ref=ref, max_chars=max_step_chars))
                    idx += 1
    if not docs:
        gaps.append('no MCP JSON-RPC events recoverable')
    if pending:
        for mid, info in pending.items():
            gaps.append(f"tools/call id={mid} ({info['name']}) never received a matching response")
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_messages': n_msgs, 'unanswered_calls': len(pending)})
