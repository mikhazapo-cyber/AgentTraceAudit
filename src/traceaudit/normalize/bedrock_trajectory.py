from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'bedrock-trajectory'

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

def _model_request_messages(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    for key in ('messages', 'input', 'contents'):
        val = payload.get(key)
        if isinstance(val, list):
            return [m for m in val if isinstance(m, dict)]
    return []

def _model_response_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return content_to_text(payload)
    for key in ('output', 'completion', 'message', 'messages'):
        if key in payload:
            return content_to_text(payload[key])
    if isinstance(payload.get('content'), list):
        return content_to_text(payload['content'])
    return content_to_text(payload)

def _model_tool_calls(payload: Any) -> list[dict]:
    calls: list[dict] = []
    if not isinstance(payload, dict):
        return calls
    content = payload.get('content') or payload.get('output')
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get('type', '')).lower()
            if btype in ('tool_use', 'function_call') or 'toolUse' in block:
                inner = block.get('toolUse') if 'toolUse' in block else block
                if not isinstance(inner, dict):
                    continue
                calls.append({'id': str(inner.get('id') or inner.get('toolUseId') or ''), 'name': str(inner.get('name') or ''), 'arguments': inner.get('input') or inner.get('arguments') or {}})
    if isinstance(payload.get('tool_calls'), list):
        for tc in payload['tool_calls']:
            if not isinstance(tc, dict):
                continue
            fn = tc.get('function') or tc
            calls.append({'id': str(tc.get('id') or ''), 'name': str(fn.get('name') or ''), 'arguments': fn.get('arguments') or {}})
    return calls

def _iter_tool_invocation_pairs(invocations: list[dict]) -> list[tuple[dict, dict | None]]:
    pairs: list[tuple[dict, dict | None]] = []
    pending: dict | None = None
    for entry in invocations:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get('status') or '').lower()
        has_output = 'output' in entry or status in ('succeeded', 'success', 'ok', 'completed', 'failed', 'error')
        if has_output and pending is not None:
            pairs.append((pending, entry))
            pending = None
            continue
        if pending is not None:
            pairs.append((pending, None))
        pending = entry
    if pending is not None:
        pairs.append((pending, None))
    return pairs

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"bedrock-trajectory adapter: '{source}' (alias mapped)"]
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
    n_model = 0
    n_tool = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get('tools'), list):
            for spec in normalize_tools(doc['tools']):
                if spec.name and spec.name not in seen_tools:
                    tools.append(spec)
                    seen_tools.add(spec.name)
        session = doc.get('sessionAttributes') or {}
        if isinstance(session, dict):
            inv = session.get('invocationId') or session.get('invocation_id')
            if inv and (not task):
                task = content_to_text(inv)
        inp = doc.get('input') or doc.get('prompt')
        if isinstance(inp, dict) and (not task):
            task = content_to_text(inp.get('text') or inp.get('prompt') or inp.get('input'))
        elif isinstance(inp, str) and (not task):
            task = inp
        for inv in doc.get('modelInvocations') or doc.get('model_invocations') or []:
            if not isinstance(inv, dict):
                continue
            n_model += 1
            mid = str(inv.get('invocationId') or inv.get('invocation_id') or f'm{n_model}')
            model_name = str(inv.get('modelName') or inv.get('model') or '')
            ref_req = f'modelInvocations[{mid}].request'
            ref_resp = f'modelInvocations[{mid}].response'
            for msg in _model_request_messages(inv.get('request')):
                role = str(msg.get('role') or '').lower()
                content = content_to_text(msg.get('content'))
                if role == 'system' and (not instructions) and content.strip():
                    instructions = content
                    continue
                if role == 'user' and (not task) and content.strip():
                    task = content
                if content.strip():
                    steps.append(make_step(idx, 'message', role={'user': 'user', 'assistant': 'assistant', 'ai': 'assistant'}.get(role, role or 'assistant'), content=content, source_ref=ref_req, max_chars=max_step_chars, meta={'model': model_name} if model_name else None))
                    idx += 1
            resp_text = _model_response_text(inv.get('response'))
            if resp_text.strip():
                steps.append(make_step(idx, 'message', role='assistant', content=resp_text, source_ref=ref_resp, max_chars=max_step_chars, meta={'model': model_name} if model_name else None))
                idx += 1
            for tc in _model_tool_calls(inv.get('response')):
                cid = tc['id'] or f'tool-{n_tool}'
                steps.append(make_step(idx, 'tool_call', role='assistant', name=tc['name'], arguments=tc['arguments'] if isinstance(tc['arguments'], dict) else {'_raw': tc['arguments']}, call_id=cid, source_ref=ref_resp, max_chars=max_step_chars))
                idx += 1
        tool_pairs = _iter_tool_invocation_pairs(doc.get('toolInvocations') or doc.get('tool_invocations') or [])
        for call, result in tool_pairs:
            n_tool += 1
            cid = str(call.get('invocationId') or call.get('id') or '')
            if not cid:
                cid = f'tool-{n_tool}'
            name = str(call.get('name') or '')
            if name and name not in seen_tools:
                tools.append(ToolSpec(name=name))
                seen_tools.add(name)
            args = call.get('input')
            steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref='toolInvocations', max_chars=max_step_chars))
            idx += 1
            if result is not None:
                status = str(result.get('status') or '').lower()
                is_err = status in ('failed', 'error', 'timeout', 'exception', 'denied')
                out = result.get('output')
                body = content_to_text(out)
                if not is_err:
                    is_err = detect_error(out)
                steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=cid, is_error=is_err, source_ref='toolInvocations', max_chars=max_step_chars, meta={'status': status} if status else None))
                idx += 1
            else:
                gaps.append(f"tool invocation '{name}' ({cid}) unpaired; bedrock did not emit an adjacent result")
                steps.append(make_step(idx, 'tool_result', role='tool', name=name, content='', call_id=cid, is_error=True, source_ref='toolInvocations', max_chars=max_step_chars, meta={'orphan': True}))
                idx += 1
    if not docs:
        gaps.append('no Bedrock trajectory records recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_model_invocations': n_model, 'n_tool_invocations': n_tool})
