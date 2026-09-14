from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'otel-genai'

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

def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(attrs, list):
        return out
    for entry in attrs:
        if not isinstance(entry, dict):
            continue
        key = entry.get('key') or entry.get('name')
        if not isinstance(key, str):
            continue
        val = entry.get('value')
        if isinstance(val, dict):
            for vk in ('stringValue', 'intValue', 'doubleValue', 'boolValue', 'arrayValue', 'kvlistValue'):
                if vk in val:
                    val = val[vk]
                    break
        out[key] = val
    return out

def _otel_value_to_text(val: Any) -> str:
    if val is None:
        return ''
    if isinstance(val, dict):
        if 'stringValue' in val:
            return str(val['stringValue'])
        if 'arrayValue' in val and isinstance(val['arrayValue'], dict):
            items = val['arrayValue'].get('values') or []
            return content_to_text([_otel_value_to_text(v) for v in items])
        return content_to_text(val)
    if isinstance(val, list):
        return content_to_text([_otel_value_to_text(v) for v in val])
    return str(val)

def _parse_json_attr(val: Any) -> Any:
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    return val

def _collect_spans(docs: list[dict]) -> list[dict]:
    spans: list[dict] = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for rs in doc.get('resourceSpans') or doc.get('resource_spans') or []:
            if not isinstance(rs, dict):
                continue
            for ss in rs.get('scopeSpans') or rs.get('scope_spans') or []:
                if not isinstance(ss, dict):
                    continue
                for sp in ss.get('spans') or []:
                    if isinstance(sp, dict):
                        spans.append(sp)
        for ss in doc.get('scopeSpans') or doc.get('scope_spans') or []:
            if not isinstance(ss, dict):
                continue
            for sp in ss.get('spans') or []:
                if isinstance(sp, dict):
                    spans.append(sp)
        if isinstance(doc.get('spans'), list):
            for sp in doc['spans']:
                if isinstance(sp, dict):
                    spans.append(sp)
    return spans

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"otel-genai adapter: '{source}' (alias mapped)"]
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
    spans = _collect_spans(docs)
    spans.sort(key=lambda s: (str(s.get('startTimeUnixNano') or s.get('start_time_unix_nano') or ''), str(s.get('span_id') or s.get('spanId') or '')))
    pending_calls: dict[str, dict] = {}
    seen_genai = 0
    for sp in spans:
        attrs = _attrs_to_dict(sp.get('attributes'))
        if not any((k.startswith('gen_ai.') for k in attrs)):
            continue
        seen_genai += 1
        sid = str(sp.get('span_id') or sp.get('spanId') or '')
        stid = str(sp.get('trace_id') or sp.get('traceId') or '')
        ref = f'scopeSpans/spans[{sid}]'
        model = str(attrs.get('gen_ai.response.model') or attrs.get('gen_ai.request.model') or '')
        tool_defs = _parse_json_attr(attrs.get('gen_ai.tool.definitions'))
        if isinstance(tool_defs, list):
            for spec in normalize_tools(tool_defs):
                if spec.name and spec.name not in seen_tools:
                    tools.append(spec)
                    seen_tools.add(spec.name)
        req_msgs_raw = _parse_json_attr(attrs.get('gen_ai.request.messages'))
        req_msgs: list[dict] = []
        if isinstance(req_msgs_raw, list):
            for entry in req_msgs_raw:
                if isinstance(entry, dict):
                    req_msgs.append(entry)
                elif isinstance(entry, str):
                    inner = _parse_json_attr(entry)
                    if isinstance(inner, dict):
                        req_msgs.append(inner)
                    else:
                        req_msgs.append({'role': 'user', 'content': entry})
        for msg in req_msgs:
            role = str(msg.get('role') or '').lower()
            content = content_to_text(msg.get('content') or msg.get('parts'))
            if role == 'system' and (not instructions) and content.strip():
                instructions = content
                continue
            if role == 'user' and (not task) and content.strip():
                task = content
            if content.strip():
                steps.append(make_step(idx, 'message', role={'user': 'user', 'assistant': 'assistant', 'ai': 'assistant', 'system': 'system'}.get(role, role or 'user'), content=content, source_ref=f'{ref}.gen_ai.request.messages', max_chars=max_step_chars, meta={'model': model} if model else None))
                idx += 1
        resp_choices = _parse_json_attr(attrs.get('gen_ai.response.choices'))
        if isinstance(resp_choices, list):
            for choice in resp_choices:
                if not isinstance(choice, dict):
                    continue
                inner = choice.get('message') or choice
                content = content_to_text(inner.get('content') or inner)
                if content.strip():
                    steps.append(make_step(idx, 'message', role='assistant', content=content, source_ref=f'{ref}.gen_ai.response.choices', max_chars=max_step_chars, meta={'model': model} if model else None))
                    idx += 1
                for tc in inner.get('tool_calls') or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get('function') or tc
                    name = str(fn.get('name') or '')
                    if name and name not in seen_tools:
                        tools.append(ToolSpec(name=name))
                        seen_tools.add(name)
                    cid = str(tc.get('id') or f'otel-{sid}-{len(pending_calls)}')
                    args = fn.get('arguments')
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref=f'{ref}.gen_ai.tool.call', max_chars=max_step_chars))
                    idx += 1
                    pending_calls[cid] = {'name': name, 'ref': ref}
        for key, val in attrs.items():
            if not key.startswith('gen_ai.tool.call.'):
                continue
            sub = key[len('gen_ai.tool.call.'):]
            payload = _parse_json_attr(val)
            if sub in ('id', 'name'):
                continue
            name = str(attrs.get('gen_ai.tool.call.name') or sub)
            if name and name not in seen_tools:
                tools.append(ToolSpec(name=name))
                seen_tools.add(name)
            cid = str(attrs.get('gen_ai.tool.call.id') or f'otel-{sid}')
            args = payload if isinstance(payload, (dict, str)) else {'_raw': val}
            steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref=f'{ref}.{key}', max_chars=max_step_chars))
            idx += 1
            pending_calls[cid] = {'name': name, 'ref': ref}
        for key, val in attrs.items():
            if not key.startswith('gen_ai.tool.result.'):
                continue
            sub = key[len('gen_ai.tool.result.'):]
            payload = _parse_json_attr(val)
            cid = str(attrs.get('gen_ai.tool.result.id') or f'otel-{sid}')
            info = pending_calls.pop(cid, {})
            name = str(info.get('name') or attrs.get('gen_ai.tool.result.name') or sub)
            body = content_to_text(payload)
            steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=cid, is_error=detect_error(payload), source_ref=f'{ref}.{key}', max_chars=max_step_chars, meta={'trace_id': stid} if stid else None))
            idx += 1
        tool_name = attrs.get('gen_ai.tool.name')
        if tool_name:
            tool_input = _parse_json_attr(attrs.get('gen_ai.tool.input'))
            tool_output = _parse_json_attr(attrs.get('gen_ai.tool.output'))
            cid = str(attrs.get('gen_ai.tool.call.id') or attrs.get('gen_ai.tool.result.id') or f'otel-{sid}')
            if tool_input not in (None, ''):
                steps.append(make_step(idx, 'tool_call', role='assistant', name=str(tool_name), arguments=tool_input if isinstance(tool_input, dict) else {'_raw': tool_input}, call_id=cid, source_ref=f'{ref}.gen_ai.tool.input', max_chars=max_step_chars))
                idx += 1
            if tool_output not in (None, ''):
                steps.append(make_step(idx, 'tool_result', role='tool', name=str(tool_name), content=content_to_text(tool_output), call_id=cid, is_error=detect_error(tool_output), source_ref=f'{ref}.gen_ai.tool.output', max_chars=max_step_chars))
                idx += 1
    if not docs:
        gaps.append('no OTel resourceSpans/scopespans records recoverable')
    elif seen_genai == 0:
        gaps.append('OTel record loaded but no span carried gen_ai.* attributes')
    if pending_calls:
        for cid, info in pending_calls.items():
            gaps.append(f"OTel tool call '{info.get('name')}' ({cid}) had no matching gen_ai.tool.result.* attribute")
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_spans': len(spans), 'n_genai_spans': seen_genai})
