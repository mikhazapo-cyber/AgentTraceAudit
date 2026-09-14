from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'litellm-telemetry'

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

def _span_attributes(span: dict) -> dict:
    attrs = span.get('attributes')
    if isinstance(attrs, dict):
        return attrs
    if isinstance(attrs, list):
        out: dict = {}
        for kv in attrs:
            if isinstance(kv, dict):
                key = kv.get('key')
                if isinstance(key, str):
                    out[key] = kv.get('value')
        return out
    return {}

def _maybe_messages(val: Any) -> list[dict]:
    if not isinstance(val, list):
        return []
    out: list[dict] = []
    for entry in val:
        if isinstance(entry, dict):
            out.append(entry)
        elif isinstance(entry, str):
            try:
                parsed = json.loads(entry)
                if isinstance(parsed, dict):
                    out.append(parsed)
                else:
                    out.append({'role': 'user', 'content': entry})
            except (json.JSONDecodeError, ValueError):
                out.append({'role': 'user', 'content': entry})
    return out

def _response_choices(val: Any) -> list[dict]:
    if isinstance(val, list):
        if val and isinstance(val[0], dict) and ('message' in val[0]):
            return val
        if val and isinstance(val[0], dict) and ('text' in val[0] or 'type' in val[0]):
            return [{'message': {'role': 'assistant', 'content': val}}]
        return []
    if isinstance(val, dict):
        choices = val.get('choices')
        if isinstance(choices, list):
            return [c for c in choices if isinstance(c, dict)]
        if isinstance(val.get('content'), list):
            return [{'message': {'role': 'assistant', 'content': val.get('content')}}]
    return []

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"litellm-telemetry adapter: '{source}' (alias mapped)"]
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
        tid = str(doc.get('trace_id') or '')
        spans = doc.get('spans')
        if not isinstance(spans, list):
            spans = []
        calls = doc.get('calls') or []
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    spans.append({'name': 'litellm', 'type': 'litellm', 'attributes': {'model': call.get('model'), 'messages': call.get('messages'), 'response': call.get('response'), 'tool_calls': call.get('tool_calls')}})
        spans = [s for s in spans if isinstance(s, dict)]
        spans.sort(key=lambda s: (str(s.get('start') or s.get('start_time') or s.get('start_time_ns') or ''), str(s.get('span_id') or s.get('id') or '')))
        pending_calls: dict[str, dict] = {}
        for sp in spans:
            stype = str(sp.get('type') or sp.get('span_type') or sp.get('name') or '').lower()
            if stype and 'litellm' not in stype and (stype not in ('llm', 'completion', 'chat')):
                continue
            attrs = _span_attributes(sp)
            sid = str(sp.get('span_id') or sp.get('id') or '')
            ref = f"spans[{sid or stype or 'litellm'}]"
            model = str(attrs.get('model') or sp.get('model') or '')
            tools_attr = attrs.get('tools')
            if isinstance(tools_attr, list):
                for spec in normalize_tools(tools_attr):
                    if spec.name and spec.name not in seen_tools:
                        tools.append(spec)
                        seen_tools.add(spec.name)
            for msg in _maybe_messages(attrs.get('messages')):
                role = str(msg.get('role') or '').lower()
                content = content_to_text(msg.get('content'))
                if role == 'system' and (not instructions) and content.strip():
                    instructions = content
                    continue
                if role == 'user' and (not task) and content.strip():
                    task = content
                if content.strip():
                    steps.append(make_step(idx, 'message', role={'user': 'user', 'assistant': 'assistant', 'ai': 'assistant', 'system': 'system'}.get(role, role or 'user'), content=content, source_ref=f'{ref}.messages', max_chars=max_step_chars, meta={'model': model, 'trace_id': tid} if model or tid else None))
                    idx += 1
            for choice in _response_choices(attrs.get('response')):
                msg = choice.get('message') if isinstance(choice, dict) else None
                if not isinstance(msg, dict):
                    continue
                content = content_to_text(msg.get('content'))
                if content.strip():
                    steps.append(make_step(idx, 'message', role='assistant', content=content, source_ref=f'{ref}.response', max_chars=max_step_chars, meta={'model': model} if model else None))
                    idx += 1
                for tc in msg.get('tool_calls') or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get('function') or tc
                    name = str(fn.get('name') or '')
                    if name and name not in seen_tools:
                        tools.append(ToolSpec(name=name))
                        seen_tools.add(name)
                    cid = str(tc.get('id') or f'litellm-{sid}-{len(pending_calls)}')
                    args = fn.get('arguments')
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref=f'{ref}.response.tool_calls', max_chars=max_step_chars))
                    idx += 1
                    pending_calls[cid] = {'name': name, 'ref': ref}
            for tc in _maybe_messages(attrs.get('tool_calls')):
                fn = tc.get('function') or tc
                name = str(fn.get('name') or '')
                if name and name not in seen_tools:
                    tools.append(ToolSpec(name=name))
                    seen_tools.add(name)
                cid = str(tc.get('id') or f'litellm-{sid}-{len(pending_calls)}')
                args = fn.get('arguments')
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args) if args is not None else None, call_id=cid, source_ref=f'{ref}.tool_calls', max_chars=max_step_chars))
                idx += 1
                pending_calls[cid] = {'name': name, 'ref': ref}
            tool_results = attrs.get('tool_results') or attrs.get('tool_responses')
            if isinstance(tool_results, list):
                for entry in tool_results:
                    if not isinstance(entry, dict):
                        continue
                    cid = str(entry.get('id') or entry.get('call_id') or '')
                    info = pending_calls.pop(cid, {})
                    name = str(info.get('name') or entry.get('name') or '')
                    body = content_to_text(entry.get('output') or entry.get('result') or entry)
                    steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=cid, is_error=detect_error(entry), source_ref=f'{ref}.tool_results', max_chars=max_step_chars))
                    idx += 1
            if 'litellm_params' in attrs and isinstance(attrs['litellm_params'], dict):
                params = attrs['litellm_params']
                for spec in normalize_tools(params.get('tools') or []):
                    if spec.name and spec.name not in seen_tools:
                        tools.append(spec)
                        seen_tools.add(spec.name)
            metadata = attrs.get('metadata') or sp.get('metadata')
            if isinstance(metadata, dict):
                if not task and metadata.get('user_request'):
                    task = content_to_text(metadata['user_request'])
                if metadata.get('error'):
                    steps.append(make_step(idx, 'message', role='assistant', content=content_to_text(metadata['error']), is_error=True, source_ref=f'{ref}.metadata.error', max_chars=max_step_chars))
                    idx += 1
        if pending_calls:
            for cid, info in pending_calls.items():
                gaps.append(f"litellm tool call '{info.get('name')}' ({cid}) had no tool_results entry")
    if not docs:
        gaps.append('no LiteLLM telemetry records recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_spans': sum((len(d.get('spans') or []) if isinstance(d, dict) else 0 for d in docs))})
