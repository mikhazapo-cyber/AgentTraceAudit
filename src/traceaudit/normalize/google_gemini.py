from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'google-gemini'

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
        return recs
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        gaps.append(f'{path}: unparsable JSON')
        return None

def _resolve_parts(part: Any) -> list[dict]:
    if isinstance(part, dict):
        return [part]
    if isinstance(part, str):
        return [{'text': part}]
    if isinstance(part, list):
        return [p for p in part if isinstance(p, dict)]
    return [{'text': str(part)}]

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"google-gemini adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        rec = _parse(text, path, gaps)
        if rec:
            docs.append(rec)
    tools: list[ToolSpec] = []
    task = ''
    instructions = ''
    steps: list = []
    idx = 0
    last_call_name = ''
    fallback_pending_calls: dict[str, dict] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get('system_instruction'), dict):
            sys_text = content_to_text(doc['system_instruction'].get('parts'))
            if sys_text.strip() and (not instructions):
                instructions = sys_text
        if isinstance(doc.get('tools'), list):
            for t in doc['tools']:
                if not isinstance(t, dict):
                    continue
                decls = t.get('functionDeclarations') or t.get('function_declarations') or []
                if isinstance(decls, list):
                    tools.extend(normalize_tools(decls))
        conts = doc.get('contents')
        if not isinstance(conts, list):
            continue
        for c in conts:
            if not isinstance(c, dict):
                continue
            role = str(c.get('role') or '').lower()
            parts = _resolve_parts(c.get('parts'))
            if role == 'user' and (not task):
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get('text'), str):
                        task = p['text']
                        break
            for p in parts:
                if not isinstance(p, dict):
                    continue
                if 'text' in p and (not ('functionCall' in p or 'functionResponse' in p or 'inlineData' in p)):
                    body = content_to_text(p.get('text'))
                    if body.strip():
                        steps.append(make_step(idx, 'message', role={'user': 'user', 'model': 'assistant'}.get(role, 'assistant'), content=body, source_ref='contents[parts]', max_chars=max_step_chars))
                        idx += 1
                if 'functionCall' in p:
                    fc = p['functionCall']
                    name = str(fc.get('name') or '')
                    last_call_name = name or last_call_name
                    cid = str(fc.get('id') or '')
                    args = fc.get('args') or {}
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(args), call_id=cid, source_ref='contents[functionCall]', max_chars=max_step_chars))
                    if cid:
                        fallback_pending_calls[cid] = {'name': name}
                    else:
                        pass
                    idx += 1
                if 'functionResponse' in p:
                    fr = p['functionResponse']
                    name = str(fr.get('name') or '')
                    cid = str(fr.get('id') or '')
                    response = fr.get('response')
                    body = content_to_text(response)
                    steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=cid, is_error=detect_error(response), source_ref='contents[functionResponse]', max_chars=max_step_chars))
                    idx += 1
                if 'inlineData' in p:
                    steps.append(make_step(idx, 'message', role='assistant', content='[inlineData omitted: ' + str(p['inlineData'].get('mimeType') or '') + ']', source_ref='contents[inlineData]', max_chars=max_step_chars))
                    idx += 1
    if not docs:
        gaps.append('no Gemini records recoverable')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'unpaired_calls': len(fallback_pending_calls)})
