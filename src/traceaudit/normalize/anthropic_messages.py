from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools

_SOURCE_FORMAT = 'anthropic-messages'

def _read_doc(path: str, gaps: list[str]):
    try:
        with open(path, encoding='utf-8') as fh:
            return fh.read()
    except OSError as exc:
        gaps.append(f'{path}: unreadable ({exc})')
        return None

def _parse(text: str, path: str, gaps: list[str]):
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

def _resolve_blocks(content: Any) -> list[tuple[str, dict]]:
    if isinstance(content, str):
        return [('text_block', {'type': 'text', 'text': content})]
    if isinstance(content, list):
        return [('block', b) for b in content if isinstance(b, dict)]
    if isinstance(content, dict):
        return [('block', content)]
    return []

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"anthropic-messages adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        rec = _parse(text, path, gaps) if text else None
        if rec:
            docs.append(rec)
    tools: list[ToolSpec] = []
    task = ''
    instructions = ''
    steps: list = []
    idx = 0
    last_call_name = ''
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get('tools'), list):
            tspecs = normalize_tools(doc['tools'])
            if tspecs:
                tools = tspecs
        if 'system' in doc and isinstance(doc.get('system'), (str, list)):
            sys_text = content_to_text(doc['system'])
            if sys_text.strip() and (not instructions):
                instructions = sys_text
        msgs = doc.get('messages') or []
        if not isinstance(msgs, list):
            continue
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get('role') or '').lower()
            blocks = _resolve_blocks(m.get('content'))
            if role == 'user' and (not task) and blocks:
                for kind, b in blocks:
                    if kind in ('text_block', 'block') and isinstance(b, dict):
                        if b.get('type') == 'text' and (b.get('text') or '').strip():
                            task = b['text']
                            break
            for kind, b in blocks:
                if not isinstance(b, dict):
                    continue
                btype = b.get('type')
                if btype == 'text':
                    body = b.get('text') or ''
                    if body.strip():
                        steps.append(make_step(idx, 'message', role=role or 'assistant', content=body, source_ref=f'messages[{btype}]', max_chars=max_step_chars))
                        idx += 1
                elif btype == 'tool_use':
                    name = b.get('name') or ''
                    last_call_name = name or last_call_name
                    inp = b.get('input') or {}
                    cid = str(b.get('id') or '')
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=inp if isinstance(inp, dict) else {'_raw': inp}, call_id=cid, source_ref=f'messages[{btype}]', max_chars=max_step_chars))
                    idx += 1
                elif btype == 'tool_result':
                    cid = str(b.get('tool_use_id') or '')
                    content = b.get('content')
                    body = content_to_text(content)
                    is_err = detect_error(content)
                    steps.append(make_step(idx, 'tool_result', role='tool', name=last_call_name, content=body, call_id=cid, is_error=is_err, source_ref=f'messages[{btype}]', max_chars=max_step_chars))
                    idx += 1
                elif btype == 'thinking':
                    body = b.get('thinking') or ''
                    if body.strip():
                        steps.append(make_step(idx, 'message', role='assistant', content='[thinking] ' + body if body else '', source_ref=f'messages[{btype}]', max_chars=max_step_chars, meta={'type': 'thinking', 'signature': b.get('signature')}))
                        idx += 1
                else:
                    gaps.append(f'skipped Anthropic block type: {btype}')
    if not docs:
        gaps.append('no Anthropic messages recoverable from files')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs)})
