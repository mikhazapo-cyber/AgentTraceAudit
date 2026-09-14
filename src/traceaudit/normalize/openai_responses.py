from __future__ import annotations

import json

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'openai-responses'
_TOOL_BLOCKS = {'function_call', 'web_search_call', 'file_search_call', 'code_interpreter_call', 'mcp_call', 'computer_call', 'image_generation_call'}

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
        recs = json.loads(text)
    except json.JSONDecodeError:
        gaps.append(f'{path}: unparsable JSON')
        return None
    if not isinstance(recs, list):
        return [recs]
    return recs

def _resolve_records(records: list) -> list[dict]:
    by_prev: dict[str, dict] = {}
    for r in records:
        if isinstance(r, dict) and r.get('previous_response_id'):
            by_prev[str(r['previous_response_id'])] = r
    out: list[dict] = []
    roots = [r for r in records if isinstance(r, dict) and (not r.get('previous_response_id'))]
    for r in roots:
        cur: dict | None = r
        while isinstance(cur, dict) and cur.get('id'):
            out.append(cur)
            cur = by_prev.get(str(cur.get('id')))
    remaining = [r for r in records if isinstance(r, dict) and r not in out]
    out.extend(remaining)
    return out

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"openai-responses adapter: '{source}' (alias mapped)"]
    records: list[dict] = []
    for path in files:
        text = _read_doc(path, gaps)
        recs = _parse(text, path, gaps) if text else None
        if recs:
            records.extend([r for r in recs if isinstance(r, dict)])
    tools: list[ToolSpec] = []
    for r in records:
        if isinstance(r.get('tools'), list):
            tspecs = normalize_tools(r['tools'])
            if tspecs:
                tools = tspecs
                break
    task = ''
    instructions = ''
    outcome = ''
    steps: list = []
    pairing: dict[str, str] = {}
    idx = 0
    last_call_name = ''
    ordered = _resolve_records(records)
    skipped_blocks: list[str] = []
    for rec in ordered:
        if isinstance(rec.get('instructions'), str) and (not instructions):
            instructions = rec['instructions']
        if isinstance(rec.get('input'), str) and (not task):
            task = rec['input']
        output = rec.get('output')
        if not isinstance(output, list):
            continue
        for block in output:
            if not isinstance(block, dict):
                skipped_blocks.append('output_block:non-dict')
                continue
            btype = block.get('type')
            if btype == 'message':
                content = content_to_text(block.get('content'))
                if content.strip():
                    steps.append(make_step(idx, 'message', role='assistant', content=content, source_ref=f'output[{btype}]', max_chars=max_step_chars))
                    idx += 1
            elif btype == 'reasoning':
                content = content_to_text(block.get('summary') or block.get('content') or '')
                if content.strip():
                    steps.append(make_step(idx, 'message', role='assistant', content='[reasoning] ' + content if content else '', source_ref=f'output[{btype}]', max_chars=max_step_chars, meta={'reasoning_tokens': block.get('usage')} if isinstance(block.get('usage'), dict) else None))
                    idx += 1
            elif btype in _TOOL_BLOCKS:
                name = block.get('name') or block.get('tool') or btype.replace('_call', '')
                args = block.get('arguments') or block.get('input') or block.get('arguments') or {}
                if not isinstance(args, dict):
                    args = parse_arguments(args)
                cid = str(block.get('call_id') or block.get('id') or '')
                last_call_name = name or last_call_name
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=args, call_id=cid, source_ref=f'output[{btype}]', max_chars=max_step_chars))
                if cid:
                    pairing[cid] = cid
                idx += 1
            else:
                skipped_blocks.append(f'output[{btype}]')
        call_results = rec.get('call_results')
        if isinstance(call_results, list):
            for cr in call_results:
                if not isinstance(cr, dict):
                    continue
                cid = str(cr.get('call_id') or cr.get('id') or '')
                content = content_to_text(cr.get('output') or cr.get('result'))
                steps.append(make_step(idx, 'tool_result', role='tool', name=last_call_name, content=content, call_id=cid, is_error=detect_error(cr.get('output') or cr.get('result')), source_ref='call_results', max_chars=max_step_chars))
                idx += 1
        if isinstance(rec.get('status'), str) and (not outcome):
            outcome = rec.get('status', '')
    if not records:
        gaps.append('no OpenAI Responses records recovered; trace empty')
    if skipped_blocks:
        gaps.append(f"skipped unrecognized output blocks: {','.join(sorted(set(skipped_blocks)))[:120]}")
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, outcome=outcome, capture_gaps=gaps, meta={'n_records': len(records), 'skipped': skipped_blocks})
