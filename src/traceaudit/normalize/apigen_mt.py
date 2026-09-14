from __future__ import annotations

import json

from ..schemas import CanonicalTrace
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments


def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    with open(files[0], encoding='utf-8') as fh:
        doc = json.load(fh)
    gaps: list[str] = []
    instructions = content_to_text(doc.get('system', ''))
    tools = normalize_tools(doc.get('tools'))
    conversations = doc.get('conversations') or []
    if not isinstance(conversations, list):
        gaps.append('conversations field missing or not a list')
        conversations = []
    steps = []
    task = ''
    idx = 0
    last_call = ''
    for i, turn in enumerate(conversations):
        frm = str(turn.get('from', ''))
        value = turn.get('value', '')
        ref = f'conversations[{i}]'
        if frm == 'human':
            text = content_to_text(value)
            if not task:
                task = text
            steps.append(make_step(idx, 'message', role='user', content=text, source_ref=ref, max_chars=max_step_chars))
            idx += 1
        elif frm == 'gpt':
            steps.append(make_step(idx, 'message', role='assistant', content=content_to_text(value), source_ref=ref, max_chars=max_step_chars))
            idx += 1
        elif frm == 'function_call':
            parsed = parse_arguments(value)
            name = ''
            args: dict = {}
            if isinstance(parsed, dict):
                name = str(parsed.get('name') or '')
                raw_args = parsed.get('arguments')
                args = parse_arguments(raw_args) or {}
                if '_raw' in args and name == '':
                    name = ''
            if not name:
                gaps.append(f'{ref}: function_call without a resolvable tool name')
            steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=args, source_ref=ref, max_chars=max_step_chars))
            last_call = name
            idx += 1
        elif frm == 'observation':
            steps.append(make_step(idx, 'tool_result', role='tool', name=last_call, content=content_to_text(value), is_error=detect_error(value), source_ref=ref, max_chars=max_step_chars))
            idx += 1
        elif frm == 'tool':
            steps.append(make_step(idx, 'tool_result', role='tool', name=str(turn.get('name') or last_call), content=content_to_text(value), is_error=detect_error(value), source_ref=ref, max_chars=max_step_chars))
            idx += 1
        else:
            if frm:
                gaps.append(f"{ref}: unknown turn role '{frm}' kept as message")
            steps.append(make_step(idx, 'message', role=frm or 'unknown', content=content_to_text(value), source_ref=ref, max_chars=max_step_chars))
            idx += 1
    return CanonicalTrace(trace_id=trace_id, source_format='apigen_mt', task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'multi_turn_user_sim': True})
