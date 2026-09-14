from __future__ import annotations

import json

from ..schemas import CanonicalTrace
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments


def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    with open(files[0], encoding='utf-8') as fh:
        doc = json.load(fh)
    gaps: list[str] = []
    messages = doc.get('messages') or []
    if not isinstance(messages, list):
        gaps.append('messages field missing or not a list')
        messages = []
    tools = normalize_tools(doc.get('tools'))
    steps = []
    instructions = ''
    task = ''
    idx = 0
    last_call_name = ''
    for i, msg in enumerate(messages):
        role = str(msg.get('role', ''))
        ref = f'messages[{i}]'
        if role == 'system':
            text = content_to_text(msg.get('content'))
            if not instructions:
                instructions = text
            steps.append(make_step(idx, 'message', role='system', content=text, source_ref=ref, max_chars=max_step_chars))
            idx += 1
        elif role == 'assistant':
            text = content_to_text(msg.get('content'))
            reasoning = content_to_text(msg.get('reasoning_content'))
            if reasoning:
                text = f'[reasoning]\n{reasoning}\n[/reasoning]\n{text}'.strip()
            if text.strip():
                steps.append(make_step(idx, 'message', role='assistant', content=text, source_ref=ref, max_chars=max_step_chars))
                idx += 1
            for tc in msg.get('tool_calls') or []:
                fn = (tc or {}).get('function') or {}
                name = str(fn.get('name') or '')
                args = parse_arguments(fn.get('arguments'))
                if not name:
                    gaps.append(f'{ref}: tool_call without a function name')
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=args, call_id=str(tc.get('id') or ''), source_ref=ref, max_chars=max_step_chars))
                last_call_name = name
                idx += 1
            if not text.strip() and (not msg.get('tool_calls')):
                gaps.append(f'{ref}: empty assistant message')
        elif role == 'tool':
            content = msg.get('content')
            text = content_to_text(content)
            steps.append(make_step(idx, 'tool_result', role='tool', name=str(msg.get('name') or last_call_name), content=text, call_id=str(msg.get('tool_call_id') or ''), is_error=detect_error(content), source_ref=ref, max_chars=max_step_chars))
            idx += 1
        elif role in ('user', 'human'):
            text = content_to_text(msg.get('content'))
            if not task:
                task = text
            steps.append(make_step(idx, 'message', role='user', content=text, source_ref=ref, max_chars=max_step_chars))
            idx += 1
        else:
            if role:
                gaps.append(f"{ref}: unknown role '{role}' kept as message")
            steps.append(make_step(idx, 'message', role=role or 'unknown', content=content_to_text(msg.get('content')), source_ref=ref, max_chars=max_step_chars))
            idx += 1
    return CanonicalTrace(trace_id=trace_id, source_format=f"nemotron_{source.split('-')[-1]}", task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'reasoning_flag': doc.get('reasoning'), 'uuid': doc.get('uuid')})
