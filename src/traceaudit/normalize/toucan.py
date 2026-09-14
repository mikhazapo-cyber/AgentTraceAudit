from __future__ import annotations

import json
import re

from ..schemas import CanonicalTrace
from .base import (
    content_to_text,
    detect_error,
    make_step,
    maybe_json,
    normalize_tools,
    parse_arguments,
)

_TOOL_DECLARE_RE = re.compile('<\\|im_system\\|>tool_declare<\\|im_middle\\|>.*?(?:<\\|im_end\\|>|$)', re.DOTALL)

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    with open(files[0], encoding='utf-8') as fh:
        doc = json.load(fh)
    gaps: list[str] = []
    messages = maybe_json(doc.get('messages'))
    if not isinstance(messages, list):
        gaps.append('messages field missing, not a list, or unparsable JSON string')
        messages = []
    tools = normalize_tools(doc.get('available_tools'))
    if not tools:
        for m in messages:
            if isinstance(m, dict) and m.get('role') == 'system':
                inner = _TOOL_DECLARE_RE.search(content_to_text(m.get('content')))
                if inner:
                    blob = inner.group(0)
                    start = blob.find('[')
                    if start >= 0:
                        tools = normalize_tools(blob[start:])
                break
    steps = []
    instructions = ''
    task = ''
    idx = 0
    last_call_name = ''
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            gaps.append(f'messages[{i}]: non-object entry skipped')
            continue
        role = str(msg.get('role', ''))
        ref = f'messages[{i}]'
        if role == 'system':
            text = _TOOL_DECLARE_RE.sub('[tool declarations omitted; see tools list]', content_to_text(msg.get('content'))).strip()
            if not instructions:
                instructions = text
            if text.strip():
                steps.append(make_step(idx, 'message', role='system', content=text, source_ref=ref, max_chars=max_step_chars))
                idx += 1
        elif role == 'assistant':
            text = content_to_text(msg.get('content'))
            if text.strip():
                steps.append(make_step(idx, 'message', role='assistant', content=text, source_ref=ref, max_chars=max_step_chars))
                idx += 1
            fc = msg.get('function_call')
            if isinstance(fc, dict):
                name = str(fc.get('name') or '')
                args = parse_arguments(fc.get('arguments'))
                if not name:
                    gaps.append(f'{ref}: function_call without a name')
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=args, source_ref=ref, max_chars=max_step_chars))
                last_call_name = name
                idx += 1
        elif role in ('function', 'tool'):
            text = content_to_text(msg.get('content'))
            steps.append(make_step(idx, 'tool_result', role='tool', name=str(msg.get('name') or last_call_name), content=text, call_id=str(msg.get('tool_call_id') or ''), is_error=detect_error(msg.get('content')), source_ref=ref, max_chars=max_step_chars))
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
    return CanonicalTrace(trace_id=trace_id, source_format='toucan_mcp', task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'subset_name': doc.get('subset_name'), 'uuid': doc.get('uuid')})
