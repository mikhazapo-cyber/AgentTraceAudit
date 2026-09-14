from __future__ import annotations

import json
import re

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step

_ACT_RE = re.compile('Act:\\s*(bash|finish|answer)', re.IGNORECASE)
_CODE_RE = re.compile('```(?:bash|sh|shell)?\\s*\\n(.*?)```', re.DOTALL)
_ACTION_RE = re.compile('^ACTION:\\s*(.+)$', re.IGNORECASE | re.MULTILINE)
_TASK_RE = re.compile('here is your task|your task', re.IGNORECASE)

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    with open(files[0], encoding='utf-8') as fh:
        messages = json.load(fh)
    gaps: list[str] = []
    steps = []
    instructions = ''
    task = ''
    pending_call: str | None = None
    explicit_task = False
    idx = 0
    for i, msg in enumerate(messages):
        frm = str(msg.get('from', ''))
        value = content_to_text(msg.get('value', ''))
        ref = f'messages[{i}]'
        if frm == 'gpt':
            steps.append(make_step(idx, 'message', role='assistant', content=value, source_ref=ref, max_chars=max_step_chars))
            idx += 1
            act = _ACT_RE.search(value)
            action = _ACTION_RE.search(value)
            if act:
                kind_name = act.group(1).lower()
                if kind_name == 'bash':
                    m = _CODE_RE.search(value)
                    code = m.group(1).strip() if m else ''
                    if not code:
                        gaps.append(f"{ref}: 'Act: bash' without a fenced code block")
                    steps.append(make_step(idx, 'tool_call', role='assistant', name='bash', arguments={'command': code}, source_ref=ref, max_chars=max_step_chars))
                    idx += 1
                    pending_call = 'bash'
                elif kind_name == 'finish':
                    steps.append(make_step(idx, 'tool_call', role='assistant', name='finish', arguments={}, source_ref=ref, max_chars=max_step_chars))
                    idx += 1
                    pending_call = None
                else:
                    pending_call = None
            elif action:
                text = action.group(1).strip()
                steps.append(make_step(idx, 'tool_call', role='assistant', name='env_action', arguments={'action': text}, source_ref=ref, max_chars=max_step_chars))
                idx += 1
                pending_call = 'env_action'
            elif i > 0:
                gaps.append(f'{ref}: assistant message without an action directive')
        elif frm == 'human':
            if pending_call is not None:
                steps.append(make_step(idx, 'tool_result', role='tool', name=pending_call, content=value, source_ref=ref, is_error=detect_error(value), max_chars=max_step_chars))
                idx += 1
                pending_call = None
            else:
                steps.append(make_step(idx, 'message', role='user', content=value, source_ref=ref, max_chars=max_step_chars))
                idx += 1
                if i == 0:
                    instructions = value
                if _TASK_RE.search(value) and (not explicit_task):
                    task = value
                    explicit_task = True
        else:
            steps.append(make_step(idx, 'message', role=frm or 'unknown', content=value, source_ref=ref, max_chars=max_step_chars))
            idx += 1
    if not task:
        task = instructions
    return CanonicalTrace(trace_id=trace_id, source_format='agentinstruct_native', task=task, instructions=instructions, tools=[ToolSpec(name='bash', description='Execute a bash command in an Ubuntu shell.', parameters={'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command']}), ToolSpec(name='env_action', description='Emit a text action to the interactive environment (ALFWorld-style); must be one of the available actions.', parameters={'type': 'object', 'properties': {'action': {'type': 'string'}}, 'required': ['action']}), ToolSpec(name='finish', description='Signal that the task is complete.', parameters={})], steps=steps, capture_gaps=gaps, meta={'protocol': 'Think/Act or THOUGHT/ACTION text protocol'})
