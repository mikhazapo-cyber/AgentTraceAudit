from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools

_SOURCE_FORMAT = 'metagpt'

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

def _coerce_actions(raw: Any) -> list[dict]:
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get('actions'), list):
            for sub in item['actions']:
                if isinstance(sub, dict):
                    out.append(sub)
        else:
            out.append(item)
    return out

def _tool_specs_from_registry(env: dict) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    reg = env.get('tool_registry') if isinstance(env, dict) else None
    if isinstance(reg, dict):
        for name, spec in reg.items():
            if isinstance(spec, dict):
                params = spec.get('parameters') or spec.get('schema') or {}
                if not isinstance(params, dict):
                    params = {}
                specs.append(ToolSpec(name=str(name), description=str(spec.get('description') or ''), parameters=params))
            else:
                specs.append(ToolSpec(name=str(name)))
    elif isinstance(reg, list):
        for entry in reg:
            if isinstance(entry, dict):
                name = str(entry.get('name') or '')
                if name:
                    params = entry.get('parameters') or entry.get('schema') or {}
                    if not isinstance(params, dict):
                        params = {}
                    specs.append(ToolSpec(name=name, description=str(entry.get('description') or ''), parameters=params))
    return specs

def _emit_role(role: dict, role_name: str, idx_start: int, max_step_chars: int) -> tuple[list, int]:
    steps: list = []
    idx = idx_start
    actions = _coerce_actions(role.get('actions'))
    profile = role.get('profile') or role.get('name') or role_name
    role_label = (str(profile).strip() or role_name or 'assistant').lower()
    instructions = content_to_text(role.get('system_prompt') or role.get('instructions'))
    if instructions.strip() and (not getattr(_emit_role, '_seen', None)):
        steps.append(make_step(idx, 'message', role='system', name=str(profile), content=instructions, source_ref=f'roles[{role_name}].system_prompt', max_chars=max_step_chars))
        idx += 1
    for action in actions:
        ref = f"roles[{role_name}].actions[{action.get('name', '?')}]"
        instr = content_to_text(action.get('instruction'))
        if instr.strip():
            steps.append(make_step(idx, 'message', role='user', name=str(action.get('name') or ''), content=instr, source_ref=ref, max_chars=max_step_chars, meta={'role': role_label}))
            idx += 1
        output = action.get('output')
        body = content_to_text(output)
        is_err = detect_error(output)
        if body.strip():
            steps.append(make_step(idx, 'message', role=role_label, name=str(action.get('name') or ''), content=body, is_error=is_err, source_ref=ref, max_chars=max_step_chars, meta={'action': action.get('name', '')}))
            idx += 1
        memory = action.get('memory')
        mem_text = content_to_text(memory)
        if mem_text.strip() and mem_text != body:
            steps.append(make_step(idx, 'message', role=role_label, name=str(action.get('name') or ''), content=mem_text, source_ref=f'{ref}.memory', max_chars=max_step_chars, meta={'kind': 'memory'}))
            idx += 1
    return (steps, idx)

def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = [f"metagpt adapter: '{source}' (alias mapped)"]
    docs: list = []
    for path in files:
        text = _read_doc(path, gaps)
        if not text:
            continue
        rec = _parse(text, path, gaps)
        if rec:
            docs.append(rec)
    tools: list[ToolSpec] = []
    seen_tools: set[str] = set()
    steps: list = []
    task = ''
    instructions = ''
    idx = 0
    n_roles = 0
    n_actions = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        env = doc.get('environment')
        if not isinstance(env, dict):
            env = {}
        for spec in _tool_specs_from_registry(env):
            if spec.name and spec.name not in seen_tools:
                tools.append(spec)
                seen_tools.add(spec.name)
        task_field = doc.get('task') or doc.get('goal') or doc.get('objective')
        if task_field and (not task):
            task = content_to_text(task_field)
        instr_field = doc.get('instructions') or doc.get('system_prompt')
        if instr_field and (not instructions):
            instructions = content_to_text(instr_field)
        if isinstance(doc.get('tools'), list):
            for spec in normalize_tools(doc['tools']):
                if spec.name and spec.name not in seen_tools:
                    tools.append(spec)
                    seen_tools.add(spec.name)
        roles = doc.get('roles')
        top_level_actions = _coerce_actions(doc.get('actions'))
        actions_to_emit: list[dict] = []
        if isinstance(roles, list):
            for role in roles:
                if not isinstance(role, dict):
                    continue
                n_roles += 1
                role_actions = _coerce_actions(role.get('actions'))
                if role_actions:
                    actions_to_emit.extend(role_actions)
                else:
                    pass
            if not actions_to_emit and top_level_actions:
                actions_to_emit = top_level_actions
        else:
            actions_to_emit = top_level_actions
        for action in actions_to_emit:
            n_actions += 1
            ref = f"actions[{action.get('name', '?')}]"
            instr = content_to_text(action.get('instruction'))
            if instr.strip():
                steps.append(make_step(idx, 'message', role='user', name=str(action.get('name') or ''), content=instr, source_ref=ref, max_chars=max_step_chars))
                idx += 1
            output = action.get('output')
            body = content_to_text(output)
            if body.strip():
                steps.append(make_step(idx, 'message', role='assistant', name=str(action.get('name') or ''), content=body, is_error=detect_error(output), source_ref=ref, max_chars=max_step_chars, meta={'action': action.get('name', '')}))
                idx += 1
            if action.get('name') and action['name'] not in seen_tools:
                tools.append(ToolSpec(name=str(action['name'])))
                seen_tools.add(str(action['name']))
    if not docs:
        gaps.append('no MetaGPT records recoverable')
    if not tools and (not gaps):
        gaps.append('MetaGPT environment.tool_registry absent; tools inferred from action names only')
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_docs': len(docs), 'n_roles': n_roles, 'n_actions': n_actions})
