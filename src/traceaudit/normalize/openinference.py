"""OpenInference spans (Arize Phoenix / OpenTelemetry GenAI for agents).

Distinct from `otel_genai` despite both being OTLP: OpenInference names its
attributes `openinference.span.kind`, `llm.input_messages.*`, `tool.name`,
`input.value`, `output.value` — none of which start with `gen_ai.`, so the
`otel-genai` adapter drops every span and yields an empty trace.

This is the format the TRAIL benchmark ships in, and its annotations are keyed
by span id, so every canonical step records the span that produced it in
``meta['span_id']``. Without that, span-level gold labels cannot be mapped onto
step indices.

Reference: https://arize-ai.github.io/openinference/spec/
"""
from __future__ import annotations

import json
from typing import Any

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, detect_error, make_step, normalize_tools, parse_arguments

_SOURCE_FORMAT = 'openinference'

SPAN_KIND = 'openinference.span.kind'

_ROLE_MAP = {'user': 'user', 'human': 'user', 'assistant': 'assistant', 'ai': 'assistant', 'system': 'system', 'tool': 'tool', 'function': 'tool'}

# Span kinds that carry a tool invocation rather than a conversation turn.
_TOOL_KINDS = {'TOOL', 'RETRIEVER', 'RERANKER', 'EMBEDDING', 'GUARDRAIL'}


def _read(path: str, gaps: list[str]) -> Any:
    try:
        text = open(path, encoding='utf-8').read()
    except OSError as exc:
        gaps.append(f'{path}: unreadable ({exc})')
        return None
    if not text.strip():
        return None
    if path.endswith('.jsonl'):
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                gaps.append(f'{path}:{lineno}: unparsable JSONL record skipped')
        return records or None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        gaps.append(f'{path}: unparsable JSON')
        return None


def _unwrap(value: Any) -> Any:
    """Strip the OTLP AnyValue wrapper, if present."""
    if not isinstance(value, dict):
        return value
    for key in ('stringValue', 'string_value', 'intValue', 'int_value', 'doubleValue', 'double_value', 'boolValue', 'bool_value'):
        if key in value:
            return value[key]
    for key in ('arrayValue', 'array_value'):
        if key in value and isinstance(value[key], dict):
            return [_unwrap(v) for v in value[key].get('values') or []]
    for key in ('kvlistValue', 'kvlist_value'):
        if key in value and isinstance(value[key], dict):
            return {str(e.get('key')): _unwrap(e.get('value')) for e in value[key].get('values') or [] if isinstance(e, dict)}
    return value


def _attributes(span: dict) -> dict[str, Any]:
    """Attributes as a flat dict, from either OTLP lists or an exported mapping."""
    raw = span.get('attributes')
    if isinstance(raw, dict):
        return {str(k): _unwrap(v) for k, v in raw.items()}
    out: dict[str, Any] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            key = entry.get('key') or entry.get('name')
            if isinstance(key, str):
                out[key] = _unwrap(entry.get('value'))
    return out


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ('[', '{'):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, ValueError):
                return value
    return value


def _grouped(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    """Collapse ``prefix.<i>.rest`` keys into a list of ``{rest: value}`` dicts.

    OpenInference flattens message lists into indexed attribute keys, e.g.
    ``llm.input_messages.0.message.role``. An exporter may also leave the list
    whole, so that shape is accepted too.
    """
    whole = _maybe_json(attrs.get(prefix))
    if isinstance(whole, list):
        return [w for w in whole if isinstance(w, dict)]
    buckets: dict[int, dict[str, Any]] = {}
    head = f'{prefix}.'
    for key, value in attrs.items():
        if not key.startswith(head):
            continue
        index, _, rest = key[len(head):].partition('.')
        if not index.isdigit() or not rest:
            continue
        buckets.setdefault(int(index), {})[rest] = value
    return [buckets[i] for i in sorted(buckets)]


def _strip(group: dict[str, Any], prefix: str) -> dict[str, Any]:
    """``{'message.role': 'user'}`` -> ``{'role': 'user'}``."""
    head = f'{prefix}.'
    out = {}
    for key, value in group.items():
        out[key[len(head):] if key.startswith(head) else key] = value
    return out


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Tool calls on a message, flattened or whole.

    Either shape keeps the ``tool_call.`` prefix on its leaf keys, so strip it
    in both cases rather than only for the flattened one.
    """
    whole = _maybe_json(message.get('tool_calls'))
    if isinstance(whole, list):
        return [_strip(c, 'tool_call') for c in whole if isinstance(c, dict)]
    return [_strip(g, 'tool_call') for g in _grouped(message, 'tool_calls')]


def _call_parts(call: dict[str, Any]) -> tuple[str, Any, str]:
    """(name, arguments, id) from either nested or flattened tool-call shapes."""
    function = call.get('function') if isinstance(call.get('function'), dict) else {}
    name = function.get('name') or call.get('function.name') or call.get('name') or ''
    args = function.get('arguments') if 'arguments' in function else call.get('function.arguments', call.get('arguments'))
    return (str(name), args, str(call.get('id') or call.get('tool_call.id') or ''))


def _collect_spans(docs: list[Any], gaps: list[str]) -> list[dict]:
    """Pull spans out of OTLP envelopes, bare span lists, or TRAIL-style rows."""
    spans: list[dict] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        for key in ('resourceSpans', 'resource_spans', 'scopeSpans', 'scope_spans', 'batches'):
            if isinstance(node.get(key), list):
                visit(node[key])
        for key in ('spans', 'trace', 'steps'):
            value = _maybe_json(node.get(key))
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        if 'spans' in item or 'scopeSpans' in item or 'scope_spans' in item:
                            visit(item)
                        else:
                            spans.append(item)
                    else:
                        visit(item)
        # A bare span: it has attributes and an id of its own.
        if ('attributes' in node) and any((k in node for k in ('spanId', 'span_id', 'context', 'name'))):
            spans.append(node)

    visit(docs)
    deduped: list[dict] = []
    seen: set[int] = set()
    for span in spans:
        if id(span) not in seen:
            seen.add(id(span))
            deduped.append(span)
    if not deduped:
        gaps.append('no OpenInference spans recoverable from the input')
    return deduped


def _span_id(span: dict) -> str:
    context = span.get('context')
    if isinstance(context, dict):
        for key in ('span_id', 'spanId'):
            if context.get(key):
                return str(context[key])
    for key in ('spanId', 'span_id', 'id'):
        if span.get(key):
            return str(span[key])
    return ''


def _start(span: dict) -> str:
    for key in ('startTimeUnixNano', 'start_time_unix_nano', 'start_time', 'startTime'):
        if span.get(key) not in (None, ''):
            return str(span[key]).rjust(24, '0')
    return ''


def _span_failed(span: dict) -> bool:
    status = span.get('status')
    if isinstance(status, dict):
        code = status.get('code') or status.get('status_code')
        if code in (2, '2', 'STATUS_CODE_ERROR', 'ERROR', 'error'):
            return True
    elif isinstance(status, str) and status.upper() in ('ERROR', 'STATUS_CODE_ERROR'):
        return True
    for event in span.get('events') or []:
        if isinstance(event, dict) and str(event.get('name') or '').lower() == 'exception':
            return True
    return False


def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    gaps: list[str] = []
    docs = [d for d in (_read(p, gaps) for p in files) if d]
    spans = _collect_spans(docs, gaps)
    spans.sort(key=lambda s: (_start(s), _span_id(s)))

    task = ''
    instructions = ''
    # The declared set (llm.tools) is kept apart from names merely seen being
    # called. Folding the two together would tell the deterministic layer that
    # every tool the agent invented was declared, so it could never flag one.
    declared: dict[str, ToolSpec] = {}
    observed: dict[str, ToolSpec] = {}
    steps: list = []
    idx = 0
    # Every LLM span replays the whole conversation; emit each turn once.
    seen_messages: set[tuple[str, str]] = set()
    repeated_turns = 0
    pending: dict[str, dict] = {}
    kinds: dict[str, int] = {}
    unknown_kinds: set[str] = set()

    def declare(spec: ToolSpec) -> None:
        if spec.name:
            declared.setdefault(spec.name, spec)

    def observe(name: str, spec: ToolSpec | None=None) -> None:
        if name:
            observed.setdefault(name, spec or ToolSpec(name=name))

    for span in spans:
        attrs = _attributes(span)
        sid = _span_id(span)
        kind = str(attrs.get(SPAN_KIND) or attrs.get('openinference.span_kind') or '').upper()
        if not kind:
            kind = 'LLM' if any((k.startswith('llm.') for k in attrs)) else 'TOOL' if 'tool.name' in attrs else 'CHAIN'
            unknown_kinds.add(str(span.get('name') or '?'))
        kinds[kind] = kinds.get(kind, 0) + 1
        ref = f'spans[{sid}]'
        meta = {'span_id': sid, 'span_kind': kind, 'span_name': str(span.get('name') or '')}
        model = str(attrs.get('llm.model_name') or attrs.get('llm.model') or '')
        if model:
            meta['model'] = model
        failed = _span_failed(span)
        emitted = 0

        for group in _grouped(attrs, 'llm.tools'):
            schema = _maybe_json(_strip(group, 'tool').get('json_schema') or group.get('tool.json_schema'))
            for spec in normalize_tools([schema] if isinstance(schema, dict) else schema):
                declare(spec)

        if kind == 'LLM':
            for group in _grouped(attrs, 'llm.input_messages'):
                message = _strip(group, 'message')
                role = _ROLE_MAP.get(str(message.get('role') or '').lower(), 'user')
                content = content_to_text(_maybe_json(message.get('content') or message.get('contents')))
                if not content.strip():
                    continue
                key = (role, content[:400])
                if key in seen_messages:
                    repeated_turns += 1
                    continue
                seen_messages.add(key)
                if role == 'system' and not instructions:
                    instructions = content
                    continue
                if role == 'user' and not task:
                    task = content
                steps.append(make_step(idx, 'message', role=role, content=content, source_ref=f'{ref}.llm.input_messages', max_chars=max_step_chars, meta=dict(meta)))
                idx += 1
                emitted += 1

            for group in _grouped(attrs, 'llm.output_messages'):
                message = _strip(group, 'message')
                content = content_to_text(_maybe_json(message.get('content')))
                if content.strip():
                    # Register it: the next LLM span replays this turn as input.
                    seen_messages.add(('assistant', content[:400]))
                    steps.append(make_step(idx, 'message', role='assistant', content=content, source_ref=f'{ref}.llm.output_messages', max_chars=max_step_chars, meta=dict(meta)))
                    idx += 1
                    emitted += 1
                for call in _tool_calls(message):
                    name, raw_args, call_id = _call_parts(call)
                    if not name:
                        continue
                    observe(name)
                    call_id = call_id or f'oi-{sid}-{name}-{idx}'
                    steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(raw_args), call_id=call_id, source_ref=f'{ref}.llm.output_messages.tool_calls', max_chars=max_step_chars, meta=dict(meta)))
                    idx += 1
                    emitted += 1
                    pending[call_id] = {'name': name, 'span': sid}

        elif kind in _TOOL_KINDS:
            name = str(attrs.get('tool.name') or span.get('name') or kind.lower())
            description = str(attrs.get('tool.description') or '')
            parameters = _maybe_json(attrs.get('tool.parameters'))
            observe(name, ToolSpec(name=name, description=description, parameters=parameters if isinstance(parameters, dict) else {}))
            call_id = str(attrs.get('tool.call.id') or attrs.get('tool_call.id') or '')
            matched = pending.pop(call_id, None) if call_id else None
            if matched is None:
                # Fall back to the most recent unanswered call of the same name.
                for key in reversed(list(pending)):
                    if pending[key]['name'] == name:
                        call_id = key
                        matched = pending.pop(key)
                        break
            if matched is None:
                call_id = call_id or f'oi-{sid}'
                raw_input = attrs.get('input.value')
                if raw_input in (None, ''):
                    raw_input = attrs.get('tool.parameters')
                steps.append(make_step(idx, 'tool_call', role='assistant', name=name, arguments=parse_arguments(_maybe_json(raw_input)), call_id=call_id, source_ref=f'{ref}.input.value', max_chars=max_step_chars, meta=dict(meta)))
                idx += 1
                emitted += 1
            output = _maybe_json(attrs.get('output.value'))
            if output in (None, '') and kind == 'RETRIEVER':
                output = [_strip(g, 'document') for g in _grouped(attrs, 'retrieval.documents')]
            body = content_to_text(output)
            # Also replayed into the next LLM span as a role='tool' message.
            if body.strip():
                seen_messages.add(('tool', body[:400]))
            steps.append(make_step(idx, 'tool_result', role='tool', name=name, content=body, call_id=call_id, is_error=failed or detect_error(output), source_ref=f'{ref}.output.value', max_chars=max_step_chars, meta=dict(meta)))
            idx += 1
            emitted += 1

        if not emitted:
            # Keep the span addressable: span-level gold labels must land
            # somewhere even for AGENT/CHAIN spans with no messages of their own.
            body = content_to_text(_maybe_json(attrs.get('output.value'))) or content_to_text(_maybe_json(attrs.get('input.value')))
            steps.append(make_step(idx, 'other', role=kind.lower(), name=str(span.get('name') or ''), content=body, is_error=failed, source_ref=ref, max_chars=max_step_chars, meta=dict(meta)))
            idx += 1

    if declared:
        tools = list(declared.values())
    else:
        # No llm.tools anywhere: the declared set was not captured. Fall back to
        # observed names and say so, so "tool not declared" is read as unknown
        # rather than asserted against a list that was never recorded.
        tools = list(observed.values())
        if observed:
            gaps.append('no llm.tools schemas captured: the declared tool set is unknown, so undeclared-tool checks cannot apply')
    if spans and not any((s.kind in ('tool_call', 'tool_result') for s in steps)):
        gaps.append('no tool activity found in any OpenInference span')
    if unknown_kinds:
        gaps.append(f"{len(unknown_kinds)} span(s) carried no {SPAN_KIND}; kind inferred from attributes")
    for call_id, info in pending.items():
        gaps.append(f"tool call '{info['name']}' ({call_id}) from span {info['span']} had no matching TOOL span result")
    return CanonicalTrace(trace_id=trace_id, source_format=_SOURCE_FORMAT, task=task, instructions=instructions, tools=tools, steps=steps, capture_gaps=gaps, meta={'n_spans': len(spans), 'span_kinds': kinds, 'repeated_conversation_turns': repeated_turns, 'declared_tools': sorted(declared), 'observed_tools': sorted(observed), 'undeclared_tools_called': sorted(set(observed) - set(declared)) if declared else []})
