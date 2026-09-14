from __future__ import annotations

import ast
import json
from pathlib import Path

from ..schemas import CanonicalTrace, ToolSpec
from .base import content_to_text, make_step, maybe_json, normalize_tools, parse_arguments

# Recovered from the replayed prompt rather than a discrete record. The bundle
# exposes one tool list, for the turn it was captured on, so an earlier turn's
# call cannot be soundly schema-checked against it.
PROMPT_REPLAY = "prompt_replay"


def _parse_call_arguments(raw: object) -> dict | None:
    """Arguments as a dict, tolerating a Python repr.

    OpenClaw replays `arguments` as `str(dict)`, so single quotes and `True`
    make it invalid JSON. Left as an opaque blob it would read as a call whose
    arguments did not parse, which the schema gate reports as a violation.
    """
    parsed = maybe_json(raw)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, str) and parsed.strip()[:1] == "{":
        try:
            literal = ast.literal_eval(parsed)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            literal = None
        if isinstance(literal, dict):
            return {str(k): v for k, v in literal.items()}
    return parse_arguments(raw)


def load(trace_id: str, source: str, files: list[str], max_step_chars: int) -> CanonicalTrace:
    trace_path = next((f for f in files if f.endswith("trace.jsonl")))
    records = []
    gaps: list[str] = []
    with open(trace_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                gaps.append(f"{Path(trace_path).name}:{lineno}: unparsable JSONL record skipped")
    records.sort(key=lambda r: r.get("seq", 0))

    task = ""
    instructions = ""
    outcome = ""
    steps = []
    losses: list[dict] = []
    tools_by_name: dict[str, ToolSpec] = {}
    seen_assistant_ids: set[str] = set()
    seen_result_ids: set[str] = set()
    idx = 0

    # Pre-scan declared tools / prompts so replay does not invent undeclared
    # names that the later llm_input would have marked declared.
    prompt_snapshots: list[tuple[object, object, str]] = []
    for rec in records:
        kind = rec.get("kind", "")
        data = rec.get("data") or {}
        if kind == "run.start":
            inp = data.get("input") or {}
            task = content_to_text(inp.get("prompt", inp)) or task
        elif kind == "state" and data.get("type") == "openclaw.llm_input":
            value = data.get("value") or {}
            sp = value.get("system_prompt")
            if sp and not instructions:
                instructions = content_to_text(sp)
            for t in normalize_tools(value.get("tools")):
                t.declared = True
                existing = tools_by_name.get(t.name)
                if existing is None or not existing.declared:
                    tools_by_name[t.name] = t
            if not task and value.get("prompt"):
                task = content_to_text(value["prompt"])
        elif kind == "state" and data.get("type") == "openclaw.prompt_messages":
            prompt_snapshots.append((rec, data.get("value"), f"seq={rec.get('seq')}"))

    if prompt_snapshots:
        _rec, value, ref = max(
            prompt_snapshots,
            key=lambda item: len(item[1]) if isinstance(item[1], list) else 0,
        )
        idx = _replay_prompt_messages(
            value, steps, idx, tools_by_name, seen_assistant_ids, seen_result_ids, ref, max_step_chars
        )

    for rec in records:
        kind = rec.get("kind", "")
        data = rec.get("data") or {}
        seq = rec.get("seq")
        ref = f"seq={seq}"
        time = str(rec.get("time") or "")
        if kind == "state" and data.get("type") == "openclaw.assistant_message":
            value = data.get("value") or {}
            for block in value.get("content") or []:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text" and block.get("text", "").strip():
                    steps.append(
                        make_step(
                            idx,
                            "message",
                            role="assistant",
                            content=content_to_text(block["text"]),
                            source_ref=ref,
                            time=time,
                            max_chars=max_step_chars,
                        )
                    )
                    idx += 1
                elif btype == "toolCall":
                    cid = str(block.get("id") or "")
                    if cid and cid in seen_assistant_ids:
                        continue
                    seen_assistant_ids.add(cid)
                    call_name = str(block.get("name") or "")
                    if call_name and call_name not in tools_by_name:
                        tools_by_name[call_name] = ToolSpec(name=call_name, declared=False)
                    args = block.get("arguments")
                    steps.append(
                        make_step(
                            idx,
                            "tool_call",
                            role="assistant",
                            name=call_name,
                            arguments=parse_arguments(args),
                            call_id=cid,
                            source_ref=ref,
                            time=time,
                            max_chars=max_step_chars,
                        )
                    )
                    idx += 1
        elif kind == "tool.call":
            cid = str(data.get("native_call_id") or data.get("call_id") or "")
            if cid and cid in seen_assistant_ids:
                continue
            seen_assistant_ids.add(cid)
            name = str(data.get("name") or "")
            if name and name not in tools_by_name:
                tools_by_name[name] = ToolSpec(name=name, declared=False)
            args = data.get("input")
            steps.append(
                make_step(
                    idx,
                    "tool_call",
                    role="assistant",
                    name=name,
                    arguments=parse_arguments(args),
                    call_id=cid,
                    source_ref=ref,
                    time=time,
                    max_chars=max_step_chars,
                )
            )
            idx += 1
        elif kind == "tool.result":
            cid = str(data.get("native_call_id") or data.get("call_id") or "")
            if cid and cid in seen_result_ids:
                continue
            if cid:
                seen_result_ids.add(cid)
            status = str(data.get("status") or "")
            out = data.get("output")
            text = content_to_text(out)
            is_err = status not in ("", "succeeded", "success", "ok", "completed")
            steps.append(
                make_step(
                    idx,
                    "tool_result",
                    role="tool",
                    name=_lookup_call_name(steps, data),
                    content=text,
                    is_error=is_err,
                    call_id=cid,
                    source_ref=ref,
                    time=time,
                    max_chars=max_step_chars,
                    meta={"status": status} if status else None,
                )
            )
            idx += 1
        elif kind == "run.outcome":
            outcome = content_to_text(data.get("status") or data)
        elif kind == "loss":
            losses.append(
                {
                    "seq": seq,
                    "reason": data.get("reason", ""),
                    "detail": content_to_text(data.get("detail", "")),
                    "recoverable": data.get("recoverable"),
                }
            )
    return CanonicalTrace(
        trace_id=trace_id,
        source_format="semantic_trace_bundle",
        task=task,
        instructions=instructions,
        tools=[t for t in tools_by_name.values() if t.name],
        steps=steps,
        outcome=outcome,
        capture_gaps=gaps,
        meta={"loss_events": losses, "n_records": len(records)},
    )


def _replay_prompt_messages(
    messages,
    steps: list,
    idx: int,
    tools_by_name: dict[str, ToolSpec],
    seen_assistant_ids: set[str],
    seen_result_ids: set[str],
    ref: str,
    max_step_chars: int,
) -> int:
    """Recover earlier turns that survive only inside the replayed prompt.

    A multi-turn session records `tool.call` / `tool.result` for the turn being
    captured. Everything before it exists only here, so without this the audit
    sees one assistant message out of a fifteen-turn conversation -- and never
    sees the user's follow-up corrections at all, which is the whole evidence
    base for ignored feedback. Calls already carried by a discrete record are
    skipped by call id, so nothing is double counted.
    """
    if not isinstance(messages, list):
        return idx
    replayed_calls: set[str] = set()
    spoken: set[tuple[str, str]] = {
        (s.role, (s.content or "")[:160]) for s in steps if s.kind == "message"
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        blocks = message.get("content")
        if not isinstance(blocks, list):
            blocks = [blocks] if blocks else []
        if role == "toolResult":
            cid = str(message.get("toolCallId") or "")
            if cid not in replayed_calls or cid in seen_result_ids:
                continue
            seen_result_ids.add(cid)
            text = content_to_text(blocks)
            steps.append(
                make_step(
                    idx,
                    "tool_result",
                    role="tool",
                    name=str(message.get("toolName") or _lookup_call_name(steps, {"call_id": cid})),
                    content=text,
                    is_error=bool(message.get("isError")),
                    call_id=cid,
                    source_ref=ref,
                    time=str(message.get("timestamp") or ""),
                    max_chars=max_step_chars,
                    meta={"provenance": PROMPT_REPLAY},
                )
            )
            idx += 1
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "toolCall":
                cid = str(block.get("id") or "")
                if cid and cid in seen_assistant_ids:
                    continue
                if cid:
                    seen_assistant_ids.add(cid)
                    replayed_calls.add(cid)
                name = str(block.get("name") or "")
                if name and name not in tools_by_name:
                    tools_by_name[name] = ToolSpec(name=name, declared=False)
                steps.append(
                    make_step(
                        idx,
                        "tool_call",
                        role="assistant",
                        name=name,
                        arguments=_parse_call_arguments(block.get("arguments")),
                        call_id=cid,
                        source_ref=ref,
                        time=str(message.get("timestamp") or ""),
                        max_chars=max_step_chars,
                        meta={"provenance": PROMPT_REPLAY},
                    )
                )
                idx += 1
            elif btype in ("text", None) and str(block.get("text") or "").strip():
                body = content_to_text(block["text"])
                key = (role or "user", body[:160])
                if key in spoken:
                    continue
                spoken.add(key)
                steps.append(
                    make_step(
                        idx,
                        "message",
                        role=role or "user",
                        content=body,
                        source_ref=ref,
                        time=str(message.get("timestamp") or ""),
                        max_chars=max_step_chars,
                        meta={"provenance": PROMPT_REPLAY},
                    )
                )
                idx += 1
    return idx


def _lookup_call_name(steps, result_data: dict) -> str:
    cid = str(result_data.get("native_call_id") or result_data.get("call_id") or "")
    if cid:
        for s in reversed(steps):
            if s.kind == "tool_call" and s.call_id == cid:
                return s.name
    return str(result_data.get("name") or "")
