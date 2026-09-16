"""Load common harness JSON into one canonical trace.

Accepts already-canonical traces, OpenAI chat / Responses, Anthropic content
blocks, generic message lists, and event logs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import CanonicalStep, CanonicalTrace, ToolSpec

_ROLE_MAP = {
    "user": "user",
    "human": "user",
    "customer": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "agent": "assistant",
    "ai": "assistant",
    "system": "system",
    "developer": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}
_MSG_KEYS = (
    "messages",
    "conversations",
    "turns",
    "dialogue",
    "history",
    "chat_messages",
    "input",
    "output",
    "steps",
    "trace",
)
_PROMPT_SUFFIXES = {".txt", ".md", ".prompt", ".text"}
_TRACE_SUFFIXES = {".json", ".jsonl"}
_SKIP_NAMES = {"index.json", "complete.json"}
_ACTION_LINE = re.compile(r"^Action:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_ACTION_CALL = re.compile(r"^([A-Za-z_][\w\.]*)\((.*)\)\s*$")
_ACT_KIND = re.compile(r"Act:\s*(bash|finish|answer)(?:\((.*)\))?", re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:bash|sh|shell)?\s*\n(.*?)```", re.DOTALL)


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.7)
    tail = max(0, max_chars - head - 32)
    omitted = len(text) - head - tail
    return text[:head] + f"\n[...TRUNCATED {omitted} chars...]\n" + text[-tail:], True


def _block_to_text(block: Any, depth: int = 0) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, (int, float, bool)) or block is None:
        return "" if block is None else str(block)
    if isinstance(block, bytes):
        return block.decode("utf-8", errors="replace")
    if isinstance(block, list):
        return "\n".join(p for p in (_block_to_text(b, depth + 1) for b in block) if p)
    if isinstance(block, dict) and depth < 6:
        btype = str(block.get("type", "")).lower()
        if btype == "text" and isinstance(block.get("text"), str):
            return block["text"]
        if btype in {"image", "image_url", "audio", "file", "video"}:
            return f"[{btype} omitted]"
        if isinstance(block.get("text"), str):
            return block["text"]
        if "content" in block:
            inner = _block_to_text(block["content"], depth + 1)
            if inner:
                return inner
        try:
            return json.dumps(block, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(block)
    try:
        return json.dumps(block, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(block)


def content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (int, float, bool)):
        return str(value)
    return _block_to_text(value)


def maybe_json(text: Any) -> Any:
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return text
    t = text.strip()
    if t[:1] in {"[", "{"}:
        try:
            return json.loads(t)
        except (json.JSONDecodeError, ValueError):
            return text
    return text


def parse_arguments(raw: Any) -> dict | None:
    raw = maybe_json(raw)
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str) and raw.strip()[:1] == "{":
        try:
            import ast

            literal = ast.literal_eval(raw)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            literal = None
        if isinstance(literal, dict):
            return {str(k): v for k, v in literal.items()}
    return {"_raw": raw}


def normalize_tools(raw: Any) -> list[ToolSpec]:
    raw = maybe_json(raw)
    specs: list[ToolSpec] = []
    if not isinstance(raw, list):
        return specs
    for item in raw:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") or ""
        if not name:
            continue
        params = fn.get("parameters") or fn.get("input_schema") or {}
        if not isinstance(params, dict):
            params = {}
        specs.append(
            ToolSpec(
                name=str(name),
                description=str(fn.get("description") or ""),
                parameters=params,
            )
        )
    return specs


def detect_error(value: Any, depth: int = 0) -> bool:
    if depth > 5:
        return False
    obj = maybe_json(value) if isinstance(value, str) else value
    if isinstance(obj, dict):
        if obj.get("is_error") is True or obj.get("isError") is True:
            return True
        if obj.get("success") is False or obj.get("ok") is False:
            return True
        err = obj.get("error")
        if err is True or (isinstance(err, (str, dict, list)) and err):
            return True
        st = str(obj.get("status") or obj.get("state") or "").strip().lower()
        if st in {"failed", "error", "failure", "timeout", "denied", "rejected"}:
            return True
        code = obj.get("code") or obj.get("status_code") or obj.get("http_status")
        if isinstance(code, int) and code >= 400:
            return True
        return any(
            detect_error(v, depth + 1) if isinstance(v, (dict, list)) else False
            for v in obj.values()
        )
    if isinstance(obj, list):
        return any(detect_error(x, depth + 1) for x in obj)
    if isinstance(obj, str):
        low = obj[:400].lower()
        if any(n in low for n in ("no error", "without error", "error-free")):
            return False
        return any(
            h in low
            for h in ("error:", "exception", "traceback", "failed", "timed out")
        )
    return False


def make_step(
    index: int,
    kind: str,
    *,
    role: str = "",
    name: str = "",
    content: str = "",
    arguments: dict | None = None,
    call_id: str = "",
    is_error: bool = False,
    source_ref: str = "",
    max_chars: int = 6000,
) -> CanonicalStep:
    raw_len = len(content)
    body, truncated = truncate(content, max_chars)
    return CanonicalStep(
        index=index,
        kind=kind,  # type: ignore[arg-type]
        role=role,
        name=name,
        content=body,
        arguments=arguments,
        call_id=call_id,
        is_error=is_error,
        truncated=truncated,
        raw_len=raw_len,
        source_ref=source_ref,
    )


def looks_canonical(doc: Any) -> bool:
    if not isinstance(doc, dict):
        return False
    steps = doc.get("steps")
    if not isinstance(steps, list) or not steps:
        return False
    first = steps[0]
    return isinstance(first, dict) and "kind" in first


def looks_like_trace_doc(doc: Any) -> bool:
    if looks_canonical(doc):
        return True
    if isinstance(doc, dict) and any(
        k in doc
        for k in (
            "messages",
            "conversations",
            "spans",
            "output",
            "available_tools",
            "tools",
            "contents",
            "chat_messages",
        )
    ):
        return True
    if (
        isinstance(doc, list)
        and doc
        and isinstance(doc[0], dict)
        and any(
            k in doc[0]
            for k in ("role", "from", "content", "value", "tool_calls", "type", "kind")
        )
    ):
        return True
    return False


def read_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if not stripped:
        return None
    if path.suffix.lower() == ".jsonl" or stripped[0] not in "{[":
        recs = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
        return recs
    return json.loads(text)


def _task_from(doc: dict) -> str:
    for key in ("task", "user_task", "query", "question", "prompt"):
        val = doc.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _instructions_from(doc: dict) -> str:
    for key in ("instructions", "system", "system_prompt", "policy", "developer"):
        val = doc.get(key)
        if isinstance(val, (str, list)) and content_to_text(val).strip():
            return content_to_text(val)
    return ""


def load_canonical(trace_id: str, doc: dict, max_step_chars: int) -> CanonicalTrace:
    steps = []
    for i, raw in enumerate(doc.get("steps") or []):
        if not isinstance(raw, dict):
            continue
        payload = dict(raw)
        payload.setdefault("index", i)
        payload.setdefault("kind", "other")
        content = str(payload.get("content") or "")
        if len(content) > max_step_chars:
            payload["content"], payload["truncated"] = truncate(content, max_step_chars)
        steps.append(CanonicalStep.model_validate(payload))
    return CanonicalTrace(
        trace_id=str(doc.get("trace_id") or trace_id),
        source_format=str(doc.get("source_format") or "canonical"),
        task=str(doc.get("task") or ""),
        instructions=str(doc.get("instructions") or ""),
        tools=normalize_tools(doc.get("tools") or []),
        steps=steps,
        outcome=str(doc.get("outcome") or ""),
        capture_gaps=list(doc.get("capture_gaps") or []),
        meta=dict(doc.get("meta") or {}),
    )


class _Builder:
    def __init__(self, max_chars: int):
        self.max_chars = max_chars
        self.steps: list[CanonicalStep] = []
        self.last_call_name = ""
        self.pending_result = False

    def add(self, **kwargs) -> None:
        kwargs.setdefault("max_chars", self.max_chars)
        kwargs["index"] = len(self.steps)
        if kwargs.get("kind") == "tool_call" and kwargs.get("name"):
            self.last_call_name = kwargs["name"]
            self.pending_result = True
        elif kwargs.get("kind") == "tool_result":
            self.pending_result = False
        self.steps.append(make_step(**kwargs))


def _parse_action_line(text: str) -> tuple[str, dict] | None:
    act = _ACT_KIND.search(text or "")
    if act:
        kind = act.group(1).lower()
        if kind == "bash":
            fence = _CODE_FENCE.search(text or "")
            return "bash", {"command": fence.group(1).strip() if fence else ""}
        if kind == "finish":
            return "finish", {}
        return "answer", {"answer": (act.group(2) or "").strip()}
    match = _ACTION_LINE.search(text or "")
    if not match:
        return None
    call = match.group(1).strip()
    parsed = _ACTION_CALL.match(call)
    if parsed:
        raw = parsed.group(2).strip()
        args = (
            parse_arguments(raw) if raw[:1] in "{[" else ({"_raw": raw} if raw else {})
        )
        return parsed.group(1), args
    return "env_action", {"action": call}


def _ingest_blocks(
    builder: _Builder, role: str, content: Any, ref: str, name: str = ""
) -> None:
    speaker = name or ""
    if isinstance(content, str):
        if content.strip():
            kind = "tool_result" if role == "tool" else "message"
            builder.add(
                kind=kind,
                role=role or "assistant",
                name=builder.last_call_name if kind == "tool_result" else speaker,
                content=content,
                is_error=detect_error(content) if kind == "tool_result" else False,
                source_ref=ref,
            )
        return
    if not isinstance(content, list):
        text = content_to_text(content)
        if text.strip():
            builder.add(
                kind="message",
                role=role or "assistant",
                name=speaker,
                content=text,
                source_ref=ref,
            )
        return
    for block in content:
        if not isinstance(block, dict):
            text = content_to_text(block)
            if text.strip():
                builder.add(
                    kind="message",
                    role=role or "assistant",
                    name=speaker,
                    content=text,
                    source_ref=ref,
                )
            continue
        btype = str(block.get("type") or "").lower()
        if btype in {"tool_use", "function_call"}:
            tool_name = str(block.get("name") or "")
            raw_args = block.get("input", block.get("arguments", {}))
            builder.add(
                kind="tool_call",
                role="assistant",
                name=tool_name,
                arguments=parse_arguments(raw_args),
                call_id=str(block.get("id") or block.get("call_id") or ""),
                source_ref=ref,
            )
        elif btype in {"tool_result", "function_call_output"}:
            body = content_to_text(block.get("content", block.get("output", "")))
            builder.add(
                kind="tool_result",
                role="tool",
                name=builder.last_call_name,
                content=body,
                call_id=str(block.get("tool_use_id") or block.get("call_id") or ""),
                is_error=bool(block.get("is_error"))
                or detect_error(block.get("content", body)),
                source_ref=ref,
            )
        elif btype in {"text", "output_text"}:
            text = str(block.get("text") or block.get("content") or "")
            if text.strip():
                builder.add(
                    kind="message",
                    role=role or "assistant",
                    name=speaker,
                    content=text,
                    source_ref=ref,
                )
        elif btype in {"thinking"}:
            continue
        else:
            text = content_to_text(block)
            if text.strip():
                builder.add(
                    kind="message",
                    role=role or "unknown",
                    name=speaker,
                    content=text,
                    source_ref=ref,
                )


def _ingest_function_value(builder: _Builder, msg: dict, ref: str) -> None:
    parsed = parse_arguments(msg.get("value", msg.get("arguments", msg.get("content"))))
    name = ""
    args: dict = {}
    if isinstance(parsed, dict):
        name = str(parsed.get("name") or msg.get("name") or "")
        args = parse_arguments(parsed.get("arguments", parsed.get("input"))) or {}
        if not name and parsed.get("_raw"):
            args = parsed
    builder.add(
        kind="tool_call",
        role="assistant",
        name=name or str(msg.get("name") or ""),
        arguments=args,
        call_id=str(msg.get("id") or msg.get("call_id") or ""),
        source_ref=ref,
    )


def _ingest_message(
    builder: _Builder, msg: dict, ref: str, task: list[str], instructions: list[str]
) -> None:
    raw_role = str(msg.get("role") or msg.get("from") or msg.get("type") or "").lower()
    if raw_role in {"function_call", "function_call_output"}:
        if msg.get("name") or msg.get("type") in {"function_call", "tool_use"}:
            _ingest_output_item(builder, msg, ref)
        else:
            _ingest_function_value(builder, msg, ref)
        return
    if raw_role == "observation":
        text = content_to_text(msg.get("value", msg.get("content", msg.get("text"))))
        builder.add(
            kind="tool_result",
            role="tool",
            name=builder.last_call_name,
            content=text,
            is_error=detect_error(text),
            source_ref=ref,
        )
        return
    role = _ROLE_MAP.get(raw_role, raw_role)
    speaker = str(msg.get("name") or "")
    text = content_to_text(msg.get("content", msg.get("value", msg.get("text"))))
    uses_from_value = "from" in msg and "role" not in msg
    if role == "system" and text.strip() and not instructions[0]:
        if "tool_declare" not in text[:120]:
            instructions[0] = text
    if role == "user" and text.strip() and not task[0]:
        if (
            uses_from_value
            and not instructions[0]
            and (
                text.lower().startswith("you are")
                or "you can use the following tools" in text.lower()
            )
        ):
            instructions[0] = text
        else:
            task[0] = text
    if uses_from_value and role == "user" and builder.pending_result:
        builder.add(
            kind="tool_result",
            role="tool",
            name=builder.last_call_name,
            content=text,
            is_error=detect_error(text),
            source_ref=ref,
        )
        return

    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or tc
        builder.add(
            kind="tool_call",
            role="assistant",
            name=str(fn.get("name") or ""),
            arguments=parse_arguments(fn.get("arguments", tc.get("arguments"))),
            call_id=str(tc.get("id") or ""),
            source_ref=ref,
        )
    fc = msg.get("function_call")
    if isinstance(fc, dict):
        builder.add(
            kind="tool_call",
            role="assistant",
            name=str(fc.get("name") or ""),
            arguments=parse_arguments(fc.get("arguments")),
            call_id=str(fc.get("id") or msg.get("id") or ""),
            source_ref=ref,
        )

    content = msg.get("content", msg.get("value", msg.get("text")))
    if role == "tool":
        builder.add(
            kind="tool_result",
            role="tool",
            name=str(msg.get("name") or builder.last_call_name),
            content=text,
            call_id=str(msg.get("tool_call_id") or ""),
            is_error=detect_error(content),
            source_ref=ref,
        )
        return
    if isinstance(content, list):
        _ingest_blocks(builder, role or "assistant", content, ref, name=speaker)
        return
    if text.strip():
        builder.add(
            kind="message",
            role=role or raw_role or "unknown",
            name=speaker,
            content=text,
            source_ref=ref,
        )
    if uses_from_value and role == "assistant":
        action = _parse_action_line(text)
        if action:
            name, args = action
            builder.add(
                kind="tool_call",
                role="assistant",
                name=name,
                arguments=args,
                source_ref=ref,
            )


def _ingest_output_item(builder: _Builder, item: dict, ref: str) -> None:
    itype = str(item.get("type") or "").lower()
    if itype in {"function_call", "tool_use"}:
        builder.add(
            kind="tool_call",
            role="assistant",
            name=str(item.get("name") or ""),
            arguments=parse_arguments(item.get("arguments", item.get("input"))),
            call_id=str(item.get("call_id") or item.get("id") or ""),
            source_ref=ref,
        )
    elif itype in {"function_call_output", "tool_result"}:
        body = content_to_text(item.get("output", item.get("content", "")))
        builder.add(
            kind="tool_result",
            role="tool",
            name=builder.last_call_name,
            content=body,
            call_id=str(item.get("call_id") or item.get("tool_use_id") or ""),
            is_error=detect_error(item.get("output", body)),
            source_ref=ref,
        )
    elif itype in {"message", "output_text"}:
        _ingest_blocks(
            builder,
            str(item.get("role") or "assistant"),
            item.get("content", item.get("text")),
            ref,
            name=str(item.get("name") or ""),
        )


def _find_message_list(doc: Any) -> list:
    if isinstance(doc, list):
        if (
            doc
            and isinstance(doc[0], dict)
            and any(
                k in doc[0] for k in ("role", "from", "content", "type", "tool_calls")
            )
        ):
            return [v for v in doc if isinstance(v, dict)]
        for v in doc:
            if isinstance(v, dict):
                inner = _find_message_list(v)
                if inner:
                    return inner
        return [v for v in doc if isinstance(v, dict)]
    if isinstance(doc, dict):
        for key in _MSG_KEYS:
            val = maybe_json(doc.get(key))
            if isinstance(val, list) and val:
                return val
    return []


def _unwrap_strings(doc: Any) -> Any:
    if isinstance(doc, dict):
        out = dict(doc)
        for key in (
            "messages",
            "conversations",
            "available_tools",
            "tools",
            "functions",
            "tool_definitions",
        ):
            if key in out:
                out[key] = maybe_json(out[key])
        return out
    return doc


def _lookup_call_name(steps: list[CanonicalStep], result_data: dict) -> str:
    cid = str(result_data.get("native_call_id") or result_data.get("call_id") or "")
    if cid:
        for step in reversed(steps):
            if step.kind == "tool_call" and step.call_id == cid:
                return step.name
    return str(result_data.get("name") or "")


def _replay_prompt_messages(
    builder: _Builder,
    messages: list,
    tools_by_name: dict[str, ToolSpec],
    seen_calls: set[str],
    seen_results: set[str],
) -> None:
    """Recover turns that exist only inside the replayed prompt, not as events."""
    replayed_calls: set[str] = set()
    spoken: set[tuple[str, str]] = {
        (s.role, (s.content or "")[:160]) for s in builder.steps if s.kind == "message"
    }
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        speaker = str(message.get("name") or "")
        blocks = message.get("content")
        if not isinstance(blocks, list):
            blocks = [blocks] if blocks else []
        if role == "toolResult":
            cid = str(message.get("toolCallId") or "")
            if cid not in replayed_calls or cid in seen_results:
                continue
            seen_results.add(cid)
            builder.add(
                kind="tool_result",
                role="tool",
                name=str(
                    message.get("toolName")
                    or _lookup_call_name(builder.steps, {"call_id": cid})
                ),
                content=content_to_text(blocks),
                is_error=bool(message.get("isError")),
                call_id=cid,
                source_ref="prompt_replay",
            )
            continue
        for block in blocks:
            if not isinstance(block, dict):
                text = content_to_text(block)
                if text.strip():
                    key = (role or "user", text[:160])
                    if key not in spoken:
                        spoken.add(key)
                        builder.add(
                            kind="message",
                            role=role or "user",
                            name=speaker,
                            content=text,
                            source_ref="prompt_replay",
                        )
                continue
            btype = block.get("type")
            if btype == "toolCall":
                cid = str(block.get("id") or "")
                if cid and cid in seen_calls:
                    continue
                if cid:
                    seen_calls.add(cid)
                    replayed_calls.add(cid)
                name = str(block.get("name") or "")
                if name and name not in tools_by_name:
                    tools_by_name[name] = ToolSpec(name=name, declared=False)
                builder.add(
                    kind="tool_call",
                    role="assistant",
                    name=name,
                    arguments=parse_arguments(block.get("arguments")),
                    call_id=cid,
                    source_ref="prompt_replay",
                )
            elif btype in ("text", None) and str(block.get("text") or "").strip():
                body = content_to_text(block.get("text"))
                key = (role or "user", body[:160])
                if key in spoken:
                    continue
                spoken.add(key)
                builder.add(
                    kind="message",
                    role=role or "user",
                    name=speaker,
                    content=body,
                    source_ref="prompt_replay",
                )


def looks_event_log(doc: Any) -> bool:
    if not isinstance(doc, list) or not doc:
        return False
    first = next((x for x in doc if isinstance(x, dict)), None)
    return bool(first and "kind" in first and ("data" in first or "seq" in first))


def ingest_event_log(
    trace_id: str, records: list, max_step_chars: int
) -> CanonicalTrace:
    """OpenClaw / Tavii-style JSONL: kind + data, not a chat transcript."""
    builder = _Builder(max_step_chars)
    task = ""
    instructions = ""
    tools_by_name: dict[str, ToolSpec] = {}
    seen_calls: set[str] = set()
    seen_results: set[str] = set()
    gaps: list[str] = []
    outcome = ""
    prompt_msgs: list = []

    def note_tool(name: str) -> None:
        if name and name not in tools_by_name:
            tools_by_name[name] = ToolSpec(name=name, declared=False)

    for rec in records:
        if not isinstance(rec, dict):
            continue
        kind = str(rec.get("kind") or "")
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        if kind == "run.start":
            inp = data.get("input") or {}
            if isinstance(inp, dict):
                task = content_to_text(inp.get("prompt", inp)) or task
        elif kind == "state" and data.get("type") == "openclaw.llm_input":
            value = data.get("value") or {}
            if isinstance(value, dict):
                if value.get("system_prompt") and not instructions:
                    instructions = content_to_text(value.get("system_prompt"))
                for tool in normalize_tools(value.get("tools")):
                    tools_by_name[tool.name] = tool
                if not task and value.get("prompt"):
                    task = content_to_text(value.get("prompt"))
        elif kind == "state" and data.get("type") == "openclaw.prompt_messages":
            value = data.get("value")
            if isinstance(value, list) and len(value) > len(prompt_msgs):
                prompt_msgs = value
        elif kind == "loss":
            reason = data.get("reason") or "loss"
            gaps.append(f"{reason}: {content_to_text(data.get('detail') or '')[:200]}")
        elif kind == "run.outcome":
            outcome = content_to_text(data.get("status") or data)

    if prompt_msgs:
        _replay_prompt_messages(
            builder, prompt_msgs, tools_by_name, seen_calls, seen_results
        )

    for rec in records:
        if not isinstance(rec, dict):
            continue
        kind = str(rec.get("kind") or "")
        data = rec.get("data") or {}
        if not isinstance(data, dict):
            data = {}
        ref = f"seq={rec.get('seq')}"
        if kind == "state" and data.get("type") == "openclaw.assistant_message":
            value = data.get("value") or {}
            speaker = str(value.get("name") or value.get("agent") or "")
            for block in value.get("content") or []:
                if not isinstance(block, dict):
                    continue
                btype = str(block.get("type") or "")
                if btype == "text" and str(block.get("text") or "").strip():
                    builder.add(
                        kind="message",
                        role="assistant",
                        name=speaker,
                        content=content_to_text(block.get("text")),
                        source_ref=ref,
                    )
                elif btype in {"toolCall", "tool_use", "tool_call"}:
                    cid = str(block.get("id") or "")
                    if cid and cid in seen_calls:
                        continue
                    if cid:
                        seen_calls.add(cid)
                    name = str(block.get("name") or "")
                    note_tool(name)
                    builder.add(
                        kind="tool_call",
                        role="assistant",
                        name=name,
                        arguments=parse_arguments(block.get("arguments")),
                        call_id=cid,
                        source_ref=ref,
                    )
        elif kind == "tool.call":
            cid = str(data.get("native_call_id") or data.get("call_id") or "")
            if cid and cid in seen_calls:
                continue
            if cid:
                seen_calls.add(cid)
            name = str(data.get("name") or "")
            note_tool(name)
            builder.add(
                kind="tool_call",
                role="assistant",
                name=name,
                arguments=parse_arguments(data.get("input", data.get("arguments"))),
                call_id=cid,
                source_ref=ref,
            )
        elif kind == "tool.result":
            cid = str(data.get("native_call_id") or data.get("call_id") or "")
            if cid and cid in seen_results:
                continue
            if cid:
                seen_results.add(cid)
            status = str(data.get("status") or "")
            text = content_to_text(data.get("output"))
            name = str(
                data.get("name")
                or _lookup_call_name(builder.steps, data)
                or builder.last_call_name
            )
            builder.add(
                kind="tool_result",
                role="tool",
                name=name,
                content=text,
                call_id=cid,
                is_error=status not in {"", "succeeded", "success", "ok", "completed"},
                source_ref=ref,
            )

    return CanonicalTrace(
        trace_id=trace_id,
        source_format="event-log",
        task=task,
        instructions=instructions,
        tools=[t for t in tools_by_name.values() if t.name],
        steps=builder.steps,
        outcome=outcome,
        capture_gaps=gaps[:12],
    )


def _span_attr(span: dict, *keys: str) -> Any:
    attrs = span.get("attributes") or span.get("span_attributes") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    for key in keys:
        val = attrs.get(key)
        if val not in (None, ""):
            return val
        val = span.get(key)
        if val not in (None, ""):
            return val
    return None


def _span_nodes(doc: Any) -> list:
    if not isinstance(doc, dict):
        return []
    for key in ("spans", "otel_spans"):
        val = doc.get(key)
        if isinstance(val, list) and val:
            return val
    trace = doc.get("trace")
    if isinstance(trace, list) and trace:
        first = next((x for x in trace if isinstance(x, dict)), None)
        if first and (
            "span_id" in first
            or "attributes" in first
            or "child_spans" in first
            or "context" in first
        ):
            return trace
    if isinstance(trace, dict):
        inner = trace.get("spans") or trace.get("child_spans")
        if isinstance(inner, list) and inner:
            return inner
    return []


def looks_span_tree(doc: Any) -> bool:
    """OpenInference / OTEL / TRAIL nested spans (converter-shaped, not a chat list)."""
    first = next((x for x in _span_nodes(doc) if isinstance(x, dict)), None)
    if not first:
        return False
    attrs = first.get("attributes") if isinstance(first.get("attributes"), dict) else {}
    ctx = first.get("context") if isinstance(first.get("context"), dict) else {}
    if first.get("span_id") or ctx.get("span_id"):
        return True
    if first.get("child_spans") or first.get("children"):
        return True
    return bool(attrs.get("openinference.span.kind") or attrs.get("span.kind"))


def ingest_span_tree(trace_id: str, doc: dict, max_step_chars: int) -> CanonicalTrace:
    """Thin walker for span trees. Same shapes the checked-corpus converter flattens."""
    builder = _Builder(max_step_chars)

    def walk(nodes: Any, depth: int = 0) -> None:
        if depth > 24:
            return
        if isinstance(nodes, dict):
            nodes = [nodes]
        if not isinstance(nodes, list):
            return
        for span in nodes:
            if not isinstance(span, dict):
                continue
            kind = str(
                _span_attr(span, "openinference.span.kind", "span.kind") or ""
            ).upper()
            name = str(
                _span_attr(span, "tool.name", "llm.model_name")
                or span.get("name")
                or ""
            )
            inp = _span_attr(span, "input.value", "input")
            out = _span_attr(span, "output.value", "output")
            inp_text = content_to_text(inp) if inp not in (None, "") else ""
            out_text = content_to_text(out) if out not in (None, "") else ""
            ctx = span.get("context") if isinstance(span.get("context"), dict) else {}
            sid = str(span.get("span_id") or ctx.get("span_id") or "")
            ref = f"span:{sid}" if sid else "span"
            is_tool = (
                kind == "TOOL"
                or kind.endswith("TOOL")
                or bool(_span_attr(span, "tool.name"))
            )
            if is_tool and (name or inp_text or out_text):
                args = parse_arguments(inp) if inp not in (None, "") else {}
                if not isinstance(args, dict):
                    args = {"_raw": inp_text}
                builder.add(
                    kind="tool_call",
                    role="assistant",
                    name=name,
                    arguments=args,
                    call_id=sid,
                    source_ref=ref,
                )
                if out_text:
                    builder.add(
                        kind="tool_result",
                        role="tool",
                        name=name,
                        content=out_text,
                        call_id=sid,
                        is_error=detect_error(out),
                        source_ref=ref,
                    )
            elif inp_text or out_text or kind in {"LLM", "AGENT", "CHAIN"}:
                body = out_text or inp_text
                if body.strip():
                    role = "assistant" if kind in {"LLM", "AGENT", ""} else "unknown"
                    builder.add(
                        kind="message",
                        role=role,
                        name=name,
                        content=body,
                        source_ref=ref,
                    )
            walk(span.get("child_spans") or span.get("children") or [], depth + 1)

    walk(_span_nodes(doc))
    return CanonicalTrace(
        trace_id=trace_id,
        source_format="otel-spans",
        task=_task_from(doc),
        instructions=_instructions_from(doc),
        tools=normalize_tools(doc.get("tools") or []),
        steps=builder.steps,
        meta={"ingest": "otel-spans"},
    )


def ingest_doc(trace_id: str, doc: Any, max_step_chars: int) -> CanonicalTrace:
    doc = _unwrap_strings(doc)
    if looks_canonical(doc):
        return load_canonical(trace_id, doc, max_step_chars)
    if looks_event_log(doc):
        return ingest_event_log(trace_id, doc, max_step_chars)
    if looks_span_tree(doc):
        return ingest_span_tree(trace_id, doc, max_step_chars)
    if isinstance(doc, list):
        wrapper: dict[str, Any] = {"messages": doc}
        recs = [wrapper]
    else:
        recs = [_unwrap_strings(doc)]
    builder = _Builder(max_step_chars)
    task = [""]
    instructions = [""]
    tools: list[ToolSpec] = []
    source = "generic"
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        if looks_canonical(rec) and not builder.steps:
            return load_canonical(trace_id, rec, max_step_chars)
        if not task[0]:
            task[0] = _task_from(rec)
        if not instructions[0]:
            instructions[0] = _instructions_from(rec)
        if not tools:
            for key in ("tools", "available_tools", "functions", "tool_definitions"):
                found = normalize_tools(rec.get(key))
                if found:
                    tools = found
                    break
        if isinstance(rec.get("messages"), list):
            source = "openai-chat"
        if isinstance(rec.get("conversations"), list):
            source = "conversations"
        msgs = _find_message_list(rec)
        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                continue
            itype = str(msg.get("type") or "").lower()
            if (
                itype
                in {"function_call", "function_call_output", "tool_use", "tool_result"}
                and "role" not in msg
            ):
                _ingest_output_item(builder, msg, f"item[{i}]")
            else:
                _ingest_message(builder, msg, f"messages[{i}]", task, instructions)
    if not task[0] and instructions[0]:
        task[0] = instructions[0]
    return CanonicalTrace(
        trace_id=trace_id,
        source_format=source,
        task=task[0],
        instructions=instructions[0],
        tools=tools,
        steps=builder.steps,
    )


def _detect_unpaired(trace: CanonicalTrace) -> None:
    calls = {
        s.call_id: s.index for s in trace.steps if s.kind == "tool_call" and s.call_id
    }
    results = {s.call_id for s in trace.steps if s.kind == "tool_result" and s.call_id}
    unpaired = [idx for cid, idx in calls.items() if cid not in results]
    if unpaired:
        trace.capture_gaps.append(
            f"unpaired tool_call at step(s) {unpaired}: no matching tool_result"
        )


def _primary_files(files: list[str]) -> list[str]:
    jsonl = [f for f in files if str(f).endswith(".jsonl")]
    if jsonl:
        return jsonl + [f for f in files if f not in jsonl]
    return list(files)


def load_trace(
    trace_id: str,
    source: str,
    files: list[str],
    max_step_chars: int = 6000,
) -> CanonicalTrace:
    gaps: list[str] = []
    doc: Any = None
    for path in _primary_files(files):
        try:
            doc = read_json(Path(path))
        except (OSError, json.JSONDecodeError) as exc:
            gaps.append(f"{path}: {exc}")
            continue
        if doc is not None:
            break
    if doc is None:
        raise ValueError(
            f"could not read a trace from {files}: {'; '.join(gaps) or 'empty'}"
        )
    trace = ingest_doc(trace_id, doc, max_step_chars)
    if source:
        trace.meta["requested_source"] = source
    trace.capture_gaps.extend(gaps)
    _detect_unpaired(trace)
    if not trace.steps and not trace.capture_gaps:
        trace.capture_gaps.append("no steps recovered from this file")
    for i, step in enumerate(trace.steps):
        step.index = i
    return trace


def looks_like_trace(path: Path | str) -> bool:
    path = Path(path)
    if not path.is_file() or path.suffix.lower() not in _TRACE_SUFFIXES:
        return False
    if path.name.lower() in _SKIP_NAMES:
        return False
    try:
        doc = read_json(path)
    except (OSError, ValueError):
        return False
    return looks_like_trace_doc(doc)


def _tavii_bundle(folder: Path) -> list[Path] | None:
    trace = folder / "trace.jsonl"
    if not trace.is_file():
        return None
    files = [trace]
    for name in ("manifest.json", "complete.json"):
        extra = folder / name
        if extra.is_file():
            files.append(extra)
    return files


def load_dataset(path: str | Path, source_override: str = "") -> dict[str, dict]:
    path = Path(path)
    items: dict[str, dict] = {}

    def add(tid: str, files: list[Path], source: str) -> None:
        base, n = tid, 2
        while tid in items:
            tid = f"{base}-{n}"
            n += 1
        items[tid] = {
            "source": source_override or source,
            "files": [str(f) for f in files],
        }

    if path.is_file():
        if not looks_like_trace(path):
            raise FileNotFoundError(f"{path} does not look like an agent trace")
        add(path.stem, [path], "auto")
        return items
    if not path.is_dir():
        raise FileNotFoundError(f"no such file or directory: {path}")

    index = path / "index.json"
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            data = {}
        for entry in data.get("traces") or []:
            if not isinstance(entry, dict) or not entry.get("trace_id"):
                continue
            files = [
                path / f["path"]
                for f in entry.get("files") or []
                if isinstance(f, dict) and f.get("path")
            ]
            files = [f for f in files if f.is_file()]
            if files:
                add(str(entry["trace_id"]), files, str(entry.get("source") or "auto"))
        if items:
            return items

    roots = [path]
    traces_dir = path / "traces"
    if traces_dir.is_dir():
        roots.append(traces_dir)
    for root in roots:
        for child in sorted(root.iterdir()):
            if child.is_file() and looks_like_trace(child):
                add(child.stem, [child], "auto")
                continue
            if child.is_dir():
                bundle = _tavii_bundle(child)
                if bundle:
                    add(child.name, bundle, "event-log")
    if not items:
        raise FileNotFoundError(f"no recognizable traces at {path}")
    return items


def apply_overlays(
    trace: CanonicalTrace, task: str = "", instructions: str = ""
) -> CanonicalTrace:
    if task:
        trace.task = task
        trace.meta["task_overlay"] = True
    if instructions:
        trace.instructions = instructions
        trace.meta["instructions_overlay"] = True
    return trace


def read_prompt_value(value: str) -> str:
    if not value:
        return ""
    raw = value[1:] if value.startswith("@") else value
    path = Path(raw)
    explicit = value.startswith("@")
    looks_file = path.suffix.lower() in _PROMPT_SUFFIXES and len(raw) < 512
    if (explicit or looks_file) and path.is_file():
        return path.read_text(encoding="utf-8-sig")
    return value
