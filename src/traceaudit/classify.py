"""Language-agnostic classification of tool-result errors.

The mechanical lane already has structural tests (schema, unsupplied argument).
This call is the backstop for runtimes that refuse a malformed call in a language
or status vocabulary we have never seen. It writes `error_kind` onto result
steps; it does not invent findings on its own.
"""

from __future__ import annotations

from .config import Config
from .llm import CallError, LLMClient, LLMResponse
from .parsing import extract_json
from .prompts import CLASSIFY_SYSTEM
from .schemas import CallRecord, CanonicalTrace
from .structural import logical_tool_name


def _record(stage: str, resp: LLMResponse) -> CallRecord:
    return CallRecord(
        stage=stage,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=resp.cost_usd,
        latency_s=resp.latency_s,
        cached=resp.cached,
    )


def classify_error_results(
    trace: CanonicalTrace,
    cfg: Config,
    client: LLMClient,
    limit: int = 8,
) -> list[CallRecord]:
    """Annotate error results with formation | resource | other. Cheap, optional."""
    if not getattr(cfg, "classify_errors", True):
        return []
    if not getattr(client, "available", True):
        return []
    errors = [
        s
        for s in trace.steps
        if s.kind == "tool_result" and s.is_error and (s.content or "").strip()
    ]
    if not errors:
        return []
    errors = errors[:limit]
    rows = []
    for step in errors:
        name = logical_tool_name(step) or step.name or "?"
        rows.append(f"#{step.index} {name}: {(step.content or '')[:400]}")
    user = (
        "Classify each tool-result error. formation = the runtime refused the "
        "call as written (schema, parse, unknown parameter, malformed). "
        "resource = the call was well-formed; a target was missing, forbidden, "
        "or unavailable. other = transient, timeout, or unclear.\n\n"
        + "\n".join(rows)
    )
    stage_cfg = cfg.classify
    try:
        resp = client.complete(
            [
                {"role": "system", "content": CLASSIFY_SYSTEM},
                {"role": "user", "content": user},
            ],
            models=stage_cfg.models,
            temperature=stage_cfg.temperature,
            max_tokens=stage_cfg.max_output_tokens,
            stage="classify_errors",
            response_schema={"type": "json_object"},
            reasoning_effort=stage_cfg.reasoning_effort,
        )
    except (CallError, Exception):
        return []
    try:
        parsed = extract_json(resp.content)
    except ValueError:
        return [_record("classify_errors", resp)]
    items = []
    if isinstance(parsed, dict):
        items = parsed.get("classifications") or parsed.get("verdicts") or []
    elif isinstance(parsed, list):
        items = parsed
    by_index = {s.index: s for s in errors}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("step", item.get("index", -1)))
        except (TypeError, ValueError):
            continue
        kind = str(item.get("kind") or item.get("error_kind") or "").lower()
        if kind not in {"formation", "resource", "other"}:
            continue
        step = by_index.get(idx)
        if step is None:
            continue
        step.meta["error_kind"] = kind
    return [_record("classify_errors", resp)]
