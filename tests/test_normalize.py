from pathlib import Path

from traceaudit.normalize import load_trace
from traceaudit.normalize.generic import load_canonical
from traceaudit.normalize.input import detect_source

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_roundtrip():
    path = ROOT / "data" / "synthetic" / "traces" / "auth-skip.json"
    trace = load_trace("auth-skip", "canonical", [str(path)], 2000)
    assert trace.task
    assert any(s.kind == "tool_call" and s.name == "refund_order" for s in trace.steps)
    assert detect_source(path) == "canonical"


def test_toucan_extracts_tools_and_calls():
    path = ROOT / "data" / "dev" / "traces" / "external-041.json"
    trace = load_trace("external-041", "toucan-mcp", [str(path)], 4000)
    names = {s.name for s in trace.steps if s.kind == "tool_call"}
    assert any("degrees_to_radians" in n for n in names)
    assert trace.tools


def test_agentinstruct_parses_actions():
    path = ROOT / "data" / "dev" / "traces" / "external-009.json"
    trace = load_trace("external-009", "agentinstruct", [str(path)], 4000)
    assert any(s.kind == "tool_call" for s in trace.steps)
    assert "get_attributes" in " ".join(s.text() for s in trace.steps)
    assert "superlative" in (trace.instructions or "").lower()


def test_load_canonical_helper():
    doc = {
        "trace_id": "x",
        "task": "t",
        "instructions": "i",
        "tools": [{"name": "a", "parameters": {}}],
        "steps": [{"kind": "message", "role": "user", "content": "hi"}],
    }
    trace = load_canonical("x", doc, 100)
    assert trace.steps[0].content == "hi"
