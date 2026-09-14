"""OpenInference spans (Arize Phoenix / TRAIL) must normalize, keeping span ids.

`otel_genai` skips any span without a `gen_ai.*` attribute, and OpenInference
uses `openinference.span.kind` / `llm.*` / `tool.*` — so TRAIL traces fed through
that adapter produced zero steps. Span ids must survive onto each step, because
TRAIL's gold annotations are keyed by span id.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FIX = Path(__file__).parent / "fixtures"

from traceaudit import adapter_profiles  # noqa: E402
from traceaudit.normalize import _ADAPTER_MODULE_MAP, load_trace  # noqa: E402

OTLP = str(FIX / "openinference_otlp.json")
PHOENIX = str(FIX / "openinference_phoenix.json")


def test_registered_as_its_own_format():
    assert _ADAPTER_MODULE_MAP["openinference"] == "traceaudit.normalize.openinference"
    assert adapter_profiles.get("openinference") is not None


def test_trail_and_phoenix_alias_to_it():
    trace = load_trace("t", "trail", [OTLP])
    assert trace.source_format == "openinference"
    assert load_trace("t", "phoenix", [OTLP]).source_format == "openinference"


def test_otel_genai_adapter_cannot_read_it():
    """The gap this adapter exists to close."""
    trace = load_trace("t", "otel-genai", [OTLP])
    assert trace.steps == []
    assert any("gen_ai" in gap for gap in trace.capture_gaps)


def test_otlp_flattened_attributes_are_normalized():
    t = load_trace("oi-1", "openinference", [OTLP])
    assert t.instructions.startswith("Answer only from tool results")
    assert t.task == "How many moons does Mars have, and what are they called?"
    assert {s.name for s in t.tools} == {"web_search", "calculator"}
    search = next(s for s in t.tools if s.name == "web_search")
    assert search.description == "Search the web"
    assert search.parameters["properties"]["query"]["type"] == "string"

    calls = [s for s in t.steps if s.kind == "tool_call"]
    results = [s for s in t.steps if s.kind == "tool_result"]
    assert [c.name for c in calls] == ["web_search", "calculator"]
    assert calls[0].arguments == {"query": "moons of Mars"}
    assert calls[0].call_id == "call_1"
    assert results[0].call_id == "call_1"  # TOOL span paired to the LLM's call
    assert "Phobos and Deimos" in results[0].content
    assert [s.index for s in t.steps] == list(range(len(t.steps)))


def test_declared_tools_are_kept_apart_from_tools_merely_called(tmp_path):
    """Folding called names into the declared set defeats the undeclared-tool check."""
    doc = tmp_path / "undeclared.json"
    doc.write_text(
        json.dumps(
            {
                "spans": [
                    {
                        "context": {"span_id": "s1"},
                        "name": "llm",
                        "attributes": {
                            "openinference.span.kind": "LLM",
                            "llm.tools": [
                                {"tool.json_schema": {"name": "web_search", "parameters": {}}}
                            ],
                            "llm.output_messages": [
                                {
                                    "message.role": "assistant",
                                    "message.content": "using the internal tool",
                                    "message.tool_calls": [
                                        {
                                            "tool_call.id": "c1",
                                            "tool_call.function.name": "secret_digest",
                                            "tool_call.function.arguments": "{}",
                                        }
                                    ],
                                }
                            ],
                        },
                    },
                    {
                        "context": {"span_id": "s2"},
                        "name": "secret_digest",
                        "attributes": {
                            "openinference.span.kind": "TOOL",
                            "tool.name": "secret_digest",
                            "tool.call.id": "c1",
                            "output.value": "ok",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    t = load_trace("oi-4", "openinference", [str(doc)])
    assert t.tool_names() == {"web_search"}
    assert t.meta["undeclared_tools_called"] == ["secret_digest"]
    assert "secret_digest" in t.meta["observed_tools"]


def test_missing_tool_schemas_are_reported_rather_than_assumed(tmp_path):
    doc = tmp_path / "noschema.json"
    doc.write_text(
        json.dumps(
            {
                "spans": [
                    {
                        "context": {"span_id": "s1"},
                        "name": "grep",
                        "attributes": {
                            "openinference.span.kind": "TOOL",
                            "tool.name": "grep",
                            "output.value": "3 matches",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    t = load_trace("oi-5", "openinference", [str(doc)])
    assert t.tool_names() == {"grep"}
    assert t.meta["undeclared_tools_called"] == []
    assert any("declared tool set is unknown" in gap for gap in t.capture_gaps)


def test_span_ids_reach_every_step():
    t = load_trace("oi-1", "openinference", [OTLP])
    assert all(s.meta.get("span_id") for s in t.steps)
    assert {s.meta["span_id"] for s in t.steps} <= {
        "aaaa0001",
        "aaaa0002",
        "aaaa0003",
        "aaaa0004",
        "aaaa0005",
    }
    search_result = next(s for s in t.steps if s.kind == "tool_result" and s.name == "web_search")
    assert search_result.meta["span_id"] == "aaaa0003"
    assert search_result.meta["span_kind"] == "TOOL"


def test_every_span_is_addressable_for_label_mapping():
    t = load_trace("oi-1", "openinference", [OTLP])
    # An AGENT span with no messages of its own still needs a step, or a
    # span-level gold label on it could not be mapped to a step index.
    assert "aaaa0001" in {s.meta.get("span_id") for s in t.steps}


def test_replayed_conversation_prefix_is_not_duplicated():
    t = load_trace("oi-1", "openinference", [OTLP])
    user_turns = [s for s in t.steps if s.role == "user" and "moons does Mars" in s.content]
    assert len(user_turns) == 1
    # The second LLM span replays system + user + assistant + tool result.
    assert t.meta["repeated_conversation_turns"] == 4
    assert len([s for s in t.steps if "I will search for the moons" in s.content]) == 1
    assert len([s for s in t.steps if "Phobos and Deimos" in s.content]) == 1
    # The genuinely new assistant turn from the second LLM span survives.
    assert any("three moons" in s.content for s in t.steps)


def test_span_status_error_marks_the_result():
    t = load_trace("oi-1", "openinference", [OTLP])
    calc = next(s for s in t.steps if s.kind == "tool_result" and s.name == "calculator")
    assert calc.is_error is True


def test_phoenix_style_nested_attributes_are_normalized():
    t = load_trace("oi-2", "openinference", [PHOENIX])
    assert t.instructions.startswith("Never book without confirming")
    assert t.task == "Book the cheapest flight to Lisbon."
    call = next(s for s in t.steps if s.kind == "tool_call")
    assert call.name == "book_flight"
    assert call.arguments == {"flight": "TP1234", "price_eur": 412}
    result = next(s for s in t.steps if s.kind == "tool_result" and s.name == "book_flight")
    assert "XZ88QP" in result.content
    assert result.call_id == call.call_id


def test_retriever_documents_become_a_tool_result():
    t = load_trace("oi-2", "openinference", [PHOENIX])
    retrieved = next(s for s in t.steps if s.meta.get("span_kind") == "RETRIEVER" and s.kind == "tool_result")
    assert "require explicit user confirmation" in retrieved.content


def test_span_kinds_are_counted_in_meta():
    t = load_trace("oi-1", "openinference", [OTLP])
    assert t.meta["span_kinds"] == {"AGENT": 1, "LLM": 2, "TOOL": 2}
    assert t.meta["n_spans"] == 5


def test_unanswered_tool_call_is_recorded_as_a_gap(tmp_path):
    doc = tmp_path / "lonely.json"
    doc.write_text(
        '{"spans": [{"context": {"span_id": "c1"}, "name": "llm", "attributes": {'
        '"openinference.span.kind": "LLM",'
        '"llm.output_messages": [{"message.role": "assistant", "message.content": "calling",'
        ' "message.tool_calls": [{"tool_call.id": "x1", "tool_call.function.name": "ghost_tool",'
        ' "tool_call.function.arguments": "{}"}]}]}}]}',
        encoding="utf-8",
    )
    t = load_trace("oi-3", "openinference", [str(doc)])
    assert any("ghost_tool" in gap for gap in t.capture_gaps)


def test_detector_recognises_openinference_over_otel_genai():
    doc = json.loads(Path(OTLP).read_text(encoding="utf-8"))
    ranked = adapter_profiles.detect(doc, OTLP)
    assert ranked[0][0] == "openinference"
    assert ranked[0][1] > 0
