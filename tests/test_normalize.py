import json
from pathlib import Path

from traceaudit.normalize import (
    apply_overlays,
    ingest_doc,
    load_dataset,
    load_trace,
    read_prompt_value,
)


def test_canonical_example(examples: Path):
    trace = load_trace("contradict", "", [str(examples / "contradict.json")])
    assert trace.task.startswith("What is the temperature")
    assert any(s.kind == "tool_call" and s.name == "get_weather" for s in trace.steps)
    assert any(s.kind == "tool_result" for s in trace.steps)
    assert trace.steps[-1].content.startswith("It is currently")


def test_openai_chat_example(examples: Path):
    trace = load_trace("wx", "", [str(examples / "openai_chat.json")])
    assert trace.instructions
    assert "Lisbon" in trace.task
    names = [s.name for s in trace.steps if s.kind == "tool_call"]
    assert "get_weather" in names
    assert any(s.kind == "tool_result" for s in trace.steps)
    assert any(t.name == "get_weather" for t in trace.tools)


def test_anthropic_blocks():
    doc = {
        "system": "Use the weather tool.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Temp in Oslo?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "c1",
                        "name": "get_weather",
                        "input": {"city": "Oslo"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "c1",
                        "content": '{"temp_c": 12}',
                    },
                ],
            },
        ],
        "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
    }
    trace = ingest_doc("anth", doc, 4000)
    assert trace.task == "Temp in Oslo?"
    assert trace.instructions == "Use the weather tool."
    assert [s.kind for s in trace.steps] == ["message", "tool_call", "tool_result"]
    assert trace.steps[1].arguments == {"city": "Oslo"}


def test_overlays_and_prompt_file(tmp_path: Path, examples: Path):
    trace = load_trace("contradict", "", [str(examples / "contradict.json")])
    apply_overlays(trace, task="Ask Oslo", instructions="Quote the tool.")
    assert trace.task == "Ask Oslo"
    assert trace.instructions == "Quote the tool."
    policy = tmp_path / "policy.md"
    policy.write_text("No refunds without auth.\n", encoding="utf-8-sig")
    assert read_prompt_value(f"@{policy}") == "No refunds without auth.\n"


def test_folder_dataset(examples: Path):
    dataset = load_dataset(examples)
    assert "contradict" in dataset
    assert "openai_chat" in dataset


def test_dev_dataset_and_gold_steps(repo_root: Path):
    traces = repo_root / "data" / "dev"
    if not (traces / "index.json").is_file():
        return
    dataset = load_dataset(traces)
    assert len(dataset) == 17
    expected = {
        "external-009": [(33, "get_relations"), (36, "get_attributes")],
        "external-017": [(15, "modify_pending_order_items")],
        "external-041": [(3, "degrees_to_radians"), (12, "degrees_to_radians")],
        "external-042": [(3, "clear-thought-clear_thought")],
        "tavii-022": [(4, "message")],
    }
    for tid, probes in expected.items():
        trace = load_trace(tid, dataset[tid].get("source") or "", dataset[tid]["files"])
        assert trace.steps, tid
        for idx, token in probes:
            assert 0 <= idx < len(trace.steps), f"{tid} step {idx} missing"
            blob = trace.steps[idx].text().lower()
            assert token.lower() in blob, (
                f"{tid} step {idx} lacks {token!r}: {blob[:160]}"
            )


def test_act_bash_and_event_log():
    doc = [
        {
            "from": "human",
            "value": "You are a linux OS assistant.\n\nNow, my problem is:\ncount /etc",
        },
        {
            "from": "gpt",
            "value": "Think: list files\n\nAct: bash\n\n```bash\nls /etc | wc -l\n```",
        },
        {"from": "human", "value": "The output of the OS:\n220"},
        {"from": "gpt", "value": "Think: done\n\nAct: answer(220)"},
    ]
    trace = ingest_doc("os", doc, 2000)
    names = [s.name for s in trace.steps if s.kind == "tool_call"]
    assert "bash" in names
    assert any(s.kind == "tool_result" and "220" in s.content for s in trace.steps)

    events = [
        {
            "seq": 1,
            "kind": "run.start",
            "data": {"input": {"prompt": "react to the message"}},
        },
        {
            "seq": 2,
            "kind": "state",
            "data": {
                "type": "openclaw.assistant_message",
                "value": {
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "message",
                            "arguments": {"action": "react", "emoji": "eyes"},
                        }
                    ]
                },
            },
        },
        {
            "seq": 3,
            "kind": "tool.result",
            "data": {
                "native_call_id": "c1",
                "name": "message",
                "status": "failed",
                "output": "messageId required",
            },
        },
    ]
    ev = ingest_doc("ev", events, 2000)
    assert ev.source_format == "event-log"
    assert ev.steps[0].name == "message"
    assert ev.steps[1].is_error
    assert "messageId required" in ev.steps[1].content


def test_toucan_unwraps_stringified_messages():
    doc = {
        "messages": json.dumps(
            [
                {
                    "role": "system",
                    "content": "<|im_system|>tool_declare<|im_middle|>[]<|im_end|>",
                },
                {"role": "user", "content": "Convert 30 degrees."},
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "degrees_to_radians",
                        "arguments": '{"degrees": 30}',
                    },
                },
                {"role": "function", "name": "degrees_to_radians", "content": "0.52"},
            ]
        ),
        "available_tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "degrees_to_radians",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        ),
    }
    trace = ingest_doc("toucan", doc, 2000)
    assert trace.steps[0].role == "system"
    assert trace.steps[3].name == "degrees_to_radians"
    assert any(t.name == "degrees_to_radians" for t in trace.tools)


def test_message_keeps_agent_name():
    doc = {
        "task": "Count the stops.",
        "messages": [
            {"role": "user", "content": "How many stops?"},
            {
                "role": "assistant",
                "name": "Orchestrator (thought)",
                "content": "Ask the specialist.",
            },
            {
                "role": "user",
                "name": "Verification_Expert",
                "content": "I will list the stops from memory.",
            },
        ],
    }
    trace = ingest_doc("ww", doc, 2000)
    names = [s.name for s in trace.steps if s.kind == "message"]
    assert "Orchestrator (thought)" in names
    assert "Verification_Expert" in names
    packed_name = next(s for s in trace.steps if s.name == "Verification_Expert")
    assert packed_name.role == "user"


def test_otel_span_tree_fixture():
    doc = {
        "task": "Look up weather",
        "instructions": "Quote the tool.",
        "spans": [
            {
                "span_id": "s1",
                "name": "get_weather",
                "attributes": {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "get_weather",
                    "input.value": '{"city": "Oslo"}',
                    "output.value": '{"temp_c": 12}',
                },
                "child_spans": [
                    {
                        "span_id": "s2",
                        "attributes": {
                            "openinference.span.kind": "LLM",
                            "output.value": "It is 12C in Oslo.",
                        },
                    }
                ],
            }
        ],
    }
    trace = ingest_doc("otel", doc, 2000)
    assert trace.source_format == "otel-spans"
    assert [s.kind for s in trace.steps] == ["tool_call", "tool_result", "message"]
    assert trace.steps[0].name == "get_weather"
    assert trace.steps[0].arguments == {"city": "Oslo"}
    assert "12" in trace.steps[1].content
    assert "12C" in trace.steps[2].content


def test_not_a_trace(tmp_path: Path):
    junk = tmp_path / "config.json"
    junk.write_text('{"name": "not a trace"}', encoding="utf-8")
    try:
        load_dataset(junk)
    except FileNotFoundError as exc:
        assert "does not look like" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_act_bash_and_os_result():
    doc = [
        {
            "from": "human",
            "value": "You are an assistant that will act like a person.\n\nNow, my problem is:\ncount /etc",
        },
        {
            "from": "gpt",
            "value": "Think: list files.\n\nAct: bash\n\n```bash\nls /etc | wc -l\n```",
        },
        {"from": "human", "value": "The output of the OS:\n220"},
    ]
    trace = ingest_doc("os", doc, 4000)
    names = [s.name for s in trace.steps if s.kind == "tool_call"]
    assert "bash" in names
    assert any(s.kind == "tool_result" and "220" in s.content for s in trace.steps)


def test_stringified_messages_keep_system_as_step():
    import json

    messages = [
        {"role": "system", "content": "You convert degrees."},
        {"role": "user", "content": "Convert 30 and 45."},
        {
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": "degrees_to_radians",
                "arguments": json.dumps({"degrees": 30}),
            },
        },
        {"role": "function", "name": "degrees_to_radians", "content": "0.52"},
    ]
    doc = {
        "messages": json.dumps(messages),
        "available_tools": json.dumps(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "degrees_to_radians",
                        "parameters": {"type": "object"},
                    },
                }
            ]
        ),
    }
    trace = ingest_doc("toucan", doc, 4000)
    assert trace.steps[0].role == "system"
    assert trace.steps[2].kind == "tool_call"
    assert trace.steps[2].arguments == {"degrees": 30}
    assert any(t.name == "degrees_to_radians" for t in trace.tools)


def test_event_log_tool_error():
    recs = [
        {
            "seq": 1,
            "kind": "run.start",
            "data": {"input": {"prompt": "map candidates"}},
        },
        {
            "seq": 2,
            "kind": "state",
            "data": {
                "type": "openclaw.assistant_message",
                "value": {
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "message",
                            "arguments": {"action": "react", "emoji": "eyes"},
                        }
                    ]
                },
            },
        },
        {
            "seq": 3,
            "kind": "tool.result",
            "data": {
                "native_call_id": "c1",
                "name": "message",
                "status": "failed",
                "output": {"error": "messageId required"},
            },
        },
    ]
    trace = ingest_doc("tavii", recs, 4000)
    assert trace.steps[0].kind == "tool_call"
    assert trace.steps[0].name == "message"
    assert trace.steps[1].kind == "tool_result"
    assert trace.steps[1].is_error
    assert "messageId required" in trace.steps[1].content


def test_labelled_dev_dataset(repo_root: Path):
    root = repo_root / "data" / "dev"
    if not (root / "index.json").is_file():
        return
    dataset = load_dataset(root)
    assert len(dataset) == 17
    expected = {
        "external-009": ("incorrect_tool_use", [33, 34], "Impersonated"),
        "external-017": ("instruction_violation", [13, 15], "gift"),
        "external-041": ("redundant_action", [12, 15], "degrees"),
        "external-042": ("incorrect_tool_use", [3, 8], "clear-thought"),
        "tavii-022": ("incorrect_tool_use", [4, 5], "message"),
    }
    for tid, (family, steps, needle) in expected.items():
        trace = load_trace(tid, dataset[tid]["source"], dataset[tid]["files"])
        assert len(trace.steps) > max(steps), tid
        blob = " ".join(trace.steps[i].text() for i in steps)
        assert needle.lower() in blob.lower(), (tid, family, blob[:200])
