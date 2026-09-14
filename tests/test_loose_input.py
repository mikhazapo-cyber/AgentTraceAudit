"""A raw file or folder of traces is enough — index.json is optional."""

import json
from pathlib import Path

import pytest

from traceaudit.normalize import load_dataset, load_trace
from traceaudit.normalize.input import (
    apply_overlays,
    detect_source,
    looks_like_trace,
    read_prompt_value,
)

ROOT = Path(__file__).resolve().parents[1]


def test_single_file_is_a_valid_dataset():
    path = ROOT / "data" / "dev" / "traces" / "external-009.json"
    entries = load_dataset(path)
    assert list(entries) == ["external-009"]
    assert entries["external-009"]["source"] == "agentinstruct"
    trace = load_trace("external-009", entries["external-009"]["source"], entries["external-009"]["files"])
    assert trace.steps
    assert "get_relations" in (trace.instructions or "")


def test_folder_of_json_files_without_index(tmp_path):
    src = ROOT / "data" / "dev" / "traces" / "external-001.json"
    (tmp_path / "run.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    entries = load_dataset(tmp_path)
    assert "run" in entries
    assert entries["run"]["source"] == "agentinstruct"


def test_empty_folder_is_not_a_dataset(tmp_path):
    (tmp_path / "readme.txt").write_text("not a trace", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no recognizable traces"):
        load_dataset(tmp_path)


def test_detects_openai_chat_shape(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(
        json.dumps(
            {
                "model": "gpt-4.1",
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Say hi."},
                    {"role": "assistant", "content": "Hi."},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert looks_like_trace(path)
    assert detect_source(path) == "openai-chat"
    entries = load_dataset(path)
    trace = load_trace("chat", entries["chat"]["source"], entries["chat"]["files"])
    assert trace.task.startswith("Say hi")
    assert trace.instructions == "Be brief."


def test_source_override(tmp_path):
    src = ROOT / "data" / "dev" / "traces" / "external-009.json"
    dest = tmp_path / "mystery.json"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    entries = load_dataset(dest, source_override="agentinstruct")
    assert entries["mystery"]["source"] == "agentinstruct"


def test_task_and_instruction_overlays():
    path = ROOT / "data" / "dev" / "traces" / "external-001.json"
    entries = load_dataset(path)
    tid, meta = next(iter(entries.items()))
    trace = load_trace(tid, meta["source"], meta["files"])
    apply_overlays(trace, task="overridden task", instructions="overridden policy")
    assert trace.task == "overridden task"
    assert trace.instructions == "overridden policy"
    assert trace.meta.get("task_overlay") is True


def test_read_prompt_value_from_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Never skip auth.", encoding="utf-8")
    assert read_prompt_value(f"@{path}") == "Never skip auth."
    assert read_prompt_value("literal") == "literal"
