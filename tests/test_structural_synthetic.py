from pathlib import Path

from traceaudit.config import Config
from traceaudit.llm import NullClient
from traceaudit.normalize import load_dataset, load_trace
from traceaudit.pipeline import analyze_trace

ROOT = Path(__file__).resolve().parents[1]


def _result(tid: str):
    ds = load_dataset(ROOT / "data" / "synthetic")
    meta = ds[tid]
    trace = load_trace(tid, meta["source"], meta["files"], 4000)
    return analyze_trace(trace, Config(), NullClient()), trace


def _confirmed(result, family):
    return [f for f in result.findings if f.status == "confirmed" and f.family == family]


def test_missing_arg_is_incorrect_tool_use():
    result, _ = _result("missing-arg")
    hits = _confirmed(result, "incorrect_tool_use")
    assert hits
    assert set(hits[0].steps) & {1, 2}


def test_repeat_is_redundant_and_contradiction():
    result, _ = _result("repeat-search")
    assert _confirmed(result, "redundant_action")
    assert _confirmed(result, "evidence_contradiction")
    red = _confirmed(result, "redundant_action")[0]
    assert red.alternative
    assert "4" in "".join(map(str, red.steps)) or 4 in red.steps


def test_ignore_error_flags_empty_and_repeat():
    result, _ = _result("ignore-error")
    assert _confirmed(result, "incorrect_tool_use")
    assert _confirmed(result, "ignored_feedback")


def test_clean_retry_not_flagged():
    result, _ = _result("clean-retry")
    assert not [f for f in result.findings if f.status == "confirmed"]


def test_clean_search_not_redundant():
    result, _ = _result("clean-search")
    assert not _confirmed(result, "redundant_action")


def test_clean_verify_not_flagged():
    result, _ = _result("clean-verify")
    assert not [f for f in result.findings if f.status == "confirmed"]
