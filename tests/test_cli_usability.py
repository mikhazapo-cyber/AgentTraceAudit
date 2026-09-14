"""Can a user with one trace file of their own actually use this?

Covers the parts a reader judges the tool by: the terminal output, the per-trace
report, and what happens when something is wrong. Nothing here calls a model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from traceaudit.cli import main
from traceaudit.eval.metrics import _split_label, evaluate
from traceaudit.llm import NullClient
from traceaudit.report import (
    ABSTAINED_NOTE,
    INSTRUCTION_QUOTE,
    MECHANICAL_RULE,
    TOOL_RESULT_EXCERPT,
    ascii_safe,
    build_summary,
    excerpt_sources,
    split_capture_gaps,
    step_span,
    trace_console_lines,
    trace_report,
)
from traceaudit.schemas import (
    CanonicalStep,
    CanonicalTrace,
    Evidence,
    Finding,
    ToolSpec,
    TraceResult,
    Verification,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "openai_chat.json"
SYNTHETIC = ROOT / "data" / "synthetic"


# --------------------------------------------------------------------------
# fixtures built by hand: no model, no network
# --------------------------------------------------------------------------


def _trace() -> CanonicalTrace:
    return CanonicalTrace(
        trace_id="t1",
        source_format="openai-chat",
        task="Refund order ORD-1.",
        instructions="Always call check_balance before issuing a refund.",
        tools=[ToolSpec(name="check_balance"), ToolSpec(name="refund")],
        steps=[
            CanonicalStep(index=0, kind="message", role="user", content="Refund ORD-1 please"),
            CanonicalStep(index=1, kind="tool_call", name="refund", arguments={"order": "ORD-1"}),
            CanonicalStep(index=2, kind="tool_result", name="refund", content="{\"ok\": false}", is_error=True),
            CanonicalStep(index=3, kind="message", role="assistant", content="Refund done."),
        ],
    )


def _confirmed_finding() -> Finding:
    return Finding(
        finding_id="t1-F01",
        trace_id="t1",
        family="instruction_violation",
        description="Refunded without checking the balance first.",
        explanation="The rule gates refund on a prior check_balance call; there is none.",
        steps=[1, 2],
        evidence=[Evidence(step=2, quote='{"ok": false}')],
        rule_ref="Always call check_balance before issuing a refund.",
        available_info="At step 1 no check_balance result existed in the trace.",
        status="confirmed",
        verification=Verification(locatable=True, quotes_ok=True, family_gate=True, quote_verified=1, quote_total=1),
    )


def _abstained_finding() -> Finding:
    return Finding(
        finding_id="t1-F02",
        trace_id="t1",
        family="redundant_action",
        description="The second lookup may have been unnecessary.",
        steps=[3],
        status="insufficient_evidence",
        adjudication_reason="arguments differ, so it may be search",
    )


def _redundancy_finding() -> Finding:
    return Finding(
        finding_id="t1-F03",
        trace_id="t1",
        family="redundant_action",
        description="Called refund twice with identical arguments.",
        steps=[1],
        evidence=[Evidence(step=1, quote='{"order": "ORD-1"}')],
        rule_ref="identical repeat of a call that already succeeded",
        alternative="Reuse the result already returned at step 2 instead of calling refund again.",
        status="confirmed",
    )


def _result(findings: list[Finding], status: str = "ok") -> TraceResult:
    return TraceResult(trace_id="t1", status=status, findings=findings, latency_s=12.5)


# --------------------------------------------------------------------------
# ASCII: this is a Windows PowerShell console
# --------------------------------------------------------------------------


def test_ascii_safe_folds_the_characters_that_mojibake():
    assert ascii_safe("a\u2014b") == "a--b"
    assert ascii_safe("19\u00b0C") == "19 degC"
    assert ascii_safe("a \u00b7 b") == "a - b"
    assert ascii_safe("derive \u2192 judge") == "derive -> judge"


def test_a_whole_run_prints_pure_ascii(tmp_path, capsys):
    # The sample trace says "19 deg C" with a real degree sign.
    assert main(["run", str(EXAMPLE), "--mock", "--show-report", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    offenders = sorted({c for c in printed if ord(c) > 127})
    assert not offenders, offenders


def test_deterministic_dev_run_prints_pure_ascii(tmp_path, capsys):
    assert main(["run", str(SYNTHETIC), "--deterministic", "--out", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert not {c for c in printed if ord(c) > 127}


# --------------------------------------------------------------------------
# one trace file just works
# --------------------------------------------------------------------------


def test_single_trace_file_positional_just_works(tmp_path, capsys):
    assert main(["run", str(EXAMPLE), "--mock", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert (tmp_path / "findings.jsonl").is_file()
    assert (tmp_path / "reports" / "openai_chat.md").is_file()
    assert "AUDIT SUMMARY" in out
    # The user is told where things went, so they need not guess.
    assert "ARTIFACTS" in out and str(tmp_path) in out


def test_positional_and_dataset_flag_agree(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", str(EXAMPLE), "--mock", "--out", str(a)]) == 0
    assert main(["run", "--dataset", str(EXAMPLE), "--mock", "--out", str(b)]) == 0
    left = json.loads((a / "findings.jsonl").read_text(encoding="utf-8"))
    right = json.loads((b / "findings.jsonl").read_text(encoding="utf-8"))
    left.pop("latency_s", None)
    right.pop("latency_s", None)
    assert left == right


def test_explain_prints_the_report_body(tmp_path, capsys):
    assert main(["explain", str(EXAMPLE), "--mock", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "# Audit report:" in out
    assert "## Verdict" in out
    assert "## What the agent was asked to do" in out
    # The task and the rule the agent was under both reach the terminal.
    assert "How warm is Lisbon?" in out
    assert "Quote the returned temperature exactly" in out


def test_task_and_instructions_from_files_reach_the_report(tmp_path, capsys):
    task = tmp_path / "task.txt"
    rules = tmp_path / "rules.md"
    task.write_text("Book the cheapest flight.", encoding="utf-8")
    rules.write_text("Never book without confirming the date.", encoding="utf-8")
    code = main(
        [
            "explain",
            str(EXAMPLE),
            "--mock",
            "--task",
            f"@{task}",
            "--instructions",
            f"@{rules}",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "using the task and instructions you supplied" in out
    assert "Book the cheapest flight." in out
    assert "Never book without confirming the date." in out


def test_a_bom_in_a_task_file_does_not_reach_the_report(tmp_path, capsys):
    """PowerShell Out-File and Notepad both write UTF-8 with a BOM."""
    task = tmp_path / "task.txt"
    task.write_text("Book the cheapest flight.", encoding="utf-8-sig")
    assert main(["explain", str(EXAMPLE), "--mock", "--task", f"@{task}", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "> Book the cheapest flight." in out
    assert "?Book" not in out


def test_formats_subcommand_lists_adapters(capsys):
    assert main(["formats"]) == 0
    out = capsys.readouterr().out
    for name in ("openai-chat", "anthropic-messages", "openinference", "langchain"):
        assert name in out


# --------------------------------------------------------------------------
# error messages a user can act on
# --------------------------------------------------------------------------


def test_missing_file_says_so_and_shows_an_example(tmp_path, capsys):
    assert main(["run", str(tmp_path / "nope.json"), "--mock"]) == 2
    err = capsys.readouterr().err
    assert "no such file or directory" in err
    assert "examples/openai_chat.json" in err


def test_empty_file_is_named_as_empty(tmp_path, capsys):
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert main(["run", str(empty), "--mock"]) == 2
    assert "empty (0 bytes)" in capsys.readouterr().err


def test_json_that_is_not_a_trace_is_rejected_with_a_hint(tmp_path, capsys):
    junk = tmp_path / "config.json"
    junk.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert main(["run", str(junk), "--mock"]) == 2
    err = capsys.readouterr().err
    assert "does not look like an agent trace" in err
    assert "--source" in err
    assert "traceaudit formats" in err


def test_unparsable_json_is_reported_as_such(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main(["run", str(bad), "--mock"]) == 2
    assert "does not look like an agent trace" in capsys.readouterr().err


def test_no_trace_argument_shows_both_spellings(capsys):
    assert main(["run", "--mock"]) == 2
    err = capsys.readouterr().err
    assert "no trace given" in err
    assert "--dataset" in err


def test_missing_api_key_explains_what_is_lost(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("traceaudit.cli.client_from_env", lambda *a, **k: NullClient())
    assert main(["run", str(EXAMPLE), "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "no API key found" in out
    assert "OPENROUTER_API_KEY" in out
    assert "--api-key" in out
    assert "--deterministic" in out


def test_deterministic_on_purpose_is_not_nagged(tmp_path, capsys):
    assert main(["run", str(EXAMPLE), "--deterministic", "--out", str(tmp_path)]) == 0
    assert "no API key found" not in capsys.readouterr().out


def test_a_trace_with_no_readable_steps_is_an_error_not_a_clean_verdict(tmp_path, capsys):
    """A file the adapter cannot read must never render as 'nothing wrong'."""
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps({"messages": []}), encoding="utf-8")
    assert main(["run", str(stub), "--mock", "--source", "openai-chat", "--out", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "ERROR" in out
    assert "no steps could be read" in out
    assert "0 confirmed - nothing provable" not in out


# --------------------------------------------------------------------------
# terminal rendering
# --------------------------------------------------------------------------


def test_per_trace_lines_break_confirmed_down_by_family():
    lines = trace_console_lines(
        _result([_confirmed_finding(), _redundancy_finding(), _abstained_finding()]), 3, 17, 1.25
    )
    joined = "\n".join(lines)
    assert "[3/17]" in joined
    assert "instruction_violation 1" in joined
    assert "redundant_action 1" in joined
    assert "2 confirmed" in joined
    assert "1 abstained" in joined
    assert "12.5s" in joined
    assert "session $1.250" in joined


def test_a_clean_trace_says_so_rather_than_printing_nothing():
    joined = "\n".join(trace_console_lines(_result([]), 1, 1))
    assert "0 confirmed" in joined
    assert "nothing provable" in joined


def test_a_failed_trace_prints_the_reason():
    bad = TraceResult(trace_id="t1", status="error", error="RuntimeError: adapter exploded")
    joined = "\n".join(trace_console_lines(bad, 1, 1))
    assert "ERROR" in joined
    assert "adapter exploded" in joined


def test_run_summary_reports_families_cost_latency_and_artifacts(tmp_path, capsys):
    assert main(["run", str(SYNTHETIC), "--deterministic", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "CONFIRMED FINDINGS" in out
    assert "incorrect_tool_use" in out
    assert "ABSTAINED" in out
    assert "never scored as a false positive" in out
    assert "COST" in out and "p95" in out
    assert "LATENCY" in out
    assert "ARTIFACTS" in out
    assert "findings.csv" in out


# --------------------------------------------------------------------------
# the per-trace markdown report
# --------------------------------------------------------------------------


def test_report_leads_with_a_verdict():
    md = trace_report(_result([_confirmed_finding()]), _trace())
    head = md.split("## What the agent was asked to do")[0]
    assert md.index("## Verdict") < md.index("## What the agent was asked to do")
    assert "1 problem(s) confirmed" in head
    assert "instruction_violation 1" in head


def test_report_states_the_ask_and_the_rules():
    md = trace_report(_result([_confirmed_finding()]), _trace())
    assert "Refund order ORD-1." in md
    assert "Always call check_balance before issuing a refund." in md
    assert "`check_balance`" in md


def test_confirmed_findings_carry_steps_rule_evidence_and_available_info():
    md = trace_report(_result([_confirmed_finding()]), _trace())
    assert "steps 1-2" in md
    assert "Rule broken, quoted from your instructions" in md
    assert "What the agent could see at that point" in md
    assert "step 2 (tool_result: refund)" in md
    assert TOOL_RESULT_EXCERPT in md


def test_repeated_evidence_is_listed_once():
    f = _confirmed_finding()
    f.evidence = [
        Evidence(step=2, quote='{"ok": false}'),
        Evidence(step=2, quote='{"ok": false}'),
        Evidence(step=1, quote='{"order": "ORD-1"}'),
    ]
    md = trace_report(_result([f]), _trace())
    assert md.count('step 2 (tool_result: refund)') == 1
    assert 'step 1 (tool_call: refund)' in md


def test_inefficiency_names_a_cheaper_alternative():
    md = trace_report(_result([_redundancy_finding()]), _trace())
    assert "Cheaper alternative available then." in md
    assert "Reuse the result already returned at step 2" in md


def test_abstentions_are_separated_and_marked_as_not_accusations():
    md = trace_report(_result([_confirmed_finding(), _abstained_finding()]), _trace())
    assert "## Abstained: insufficient evidence (NOT findings)" in md
    assert ABSTAINED_NOTE in md
    # The abstained finding must sit after the confirmed section, not inside it.
    assert md.index("t1-F01") < md.index("Abstained: insufficient evidence")
    assert md.index("Abstained: insufficient evidence") < md.index("t1-F02")


def test_an_abstention_alone_never_reads_as_a_confirmed_problem():
    md = trace_report(_result([_abstained_finding()]), _trace())
    assert "Nothing confirmed." in md
    assert "abstained" in md.lower()
    assert "1 problem(s) confirmed" not in md


def test_a_failed_trace_report_refuses_to_draw_a_conclusion():
    md = trace_report(TraceResult(trace_id="t1", status="error", error="boom"), _trace())
    assert "This trace was not audited" in md
    assert "boom" in md
    assert "No conclusion should be drawn" in md


def test_adapter_provenance_is_not_dressed_up_as_data_loss():
    loss, notes = split_capture_gaps(
        ["langchain adapter: 'langchain' (alias mapped)", "loss event at step 4: output truncated"]
    )
    assert notes == ["langchain adapter: 'langchain' (alias mapped)"]
    assert loss == ["loss event at step 4: output truncated"]
    trace = _trace()
    trace.capture_gaps = ["openai-chat adapter: 'openai-chat' (alias mapped)"]
    md = trace_report(_result([]), trace)
    assert "Capture gaps" not in md
    assert "Adapter notes" in md


def test_step_span_reads_as_prose_not_a_python_list():
    assert step_span([4]) == "step 4"
    assert step_span([4, 5, 6]) == "steps 4-6"
    assert step_span([4, 9]) == "steps 4, 9"
    assert step_span([]) == "no step located"


# --------------------------------------------------------------------------
# is the finding actually sourced?
# --------------------------------------------------------------------------


def test_a_rule_quoted_from_the_instructions_is_labelled_as_a_quote():
    sources = excerpt_sources(_confirmed_finding(), _trace())
    assert INSTRUCTION_QUOTE in sources
    assert TOOL_RESULT_EXCERPT in sources


def test_a_local_rule_name_is_not_passed_off_as_an_instruction_quote():
    sources = excerpt_sources(_redundancy_finding(), _trace())
    assert MECHANICAL_RULE in sources
    assert INSTRUCTION_QUOTE not in sources


def test_summary_counts_how_findings_are_sourced():
    summary = build_summary([_result([_confirmed_finding(), _redundancy_finding()])], {"t1": _trace()})
    cov = summary["excerpt_coverage"]
    assert cov["n_confirmed"] == 2
    assert cov["with_instruction_quote"] == 1
    assert cov["with_tool_result_excerpt"] == 1
    assert cov["with_nothing"] == 0


# --------------------------------------------------------------------------
# structured artifacts
# --------------------------------------------------------------------------


def test_summary_json_carries_per_trace_cost_latency_and_families(tmp_path):
    assert main(["run", str(SYNTHETIC), "--deterministic", "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_traces"] == 9
    assert summary["confirmed_by_family"]
    assert "total" in summary["latency_s"]
    per = {t["trace_id"]: t for t in summary["per_trace"]}
    assert per["missing-arg"]["n_confirmed"] == 1
    for entry in summary["per_trace"]:
        assert "latency_s" in entry
        assert "cost_usd" in entry
        assert "n_insufficient_evidence" in entry


def test_findings_csv_carries_the_rule_and_the_evidence(tmp_path):
    assert main(["run", str(SYNTHETIC), "--deterministic", "--out", str(tmp_path)]) == 0
    with (tmp_path / "findings.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for col in (
        "rule_ref",
        "available_info",
        "evidence_quotes",
        "excerpt_sources",
        "is_confirmed",
        "trace_latency_s",
        "alternative",
    ):
        assert col in rows[0], col
    assert any(r["rule_ref"] for r in rows)
    assert {r["is_confirmed"] for r in rows} <= {"yes", "no"}


def test_summary_md_never_lets_abstentions_read_as_findings(tmp_path):
    assert main(["run", str(SYNTHETIC), "--deterministic", "--out", str(tmp_path)]) == 0
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "confirmed finding(s)**" in md
    assert "abstained" in md
    assert ABSTAINED_NOTE in md


def test_eval_reports_latency_per_trace(tmp_path):
    assert main(
        [
            "run",
            str(SYNTHETIC),
            "--deterministic",
            "--labels",
            str(SYNTHETIC / "labels"),
            "--out",
            str(tmp_path),
        ]
    ) == 0
    report = evaluate(tmp_path / "findings.jsonl", SYNTHETIC / "labels")
    for entry in report["per_trace"].values():
        if entry.get("status") != "missing":
            assert "latency_s" in entry
    assert "latency_s" in report
    md = (tmp_path / "eval.md").read_text(encoding="utf-8")
    assert "## Per trace" in md
    assert "latency" in md


def test_eval_terminal_output_distinguishes_abstentions_and_clean_fps(tmp_path, capsys):
    assert main(
        [
            "run",
            str(SYNTHETIC),
            "--deterministic",
            "--labels",
            str(SYNTHETIC / "labels"),
            "--out",
            str(tmp_path),
        ]
    ) == 0
    capsys.readouterr()
    assert main(["eval", "--results", str(tmp_path / "findings.jsonl"), "--labels", str(SYNTHETIC / "labels")]) == 0
    out = capsys.readouterr().out
    assert "clean-trace false positives" in out
    assert "abstentions (never an FP)" in out
    assert not {c for c in out if ord(c) > 127}


def test_split_label_cannot_call_heldout_numbers_development():
    assert _split_label(["dev"]) == "development"
    assert _split_label(["heldout"]) == "held-out"
    assert _split_label(["dev", "heldout"]) == "mixed development + held-out"
