from pathlib import Path

from traceaudit.cli import main
from traceaudit.config import load_config
from traceaudit.llm import NullClient, PaymentRequired, ScriptedClient
from traceaudit.schemas import TraceResult


def test_missing_file() -> None:
    assert main(["no-such-trace.json"]) == 2


def test_help(capsys) -> None:
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("argparse help should SystemExit 0")
    text = capsys.readouterr().out.lower()
    assert "--labels" in text
    assert "--yes" in text
    assert "--fresh" in text


def test_no_api_key(monkeypatch, examples: Path, capsys) -> None:
    monkeypatch.setattr("traceaudit.cli.client_from_env", lambda cfg: NullClient())
    code = main([str(examples / "contradict.json")])
    assert code == 2
    err = capsys.readouterr().err
    assert "no API key" in err


def test_usage_without_args(monkeypatch, capsys) -> None:
    monkeypatch.setattr("traceaudit.cli.sys.stdin.isatty", lambda: False)
    assert main([]) == 2
    assert "traceaudit" in capsys.readouterr().out


def test_cli_writes_families(monkeypatch, examples: Path, tmp_path: Path) -> None:
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {
                "checks": [
                    {
                        "check_id": "C1",
                        "family": "evidence_contradiction",
                        "description": "claim must match the tool",
                        "condition": "stated temp differs",
                        "source_refs": ["Answer only from tool results"],
                    }
                ],
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "evidence_contradiction",
                        "steps": [2, 3],
                        "description": "Said 28C but tool returned 12",
                        "explanation": "The 12C result was already visible",
                        "evidence": [
                            {"step": 2, "quote": 'temp_c": 12'},
                            {"step": 3, "quote": "28"},
                        ],
                        "rule_ref": "Answer only from tool results",
                    }
                ],
            }
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "reason": "contradicts the tool",
                }
            ]
        }

    monkeypatch.setattr(
        "traceaudit.cli.client_from_env", lambda cfg: ScriptedClient(handler)
    )
    out = tmp_path / "out"
    code = main([str(examples / "contradict.json"), "--out", str(out)])
    assert code == 0
    csv_text = (out / "findings.csv").read_text(encoding="utf-8")
    assert "evidence_contradiction" in csv_text
    assert "contradicts_tool_output" in csv_text
    assert "confirmed" in csv_text
    report = (out / "reports" / "contradict.md").read_text(encoding="utf-8")
    assert "contradicts_tool_output" in report
    assert "80%" in report


def test_cli_labels_and_yes(monkeypatch, examples: Path, tmp_path: Path) -> None:
    def handler(stage, _messages):
        if stage in {"agent_a", "agent_b"}:
            return {
                "checks": [],
                "findings": [
                    {
                        "check_id": "C1",
                        "decision": "violated",
                        "family": "evidence_contradiction",
                        "steps": [2, 3],
                        "description": "Said 28C but tool returned 12",
                        "evidence": [
                            {"step": 2, "quote": 'temp_c": 12'},
                            {"step": 3, "quote": "28"},
                        ],
                    }
                ],
            }
        return {
            "verdicts": [
                {
                    "index": 0,
                    "verdict": "confirmed",
                    "family": "evidence_contradiction",
                    "reason": "ok",
                }
            ]
        }

    monkeypatch.setattr(
        "traceaudit.cli.client_from_env", lambda cfg: ScriptedClient(handler)
    )
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "contradict.json").write_text(
        """
        {
          "trace_id": "contradict",
          "clean": false,
          "findings": [
            {"family": "evidence_contradiction", "steps": [3], "description": "wrong temp"}
          ]
        }
        """,
        encoding="utf-8",
    )
    out = tmp_path / "out"
    code = main(
        [
            str(examples / "contradict.json"),
            "--out",
            str(out),
            "--labels",
            str(labels),
            "--yes",
        ]
    )
    assert code == 0
    eval_text = (out / "eval.json").read_text(encoding="utf-8")
    assert '"tp": 1' in eval_text
    assert "development" in eval_text


def test_cli_resumes_finished(monkeypatch, examples: Path, tmp_path: Path) -> None:
    called: list[str] = []

    def handler(stage, _messages):
        called.append(stage)
        return {"checks": [], "findings": []}

    monkeypatch.setattr(
        "traceaudit.cli.client_from_env", lambda cfg: ScriptedClient(handler)
    )
    out = tmp_path / "out"
    out.mkdir()
    prior = TraceResult(trace_id="contradict", status="ok")
    (out / "findings.jsonl").write_text(
        prior.model_dump_json() + "\n", encoding="utf-8"
    )
    code = main([str(examples / "contradict.json"), "--out", str(out), "--yes"])
    assert code == 0
    assert called == []


def test_cli_fresh_reruns(monkeypatch, examples: Path, tmp_path: Path) -> None:
    called: list[str] = []

    def handler(stage, _messages):
        called.append(stage)
        return {"checks": [], "findings": []}

    monkeypatch.setattr(
        "traceaudit.cli.client_from_env", lambda cfg: ScriptedClient(handler)
    )
    out = tmp_path / "out"
    out.mkdir()
    prior = TraceResult(trace_id="contradict", status="ok")
    (out / "findings.jsonl").write_text(
        prior.model_dump_json() + "\n", encoding="utf-8"
    )
    code = main(
        [str(examples / "contradict.json"), "--out", str(out), "--yes", "--fresh"]
    )
    assert code == 0
    assert "agent_a" in called
    assert "agent_b" in called


def test_cli_stops_on_402(monkeypatch, examples: Path, tmp_path: Path, capsys) -> None:
    calls = {"n": 0}

    def handler(_stage, _messages):
        calls["n"] += 1
        raise PaymentRequired("402 Payment Required")

    monkeypatch.setattr(
        "traceaudit.cli.client_from_env", lambda cfg: ScriptedClient(handler)
    )
    out = tmp_path / "out"
    code = main([str(examples), "--out", str(out), "--yes"])
    assert code == 0
    text = capsys.readouterr().out.lower()
    assert "payment required" in text
    assert calls["n"] <= 2


def test_env_cost_cap_at_mean_uses_abort_floor(monkeypatch) -> None:
    monkeypatch.setenv("TRACEAUDIT_COST_CAP", "1.5")
    cfg = load_config()
    assert cfg.cost_cap_usd == 3.0


def test_env_min_confidence(monkeypatch) -> None:
    monkeypatch.setenv("TRACEAUDIT_MIN_CONFIDENCE", "85")
    cfg = load_config()
    assert cfg.min_confidence == 85
