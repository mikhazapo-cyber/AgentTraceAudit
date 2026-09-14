from pathlib import Path

from traceaudit.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "package root" in out


def test_demo_deterministic(tmp_path: Path):
    out = tmp_path / "demo"
    code = main(
        [
            "demo",
            "--deterministic",
            "--dataset",
            str(ROOT / "data" / "synthetic"),
            "--labels",
            str(ROOT / "data" / "synthetic" / "labels"),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    assert (out / "findings.jsonl").is_file()
    assert (out / "eval.md").is_file()


def test_normalize_canonical(capsys):
    code = main(
        [
            "normalize",
            "--dataset",
            str(ROOT / "data" / "synthetic" / "traces" / "missing-arg.json"),
            "--limit",
            "2000",
        ]
    )
    assert code == 0
    assert "messageId" in capsys.readouterr().out
