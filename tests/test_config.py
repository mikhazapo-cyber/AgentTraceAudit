import json
from pathlib import Path

from traceaudit.config import Config, cost_usd

ROOT = Path(__file__).resolve().parents[1]


def _load_file(name: str) -> Config:
    data = json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
    return Config.model_validate(data)


def test_default_json_loads_new_stages():
    cfg = _load_file("default.json")
    assert cfg.agent_a.models[0].startswith("anthropic/claude-opus")
    assert cfg.agent_b.models[0].startswith("openai/gpt-5.6-sol")
    assert cfg.judge.models[0].startswith("anthropic/claude-opus")
    assert cfg.judge.models[1].startswith("anthropic/claude-sonnet-4.6")
    assert cfg.max_trace_chars >= 160000
    assert cfg.max_instruction_chars >= 64000
    assert cfg.max_step_chars >= 24000
    assert cfg.judge_enabled
    assert cfg.derive.reasoning_effort == "high"
    assert cfg.agent_a.reasoning_effort == "xhigh"
    assert cfg.judge.reasoning_effort == "high"
    assert cfg.falsify_enabled
    assert cfg.classify_errors
    assert cfg.critique_enabled
    assert cfg.critique.reasoning_effort == "xhigh"
    assert cfg.target_mean_usd == 1.5
    assert cfg.cost_cap_usd == 3.0
    assert cfg.max_spend_usd == 40.0


def test_deterministic_json_disables_judge():
    cfg = _load_file("deterministic.json")
    assert cfg.cost_cap_usd == 0
    assert not cfg.judge_enabled


def test_migrates_old_evaluate_adjudicate_keys():
    cfg = Config.model_validate(
        {
            "evaluate": {"models": ["openai/gpt-4.1-mini"], "max_output_tokens": 100},
            "adjudicate": {"models": ["openai/gpt-4.1"], "max_output_tokens": 50},
            "adjudicate_enabled": False,
        }
    )
    assert cfg.agent_a.models == ["openai/gpt-4.1-mini"]
    assert cfg.judge.models == ["openai/gpt-4.1"]
    assert not cfg.judge_enabled


def test_opus_and_sol_prices():
    assert abs(cost_usd("anthropic/claude-opus-5", 1_000_000, 0) - 5.0) < 1e-9
    assert abs(cost_usd("openai/gpt-5.6-sol", 0, 1_000_000) - 30.0) < 1e-9
