from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

# Approximate USD per million tokens (OpenRouter list, Sept 2026).
# Used only for budgeting and reported cost; billed usage may differ.
# OpenRouter pay-as-you-go adds a ~5.5% platform fee on top of these rates.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-opus-4.6": (5.00, 25.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-sonnet-4": (3.00, 15.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "openai/gpt-5.6-sol": (5.00, 30.00),
    "openai/gpt-5.6-terra": (2.00, 12.00),
    "openai/gpt-5.6-luna": (1.00, 6.00),
    "openai/gpt-5.4": (2.50, 15.00),
    "openai/gpt-4.1": (2.00, 8.00),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-4o": (2.50, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "google/gemini-3.1-pro": (2.00, 12.00),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "google/gemini-2.5-flash": (0.15, 0.60),
    "deepseek/deepseek-v4-pro-0813": (0.66, 1.98),
    "deepseek/deepseek-v4.1-flash": (0.14, 0.28),
    "deepseek/deepseek-chat": (0.27, 1.10),
    "z-ai/glm-5.3-flash": (0.10, 0.40),
    "z-ai/glm-4.6": (0.50, 1.75),
    "qwen/qwen3.8-flash": (0.10, 0.40),
    "default": (5.00, 25.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES_PER_MTOK.get(model) or PRICES_PER_MTOK["default"]
    for key, pair in PRICES_PER_MTOK.items():
        if key != "default" and (model.endswith(key) or key in model):
            inp, out = pair
            break
    return (input_tokens * inp + output_tokens * out) / 1_000_000


class StageConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["anthropic/claude-opus-5"])
    temperature: float = 0.0
    max_output_tokens: int = 6000
    reasoning_effort: str = ""


def _opus() -> StageConfig:
    return StageConfig(
        models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
        max_output_tokens=12000,
        temperature=0.0,
        reasoning_effort="high",
    )


def _opus_auditor() -> StageConfig:
    return StageConfig(
        models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
        max_output_tokens=16000,
        temperature=0.0,
        reasoning_effort="xhigh",
    )


def _sol_auditor() -> StageConfig:
    return StageConfig(
        models=["openai/gpt-5.6-sol", "anthropic/claude-opus-5"],
        max_output_tokens=16000,
        temperature=0.0,
        reasoning_effort="xhigh",
    )


def _opus_judge() -> StageConfig:
    return StageConfig(
        models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
        max_output_tokens=12000,
        temperature=0.0,
        reasoning_effort="high",
    )


def _mini_classify() -> StageConfig:
    return StageConfig(
        models=["openai/gpt-4.1-mini", "google/gemini-2.5-flash"],
        max_output_tokens=800,
        temperature=0.0,
    )


def _opus_falsify() -> StageConfig:
    return StageConfig(
        models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
        max_output_tokens=3000,
        temperature=0.0,
        reasoning_effort="high",
    )


def _opus_critique() -> StageConfig:
    return StageConfig(
        models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
        max_output_tokens=12000,
        temperature=0.0,
        reasoning_effort="xhigh",
    )


class Config(BaseModel):
    name: str = "default"
    notes: str = ""
    derive: StageConfig = Field(default_factory=_opus)
    agent_a: StageConfig = Field(default_factory=_opus_auditor)
    agent_b: StageConfig = Field(default_factory=_sol_auditor)
    judge: StageConfig = Field(default_factory=_opus_judge)
    classify: StageConfig = Field(default_factory=_mini_classify)
    falsify: StageConfig = Field(default_factory=_opus_falsify)
    critique: StageConfig = Field(default_factory=_opus_critique)
    target_mean_usd: float = 1.50
    cost_cap_usd: float = 3.00
    max_spend_usd: float = 40.0
    judge_min_budget_usd: float = 0.25
    classify_errors: bool = True
    falsify_enabled: bool = True
    critique_enabled: bool = True
    max_falsify_per_trace: int = 3
    max_trace_chars: int = 200000
    max_instruction_chars: int = 64000
    max_step_chars: int = 24000
    judge_enabled: bool = True
    confirm_recovered_formation_errors: bool = True
    merge_similar: bool = True
    # Aliases kept so older configs and --deterministic JSON still load.
    adjudicate_enabled: bool = True
    adjudicate_min_budget_usd: float = 0.25

    @model_validator(mode="before")
    @classmethod
    def _migrate_stage_names(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "evaluate" in data and "agent_a" not in data:
            data["agent_a"] = data.pop("evaluate")
        elif "evaluate" in data:
            data.pop("evaluate", None)
        if "adjudicate" in data and "judge" not in data:
            data["judge"] = data.pop("adjudicate")
        elif "adjudicate" in data:
            data.pop("adjudicate", None)
        if "adjudicate_enabled" in data and "judge_enabled" not in data:
            data["judge_enabled"] = data["adjudicate_enabled"]
        elif "judge_enabled" in data:
            # An explicit new-style key wins; otherwise the stale alias below would
            # AND them together and silently switch the judge off for the whole run.
            data["adjudicate_enabled"] = data["judge_enabled"]
        if "adjudicate_min_budget_usd" in data and "judge_min_budget_usd" not in data:
            data["judge_min_budget_usd"] = data["adjudicate_min_budget_usd"]
        return data

    @model_validator(mode="after")
    def _sync_judge_aliases(self) -> Config:
        if not self.judge_enabled or not self.adjudicate_enabled:
            self.judge_enabled = False
            self.adjudicate_enabled = False
        if self.judge_min_budget_usd != 0.25:
            self.adjudicate_min_budget_usd = self.judge_min_budget_usd
        elif self.adjudicate_min_budget_usd != 0.25:
            self.judge_min_budget_usd = self.adjudicate_min_budget_usd
        return self

    def llm_stages(self) -> tuple[StageConfig, ...]:
        return self.derive, self.agent_a, self.agent_b, self.judge, self.critique

    def fingerprint(self) -> str:
        import hashlib

        blob = self.model_dump_json(exclude={"notes"})
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _apply_env(cfg: Config) -> Config:
    if os.environ.get("TRACEAUDIT_MAX_SPEND"):
        cfg.max_spend_usd = float(os.environ["TRACEAUDIT_MAX_SPEND"])
    if os.environ.get("TRACEAUDIT_COST_CAP"):
        cfg.cost_cap_usd = float(os.environ["TRACEAUDIT_COST_CAP"])
    if os.environ.get("TRACEAUDIT_MODEL"):
        model = os.environ["TRACEAUDIT_MODEL"]
        for stage in cfg.llm_stages():
            stage.models = [model]
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return _apply_env(Config())
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _apply_env(Config.model_validate(data))


def default_config_dict() -> dict[str, Any]:
    return Config().model_dump()
