from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

# OpenRouter list prices, September 2026. Used when the API does not return usage.cost.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-opus-5": (5.00, 25.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "openai/gpt-5.6-sol": (5.00, 30.00),
    "openai/gpt-4.1": (2.00, 8.00),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "default": (5.00, 25.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES_PER_MTOK["default"]
    for key, pair in PRICES_PER_MTOK.items():
        if key != "default" and (model.endswith(key) or key in model):
            inp, out = pair
            break
    return (input_tokens * inp + output_tokens * out) / 1_000_000


class StageConfig(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["anthropic/claude-opus-5"])
    temperature: float = 0.0
    reasoning_effort: str = "high"


class Config(BaseModel):
    name: str = "default"
    notes: str = (
        "Two high-power auditors in parallel, then one judge. "
        "No per-call output cap. Pack size holds the $1.50 mean. Hard abort $3."
    )
    agent_a: StageConfig = Field(
        default_factory=lambda: StageConfig(
            models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
            reasoning_effort="xhigh",
        )
    )
    agent_b: StageConfig = Field(
        default_factory=lambda: StageConfig(
            models=["openai/gpt-5.6-sol", "anthropic/claude-opus-5"],
            reasoning_effort="xhigh",
        )
    )
    judge: StageConfig = Field(
        default_factory=lambda: StageConfig(
            models=["anthropic/claude-opus-5", "anthropic/claude-sonnet-4.6"],
            reasoning_effort="high",
        )
    )
    target_mean_usd: float = 1.50
    cost_cap_usd: float = 3.00
    max_spend_usd: float = 80.0
    max_instruction_chars: int = 24000
    max_step_chars: int = 24000
    request_timeout_s: float = 900.0
    merge_similar: bool = True
    min_confidence: int = Field(default=70, ge=0, le=100)

    def fingerprint(self) -> str:
        import hashlib

        return hashlib.sha256(
            self.model_dump_json(exclude={"notes"}).encode()
        ).hexdigest()[:16]


def _apply_env(cfg: Config) -> Config:
    if os.environ.get("TRACEAUDIT_MAX_SPEND"):
        cfg.max_spend_usd = float(os.environ["TRACEAUDIT_MAX_SPEND"])
    if os.environ.get("TRACEAUDIT_COST_CAP"):
        raw = float(os.environ["TRACEAUDIT_COST_CAP"])
        # A value at or below the mean is almost always someone setting the
        # target as the abort. That skips the judge. Use the $3 p95 abort.
        if raw <= cfg.target_mean_usd + 1e-9:
            cfg.cost_cap_usd = max(3.0, cfg.target_mean_usd * 2)
        else:
            cfg.cost_cap_usd = raw
    if os.environ.get("TRACEAUDIT_MODEL"):
        model = os.environ["TRACEAUDIT_MODEL"]
        for stage in (cfg.agent_a, cfg.agent_b, cfg.judge):
            stage.models = [model]
    if os.environ.get("TRACEAUDIT_MIN_CONFIDENCE"):
        cfg.min_confidence = max(
            0, min(100, int(os.environ["TRACEAUDIT_MIN_CONFIDENCE"]))
        )
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return _apply_env(Config())
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _apply_env(Config.model_validate(data))
