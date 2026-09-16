from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Config
from .config import cost_usd as price_of

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAICompatClient",
    "MockClient",
    "ScriptedClient",
    "NullClient",
    "CallError",
    "PaymentRequired",
    "LLMUnavailable",
    "client_from_env",
    "load_env_files",
]


def usage_tokens(usage: dict | None) -> tuple[int, int]:
    usage = usage or {}
    inp = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    details = (
        usage.get("completion_tokens_details")
        or usage.get("output_tokens_details")
        or {}
    )
    reasoning = 0
    for blob in (usage, details):
        if not isinstance(blob, dict):
            continue
        for key in ("reasoning_tokens", "reasoning", "native_tokens_reasoning"):
            raw = blob.get(key)
            if raw:
                reasoning = int(raw)
                break
        if reasoning:
            break
    if reasoning > out:
        out = reasoning
    return inp, out


class CallError(RuntimeError):
    pass


class PaymentRequired(CallError):
    pass


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_s: float
    cached: bool = False
    finish_reason: str = ""


class LLMClient:
    available: bool = True
    is_mock: bool = False

    def complete(
        self,
        messages: list[dict],
        *,
        models: list[str],
        temperature: float,
        stage: str,
        response_schema: dict | None = None,
        reasoning_effort: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class ResponseCache:
    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        if enabled and self.path.exists():
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["value"]
                    except (json.JSONDecodeError, KeyError):
                        continue

    @staticmethod
    def key_for(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        return self._mem.get(key)

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._mem[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n"
                )


class OpenAICompatClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        cache: ResponseCache | None = None,
        timeout: float = 900.0,
        extra_headers: dict[str, str] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self.available = bool(api_key)

    def complete(
        self,
        messages: list[dict],
        *,
        models: list[str],
        temperature: float,
        stage: str,
        response_schema: dict | None = None,
        reasoning_effort: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if not self.available:
            raise LLMUnavailable("no API key")
        last_err: Exception | None = None
        for model in models or ["anthropic/claude-opus-5"]:
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if response_schema:
                payload["response_format"] = {"type": "json_object"}
            if reasoning_effort:
                payload["reasoning"] = {
                    "effort": reasoning_effort,
                    "enabled": True,
                    "exclude": False,
                }
            cache_key = ""
            if self.cache:
                cache_key = self.cache.key_for({"stage": stage, **payload})
                hit = self.cache.get(cache_key)
                if hit:
                    return LLMResponse(
                        content=hit["content"],
                        model=hit.get("model", model),
                        input_tokens=hit.get("input_tokens", 0),
                        output_tokens=hit.get("output_tokens", 0),
                        cost_usd=hit.get("cost_usd", 0.0),
                        latency_s=0.0,
                        cached=True,
                    )
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            }
            t0 = time.monotonic()
            data = None
            timeout = httpx.Timeout(self.timeout, connect=30.0)
            try:
                with httpx.Client(timeout=timeout) as client:
                    for attempt in range(4):
                        try:
                            resp = client.post(
                                f"{self.base_url}/chat/completions",
                                headers=headers,
                                json=payload,
                            )
                            if resp.status_code == 402:
                                raise PaymentRequired(
                                    str(resp.status_code) + " " + resp.text[:200]
                                )
                            if (
                                resp.status_code in {429, 500, 502, 503, 504}
                                and attempt < 3
                            ):
                                time.sleep(2**attempt)
                                continue
                            resp.raise_for_status()
                            data = resp.json()
                            break
                        except PaymentRequired:
                            raise
                        except httpx.HTTPStatusError as exc:
                            last_err = exc
                            break
                        except (httpx.TimeoutException, httpx.TransportError) as exc:
                            last_err = exc
                            if attempt < 3:
                                time.sleep(2**attempt)
                                continue
                            break
            except PaymentRequired as exc:
                last_err = exc
                continue
            except Exception as exc:
                last_err = exc
                continue
            if data is None:
                continue
            latency = time.monotonic() - t0
            choice = (data.get("choices") or [{}])[0]
            content = ((choice.get("message") or {}).get("content")) or ""
            usage = data.get("usage") or {}
            inp, out = usage_tokens(usage)
            used_model = data.get("model") or model
            billed = usage.get("cost")
            try:
                cost = (
                    float(billed)
                    if billed is not None
                    else price_of(used_model, inp, out)
                )
            except (TypeError, ValueError):
                cost = price_of(used_model, inp, out)
            record = LLMResponse(
                content=content,
                model=used_model,
                input_tokens=inp,
                output_tokens=out,
                cost_usd=cost,
                latency_s=latency,
                finish_reason=str(choice.get("finish_reason") or ""),
            )
            if self.cache and cache_key:
                self.cache.put(
                    cache_key,
                    {
                        "content": content,
                        "model": used_model,
                        "input_tokens": inp,
                        "output_tokens": out,
                        "cost_usd": cost,
                    },
                )
            return record
        if isinstance(last_err, PaymentRequired):
            raise last_err
        raise CallError(f"all models failed for stage {stage}: {last_err}")


class NullClient(LLMClient):
    available = False
    is_mock = False

    def complete(self, *args, **kwargs) -> LLMResponse:
        raise LLMUnavailable("no API key")


class MockClient(LLMClient):
    available = True
    is_mock = True

    def complete(
        self,
        messages: list[dict],
        *,
        models: list[str],
        temperature: float,
        stage: str,
        response_schema: dict | None = None,
        reasoning_effort: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if stage == "judge":
            content = json.dumps({"verdicts": []})
        else:
            content = json.dumps({"checks": [], "findings": []})
        return LLMResponse(
            content=content,
            model="mock",
            input_tokens=max(1, sum(len(m.get("content", "")) for m in messages) // 4),
            output_tokens=max(1, len(content) // 4),
            cost_usd=0.0,
            latency_s=0.0,
        )


class ScriptedClient(LLMClient):
    available = True
    is_mock = True

    def __init__(
        self,
        handler: Callable[[str, list[dict]], Any],
        costs: dict[str, float] | None = None,
    ):
        self.handler = handler
        self.costs = costs or {}

    def complete(
        self,
        messages: list[dict],
        *,
        models: list[str],
        temperature: float,
        stage: str,
        response_schema: dict | None = None,
        reasoning_effort: str = "",
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self.handler(stage, messages)
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return LLMResponse(
            content=content,
            model="scripted",
            input_tokens=20,
            output_tokens=max(1, len(content) // 4),
            cost_usd=float(self.costs.get(stage, 0.0)),
            latency_s=0.0,
        )


def load_env_files() -> None:
    roots = [Path.cwd() / ".env"]
    try:
        roots.append(Path(__file__).resolve().parents[2] / ".env")
    except IndexError:
        pass
    seen: set[Path] = set()
    for path in roots:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def client_from_env(
    cfg: Config | None = None,
    mock: bool = False,
    cache_path: str | Path | None = None,
) -> LLMClient:
    load_env_files()
    if mock:
        return MockClient()
    key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("TRACEAUDIT_API_KEY")
        or ""
    )
    if not key:
        return NullClient()
    base = (
        os.environ.get("TRACEAUDIT_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    cache = (
        ResponseCache(cache_path or Path(".cache") / "llm.jsonl")
        if cache_path is not False
        else None
    )
    headers = {}
    if "openrouter.ai" in base:
        headers["X-Title"] = "traceaudit"
    timeout = float(getattr(cfg, "request_timeout_s", 900.0)) if cfg else 900.0
    return OpenAICompatClient(
        api_key=key,
        base_url=base,
        cache=cache,
        extra_headers=headers,
        timeout=timeout,
    )
