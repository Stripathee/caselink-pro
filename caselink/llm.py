"""Thin wrapper around Nebius Token Factory (OpenAI-compatible API).

Every call is cached on disk, keyed by a hash of the full request, so a
repeated call costs nothing. Every real (non-cached) call is appended to a
usage log so we can track how much credit each part of the pipeline uses.
Safe to call from several threads at once.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

load_dotenv()

BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
CACHE_DIR = Path(os.getenv("CASELINK_CACHE_DIR", ".cache/llm"))
USAGE_LOG = Path(os.getenv("CASELINK_USAGE_LOG", ".cache/usage.jsonl"))

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)
_MAX_ATTEMPTS = 4

_client: OpenAI | None = None
_client_lock = threading.Lock()
_log_lock = threading.Lock()


def get_client() -> OpenAI:
    """Return a shared OpenAI client pointed at Nebius Token Factory."""
    global _client
    with _client_lock:
        if _client is None:
            api_key = os.getenv("NEBIUS_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "NEBIUS_API_KEY is not set. In .env, the line must read exactly "
                    "NEBIUS_API_KEY=<your key> (key name, equals sign, then the key)."
                )
            _client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=120)
        return _client


def _cache_key(request: dict[str, Any]) -> str:
    raw = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _log_usage(record: dict[str, Any]) -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": record["model"],
        "prompt_tokens": record["prompt_tokens"],
        "completion_tokens": record["completion_tokens"],
        "latency_s": record["latency_s"],
    }
    with _log_lock:
        USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def _create_with_retries(request: dict[str, Any]):
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return get_client().chat.completions.create(**request)
        except _RETRYABLE:
            if attempt == _MAX_ATTEMPTS:
                raise
            # Exponential backoff with jitter: about 2s, 4s, 8s.
            time.sleep(2**attempt + random.random())


THINKING_MODES = {
    # Nemotron 3 models reason before answering by default. The answer arrives
    # in `content` and the reasoning in `reasoning_content`; reasoning tokens
    # count against max_tokens and are billed as output.
    None: None,  # model default (reasoning on)
    "off": {"enable_thinking": False},
    "low": {"enable_thinking": True, "low_effort": True},
}


def chat(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    thinking: str | None = None,
    use_cache: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    """Send a chat request and return a plain dict.

    thinking: None (model default), "off", or "low".
    Returned keys: model, content, reasoning, finish_reason, prompt_tokens,
    completion_tokens, latency_s, cached.
    """
    if not model:
        raise ValueError("Model ID is empty.")
    if thinking not in THINKING_MODES:
        raise ValueError(f"thinking must be one of {list(THINKING_MODES)}")

    request = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra,
    }
    if THINKING_MODES[thinking]:
        request["extra_body"] = {"chat_template_kwargs": THINKING_MODES[thinking]}
    path = CACHE_DIR / f"{_cache_key(request)}.json"

    if use_cache and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        record["cached"] = True
        return record

    start = time.perf_counter()
    response = _create_with_retries(request)
    latency = time.perf_counter() - start

    choice = response.choices[0]
    message = choice.message
    usage = response.usage
    record = {
        "model": model,
        "content": message.content or "",
        # Reasoning models return their thinking separately.
        "reasoning": getattr(message, "reasoning_content", None),
        # "length" means max_tokens ran out, often while still reasoning.
        "finish_reason": getattr(choice, "finish_reason", None),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "latency_s": round(latency, 3),
        "cached": False,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic, so a crash never leaves a half-written cache file
    _log_usage(record)
    return record
