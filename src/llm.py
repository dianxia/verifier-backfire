# -*- coding: utf-8 -*-
"""Minimal DashScope (OpenAI-compatible) chat client with retries.

Key loading order: env var -> project .env (repo root).
The key is never printed.
"""
import os
import time

import httpx

API_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

_KEY_CACHE = None


def _load_key() -> str:
    global _KEY_CACHE
    if _KEY_CACHE:
        return _KEY_CACHE
    key = os.environ.get("DASHSCOPE_API_KEY")
    if key:
        _KEY_CACHE = key
        return key
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env",
    )
    if os.path.exists(env_path):
        for line in open(env_path, encoding="utf-8"):
            line = line.strip()
            if line.startswith("DASHSCOPE_API_KEY="):
                _KEY_CACHE = (
                    line.split("=", 1)[1].strip().strip('"').strip("'")
                )
                return _KEY_CACHE
    raise RuntimeError(
        "DASHSCOPE_API_KEY not found in env or .env files",
    )


def chat(
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 1.0,
    retries: int = 4,
) -> tuple[str, int]:
    """One chat completion. Returns (content, total_tokens)."""
    key = _load_key()
    last_err = None
    for i in range(retries):
        try:
            r = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=180,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2 ** (i + 1))
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            tokens = int(
                data.get("usage", {}).get("total_tokens", 0),
            )
            return content, tokens
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** (i + 1))
    raise RuntimeError(
        f"DashScope call failed after {retries} retries: {last_err}",
    )