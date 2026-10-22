# -*- coding: utf-8 -*-
"""Minimal DashScope (OpenAI-compatible) chat client with retries.

Key loading order: env var -> project .env (repo root).
The key is never printed.

chat() returns (content, usage) where usage = {'prompt_tokens',
'completion_tokens', 'total_tokens', 'model'}; 'model' is the
API-reported model identifier (for version pinning).
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
    enable_thinking: bool = False,
) -> tuple[str, dict]:
    """One chat completion. Returns (content, usage dict).

    enable_thinking=True requests reasoning mode: the answer arrives
    in `content`, the chain of thought in `reasoning_content` (not
    returned here).
    """
    key = _load_key()
    last_err = None
    for i in range(retries):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if enable_thinking:
                payload["enable_thinking"] = True
            r = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=300,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}")
                time.sleep(2 ** (i + 1))
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"] or ""
            u = data.get("usage", {}) or {}
            usage = {
                "prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    u.get("completion_tokens", 0) or 0),
                "total_tokens": int(u.get("total_tokens", 0) or 0),
                "model": data.get("model", model),
            }
            return content, usage
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** (i + 1))
    raise RuntimeError(
        f"DashScope call failed after {retries} retries: {last_err}",
    )