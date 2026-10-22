# -*- coding: utf-8 -*-
"""Verifier arms. Each verifier returns
(verdict: 'accept'|'reject', feedback: str, raw_log: str, usage: dict)
where usage = {'total_tokens': int, 'model': str}.

Arms:
  oracle_full    unit tests, feedback = failing test traceback
  oracle_verdict unit tests, feedback = verdict only (E3 ablation)
  judge_hetero   different-capability judge model (qwen-max), isolated ctx
  judge_self_iso same generator model (qwen-plus), isolated context
  judge_self_ctx same generator model, sees its own reasoning (collapse arm)
  judge_deepseek_iso / judge_glm_iso  cross-family judges; both are
                 reasoning models, so they need a larger max_tokens
                 budget for thinking (set in loop.ARMS).
  judge_self_iso_thinking  same-family reasoning control (qwen-plus
                 with enable_thinking, 4,096-token budget): bridges
                 the family/capability confound of the cross-family
                 arms.
"""
import re

from .llm import chat
from .sandbox import run_oracle

JUDGE_MAX_TOKENS = 512

_JUDGE_SUFFIX = (
    "\n\nRespond EXACTLY in this format:\n"
    "VERDICT: ACCEPT or VERDICT: REJECT\n"
    "REASON: <one short paragraph; if rejecting, name the precise flaw>"
)


def _judge_prompt(task: dict, code: str, gen_text: str, ctx: str) -> str:
    if ctx == "ctx":
        return (
            "You previously answered the programming problem below, and "
            "you are now asked to verify your own previous answer. "
            "Check it carefully for correctness.\n\n"
            f"Problem:\n{task['prompt']}\n\n"
            f"Your previous answer (including your reasoning):\n"
            f"---\n{gen_text[:2500]}\n---\n\n"
            f"Extracted solution code:\n```python\n{code}\n```"
            f"{_JUDGE_SUFFIX}"
        )
    # isolated context: judge sees only problem + code
    return (
        "Carefully review the proposed solution to the programming "
        "problem below for correctness.\n\n"
        f"Problem:\n{task['prompt']}\n\n"
        f"Proposed solution:\n```python\n{code}\n```"
        f"{_JUDGE_SUFFIX}"
    )


def _parse_verdict(text: str) -> str | None:
    m = re.search(r"VERDICT:\s*(ACCEPT|REJECT)", text, re.I)
    return m.group(1).lower() if m else None


def oracle_verify(task: dict, code: str, mode: str):
    """Unit-test oracle. mode: 'full' (traceback feedback) or 'verdict'.

    Returns (verdict, feedback, raw_output, usage).
    """
    passed, output = run_oracle(
        code, task["test"], task["entry_point"],
    )
    usage = {"total_tokens": 0, "model": "unit-test"}
    if passed:
        return "accept", "", output, usage
    if mode == "verdict":
        return (
            "reject",
            "Your solution failed the tests.",
            output,
            usage,
        )
    return (
        "reject",
        "Your solution failed the tests. The test harness produced "
        f"the following error:\n{output[-1200:]}",
        output,
        usage,
    )


def llm_judge_verify(task: dict, code: str, gen_text: str,
                     model: str, ctx: str,
                     max_tokens: int = JUDGE_MAX_TOKENS,
                     thinking: bool = False):
    """LLM judge with strict-format retries.

    thinking=True serves the judge in reasoning mode
    (enable_thinking); the formatted answer arrives in `content`,
    the chain of thought stays in `reasoning_content` upstream.

    Returns (verdict, feedback, raw_log, usage).
    """
    messages = [
        {"role": "user", "content": _judge_prompt(task, code, gen_text, ctx)},
    ]
    tokens = 0
    raw = ""
    api_model = model
    for attempt in range(3):
        text, u = chat(
            model,
            messages,
            max_tokens=max_tokens,
            temperature=0.2,
            enable_thinking=thinking,
        )
        api_model = u.get("model", model)
        tokens += u.get("total_tokens", 0)
        raw = text
        verdict = _parse_verdict(text)
        if verdict:
            reason = ""
            m = re.search(r"REASON:\s*(.+)", text, re.S | re.I)
            if m:
                reason = m.group(1).strip()[:1200]
            if verdict == "reject":
                feedback = (
                    "Your solution was rejected by review. The reviewer "
                    f"says:\n{reason}"
                )
            else:
                feedback = ""
            return verdict, feedback, raw, {
                "total_tokens": tokens, "model": api_model}
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                "Respond EXACTLY in the required format starting with "
                "'VERDICT: ACCEPT' or 'VERDICT: REJECT'."},
        ]
    # unparseable after retries: treat as reject with parse failure note
    return "reject", (
        "Your solution was rejected (reviewer response could not be "
        "parsed)."
    ), raw, {"total_tokens": tokens, "model": api_model}