# -*- coding: utf-8 -*-
"""The Generation-Verification-Repair (GVR) loop with full instrumentation.

Every attempt's ground-truth correctness is measured silently by the
oracle (regardless of which arm's verifier decides accept/reject), so
empirical (a, b) operating points can be estimated per arm.

Rerun (v2) instrumentation changes:
- GEN_MAX_TOKENS raised 1024 -> 8192: the original cap truncated a
  large share of retry-round completions (retry prompts regenerate the
  full plan + fixed code and need more headroom).
- per-attempt prompt/completion token counts are recorded separately;
- an ast.parse health flag (`parse_ok`) is recorded per attempt;
- the API-reported model identifiers are stored per run (version
  pinning);
- two cross-family judge arms (deepseek-v4-pro, glm-5.2) are added;
  both are reasoning models and get a larger judge max_tokens budget.
- a same-family reasoning-judge control arm (judge_self_iso_thinking:
  qwen-plus with enable_thinking, 4,096-token verdict budget) bridges
  the family x capability confound of the cross-family arms.
"""
import ast

from .llm import chat
from .sandbox import extract_code, run_oracle
from .verifiers import llm_judge_verify, oracle_verify

GEN_MODEL = "qwen-plus"
GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 8192

ARMS: dict[str, dict] = {
    "none": {"verifier": "none"},
    "oracle_full": {"verifier": "oracle", "mode": "full"},
    "oracle_verdict": {"verifier": "oracle", "mode": "verdict"},
    "judge_hetero": {
        "verifier": "llm", "model": "qwen-max", "ctx": "iso",
        "judge_max_tokens": 512,
    },
    "judge_self_iso": {
        "verifier": "llm", "model": "qwen-plus", "ctx": "iso",
        "judge_max_tokens": 512,
    },
    "judge_self_ctx": {
        "verifier": "llm", "model": "qwen-plus", "ctx": "ctx",
        "judge_max_tokens": 512,
    },
    "judge_deepseek_iso": {
        "verifier": "llm", "model": "deepseek-v4-pro", "ctx": "iso",
        "judge_max_tokens": 4096,  # reasoning model: budget for thinking
    },
    "judge_glm_iso": {
        "verifier": "llm", "model": "glm-5.2", "ctx": "iso",
        "judge_max_tokens": 4096,  # reasoning model
    },
    "judge_self_iso_thinking": {
        "verifier": "llm", "model": "qwen-plus", "ctx": "iso",
        "judge_max_tokens": 4096,  # reasoning mode: budget for thinking
        "judge_thinking": True,    # bridge: same family x reasoning x 4096
    },
}

BUDGET = 5


def _gen_messages(task: dict, attempt: int, prev_code: str | None,
                  feedback: str | None) -> list[dict]:
    base = (
        f"{task['prompt']}\n\nProvide a complete, correct Python "
        "solution. First write a brief plan (2-3 sentences), then "
        "give the final code in a single ```python``` code block."
    )
    if attempt == 1:
        return [{"role": "user", "content": base}]
    content = (
        f"{task['prompt']}\n\nYour previous solution was rejected.\n\n"
        f"Previous code:\n```python\n{prev_code}\n```\n\n"
        f"Reviewer feedback:\n{feedback}\n\n"
        "Provide a corrected, complete Python solution. First write a "
        "brief plan, then the final code in a single ```python``` block."
    )
    return [{"role": "user", "content": content}]


def _parse_ok(code: str) -> bool:
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except Exception:  # noqa: BLE001
        return False


def run_gvr(task: dict, arm_name: str) -> dict:
    """Run one (task, arm). Returns the full instrumentation record."""
    arm = ARMS[arm_name]
    attempts: list[dict] = []
    prev_code = None
    feedback = None
    delivered_correct = None
    resolved = False
    gen_model_api_id = None
    judge_model_api_id = None

    for k in range(1, BUDGET + 1):
        # ---- generate ----
        text, gu = chat(
            GEN_MODEL,
            _gen_messages(task, k, prev_code, feedback),
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
        )
        if gen_model_api_id is None:
            gen_model_api_id = gu.get("model")
        code = extract_code(text, task["entry_point"])

        # ---- silent ground truth (always) ----
        gt_pass, gt_out = run_oracle(
            code, task["test"], task["entry_point"],
        )

        # ---- arm-specific verdict ----
        verdict, fb, raw, ju = "accept", "", "", {"total_tokens": 0}
        if arm["verifier"] == "oracle":
            verdict, fb, raw, ju = oracle_verify(
                task, code, arm["mode"],
            )
        elif arm["verifier"] == "llm":
            verdict, fb, raw, ju = llm_judge_verify(
                task, code, text, arm["model"], arm["ctx"],
                max_tokens=arm.get("judge_max_tokens", 512),
                thinking=arm.get("judge_thinking", False),
            )
            if judge_model_api_id is None:
                judge_model_api_id = ju.get("model")
        # arm "none": always accept the first attempt

        attempts.append({
            "k": k,
            "gen_text": text[:2000],
            "code": code,
            "gt_pass": gt_pass,
            "gt_out": gt_out[:500],
            "verdict": verdict,
            "feedback": fb[:1500],
            "judge_raw": raw[:1000],
            "gen_tokens": gu.get("total_tokens", 0),
            "gen_prompt_tokens": gu.get("prompt_tokens", 0),
            "gen_completion_tokens": gu.get("completion_tokens", 0),
            "parse_ok": _parse_ok(code),
            "judge_tokens": ju.get("total_tokens", 0),
        })

        if verdict == "accept":
            delivered_correct = gt_pass
            resolved = True
            break
        prev_code = code
        feedback = fb

    n_attempts = len(attempts)
    leak = bool(resolved and delivered_correct is False)
    return {
        "arm": arm_name,
        "task_id": task["task_id"],
        "gen_model": GEN_MODEL,
        "gen_model_api_id": gen_model_api_id,
        "judge_model": arm.get("model"),
        "judge_model_api_id": judge_model_api_id,
        "n_attempts": n_attempts,
        "resolved": resolved,
        "delivered_correct": delivered_correct,
        "leak": leak,
        "unresolved": not resolved,
        "attempts": attempts,
    }