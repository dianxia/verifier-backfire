# -*- coding: utf-8 -*-
"""The Generation-Verification-Repair (GVR) loop with full instrumentation.

Every attempt's ground-truth correctness is measured silently by the
oracle (regardless of which arm's verifier decides accept/reject), so
empirical (a, b) operating points can be estimated per arm.
"""
from .llm import chat
from .sandbox import extract_code, run_oracle
from .verifiers import llm_judge_verify, oracle_verify

GEN_MODEL = "qwen-plus"
GEN_TEMPERATURE = 1.0
GEN_MAX_TOKENS = 1024

ARMS: dict[str, dict] = {
    "none": {"verifier": "none"},
    "oracle_full": {"verifier": "oracle", "mode": "full"},
    "oracle_verdict": {"verifier": "oracle", "mode": "verdict"},
    "judge_hetero": {
        "verifier": "llm", "model": "qwen-max", "ctx": "iso",
    },
    "judge_self_iso": {
        "verifier": "llm", "model": "qwen-plus", "ctx": "iso",
    },
    "judge_self_ctx": {
        "verifier": "llm", "model": "qwen-plus", "ctx": "ctx",
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


def run_gvr(task: dict, arm_name: str) -> dict:
    """Run one (task, arm). Returns the full instrumentation record."""
    arm = ARMS[arm_name]
    attempts: list[dict] = []
    prev_code = None
    feedback = None
    delivered_correct = None
    resolved = False

    for k in range(1, BUDGET + 1):
        # ---- generate ----
        text, tok = chat(
            GEN_MODEL,
            _gen_messages(task, k, prev_code, feedback),
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE,
        )
        code = extract_code(text, task["entry_point"])
        gen_tokens = tok

        # ---- silent ground truth (always) ----
        gt_pass, gt_out = run_oracle(
            code, task["test"], task["entry_point"],
        )

        # ---- arm-specific verdict ----
        verdict, fb, raw, jtok = "accept", "", "", 0
        if arm["verifier"] == "oracle":
            verdict, fb, raw, jtok = oracle_verify(
                task, code, arm["mode"],
            )
        elif arm["verifier"] == "llm":
            verdict, fb, raw, jtok = llm_judge_verify(
                task, code, text, arm["model"], arm["ctx"],
            )
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
            "gen_tokens": gen_tokens,
            "judge_tokens": jtok,
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
        "n_attempts": n_attempts,
        "resolved": resolved,
        "delivered_correct": delivered_correct,
        "leak": leak,
        "unresolved": not resolved,
        "attempts": attempts,
    }