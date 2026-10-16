# -*- coding: utf-8 -*-
"""Subprocess oracle: run generated code against HumanEval tests."""
import re
import subprocess
import sys


def extract_code(text: str, entry_point: str) -> str:
    """Pull the python solution out of a model response."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    if blocks:
        for b in blocks:
            if f"def {entry_point}" in b:
                return b
        return blocks[-1]
    idx = text.find(f"def {entry_point}")
    if idx >= 0:
        return text[idx:]
    return text


def run_oracle(code: str, test: str, entry_point: str,
               timeout: int = 6) -> tuple[bool, str]:
    """Run `code` + `test` + `check(entry_point)` in a subprocess.

    Returns (passed, combined_output). Timeouts count as failure.
    """
    harness = (
        f"{code}\n\n{test}\n\n"
        f"try:\n    check({entry_point})\n"
        f"    print('ORACLE_PASS')\n"
        f"except Exception:\n"
        f"    import traceback; traceback.print_exc()\n"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", harness],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = "ORACLE_PASS" in r.stdout
        return passed, ((r.stdout or "") + (r.stderr or ""))[:2000]
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: exceeded time limit"
    except Exception as e:  # noqa: BLE001
        return False, f"OracleError: {e}"