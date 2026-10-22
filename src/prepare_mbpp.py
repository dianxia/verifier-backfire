# -*- coding: utf-8 -*-
"""Build data/mbpp.jsonl in the pipeline schema from the official MBPP
release.

Sources (downloaded on the fly):
  - sanitized-mbpp.json: the hand-verified subset (Austin et al., 2021)
  - mbpp.jsonl:           the full 974-task release (cross-check only)

Split rule (from the official MBPP README): task IDs 11-510 form the
test split used for evaluation.  We keep the hand-verified (sanitized)
tasks within that range, so every oracle test has been human-checked.

Each output task carries {task_id, prompt, entry_point,
canonical_solution, test}, where `test` defines check(candidate)
compatible with src/sandbox.py: setup imports first, then alias
`candidate` to the expected function name and run the original
asserts verbatim.

Self-validation: every task's reference solution must pass its own
harness (run through src.sandbox.run_oracle); tasks that fail are
reported and dropped.

Usage: python -m src.prepare_mbpp [--limit N]   # limit = smoke only
"""
import argparse
import json
import os
import re
import urllib.request

from .sandbox import run_oracle

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(_ROOT, "data", "mbpp.jsonl")

SANITIZED_URL = (
    "https://raw.githubusercontent.com/google-research/"
    "google-research/master/mbpp/sanitized-mbpp.json"
)

# Official test split (MBPP README): task IDs 11-510.
TEST_LO, TEST_HI = 11, 510


def _fetch(url: str, path: str) -> None:
    if os.path.exists(path):
        return
    print(f"downloading {url}", flush=True)
    urllib.request.urlretrieve(url, path)


def _entry_point(task: dict) -> str | None:
    """Function name the tests call: first identifier of the first
    assert (skipping 'not'); fallback: first top-level def."""
    for t in task["test_list"]:
        m = re.match(r"assert\s+(?:not\s+)?(\w+)\s*\(", t.strip())
        if m:
            return m.group(1)
    m = re.search(r"^def\s+(\w+)\s*\(", task["code"], re.M)
    return m.group(1) if m else None


def _signature(code: str, ep: str) -> str | None:
    """The reference def line for ep, e.g. 'def min_cost(cost, m, n):'"""
    m = re.search(rf"^def\s+{ep}\s*\(.*?\):", code, re.M)
    return m.group(0) if m else None


def _build_test(task: dict, ep: str) -> str:
    """check(candidate) harness: imports, alias, then raw asserts."""
    lines = []
    for imp in task.get("test_imports") or []:
        lines.append(imp)
    if lines:
        lines.append("")
    lines.append("def check(candidate):")
    lines.append(f"    global {ep}")
    lines.append(f"    {ep} = candidate")
    for a in task["test_list"]:
        for ln in a.splitlines():
            lines.append(f"    {ln}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="cap task count (smoke only)")
    args = ap.parse_args()

    tmp = "/tmp"
    san_path = os.path.join(tmp, "mbpp_sanitized.json")
    _fetch(SANITIZED_URL, san_path)
    sanitized = json.load(open(san_path, encoding="utf-8"))

    tasks = [
        t for t in sanitized
        if TEST_LO <= t["task_id"] <= TEST_HI
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"sanitized test-split candidates: {len(tasks)}", flush=True)

    out, dropped = [], []
    for t in tasks:
        code = t["code"].replace("\r\n", "\n").replace("\r", "\n")
        ep = _entry_point(t)
        sig = _signature(code, ep) if ep else None
        if not ep or not sig:
            dropped.append((t["task_id"], "no entry point/signature"))
            continue
        test = _build_test(t, ep)
        # self-validation: the reference solution must pass its harness
        ok, outp = run_oracle(code, test, ep)
        if not ok:
            dropped.append((t["task_id"], f"reference failed: {outp[:120]}"))
            continue
        out.append({
            "task_id": f"MBPP/{t['task_id']}",
            "prompt": f"{t['prompt']}\n\n{sig}",
            "entry_point": ep,
            "canonical_solution": code,
            "test": test,
        })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out)} tasks -> {OUT_PATH}", flush=True)
    if dropped:
        print(f"dropped {len(dropped)}:", flush=True)
        for tid, why in dropped:
            print(f"  MBPP/{tid}: {why}", flush=True)


if __name__ == "__main__":
    main()