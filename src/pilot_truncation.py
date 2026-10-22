# -*- coding: utf-8 -*-
"""Reproduce the pilot truncation rates reported in the paper.

The 1,024-token completion cap of the pilot run truncated a large
fraction of retry-round completions into unparsable code, which
masqueraded as severe repair degradation.  This script recomputes
the exact per-arm rates from the archived pilot logs
(results/*_cap1024.jsonl): the ast.parse failure rate of the stored
`code` field by attempt index k (the pilot logs predate the
parse_ok flag).

Paper claims reproduced (Setup, instrumentation note):
  HumanEval: 36--68% of round-2 completions by arm
             (first attempts: 4--6%)
  MBPP:      30--45% of round-2 completions by arm
             (first attempts: 0--2%)

Usage: python -m src.pilot_truncation
"""
import ast
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PILOTS = [
    ("HumanEval", os.path.join(_ROOT, "results", "e1_cap1024.jsonl")),
    ("MBPP", os.path.join(_ROOT, "results", "e2_mbpp_cap1024.jsonl")),
]


def unparsable(code: str) -> bool:
    try:
        ast.parse(code)
        return False
    except (SyntaxError, ValueError):
        return True


def main() -> None:
    for name, path in PILOTS:
        by_arm_k: dict[tuple[str, int], list[int]] = defaultdict(
            lambda: [0, 0])   # (arm, k) -> [n, bad]
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in r:
                    continue
                for att in r["attempts"]:
                    d = by_arm_k[(r["arm"], att["k"])]
                    d[0] += 1
                    d[1] += unparsable(att.get("code") or "")
        arms = sorted({a for a, _ in by_arm_k})
        print(f"=== {name}: ast.parse failure rate of stored code ===")
        for arm in arms:
            cells = []
            for k in (1, 2):
                n, bad = by_arm_k[(arm, k)]
                if n:
                    cells.append(f"k{k}: {100 * bad / n:.0f}% ({bad}/{n})")
            print(f"  {arm:<16} " + "   ".join(cells))
        for k in (1, 2):
            rates = [by_arm_k[(a, k)][1] / by_arm_k[(a, k)][0]
                     for a in arms if by_arm_k[(a, k)][0]]
            if rates:
                print(f"  round-{k} by-arm range: "
                      f"{100 * min(rates):.0f}--{100 * max(rates):.0f}%")
        print()


if __name__ == "__main__":
    main()