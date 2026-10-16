# -*- coding: utf-8 -*-
"""Paired significance tests over arm-level outcomes.

- McNemar exact tests: no-verification baseline vs.\ each judge arm,
  paired by task (unresolved counted as failure). Backs the paper's
  Section 4.2 significance claims.
- Fisher exact test: E3 feedback ablation, second-round repair
  successes of oracle_verdict vs.\ oracle_full. Backs Section 4.4.

Usage: python -m src.paired_tests [--path results/e1.jsonl]
"""
import argparse
import json
import math
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGE_ARMS = ["judge_self_iso", "judge_hetero", "judge_self_ctx"]
LOOP_ARMS = ["oracle_full", "oracle_verdict"] + JUDGE_ARMS


def load(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value (binomial sign test).

    b: discordant pairs baseline-right/arm-wrong;
    c: discordant pairs arm-right/baseline-wrong.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(p, 1.0)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value for table [[a, b], [c, d]]."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    pmf = {
        x: math.comb(row1, x) * math.comb(row2, col1 - x)
        / math.comb(n, col1)
        for x in range(lo, hi + 1)
    }
    return min(sum(v for v in pmf.values() if v <= pmf[a] * (1 + 1e-9)), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(
        _ROOT, "results", "e1.jsonl",
    ))
    args = ap.parse_args()
    recs = [r for r in load(args.path) if "error" not in r]

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_arm[r["arm"]][r["task_id"]] = r

    print("Within-arm pairing: terminal outcome vs the arm's OWN first")
    print("attempt (McNemar exact; the paper's primary comparison).")
    print("  lost = first correct, terminal not delivered-correct;")
    print("  rescued = first wrong, terminal correct;")
    print("  p: unresolved counted as failure; p_del: unresolved excluded.")
    for arm in LOOP_ARMS:
        lost = rescued = dl = dr = 0
        for r in by_arm[arm].values():
            f = r["attempts"][0]["gt_pass"]
            t = r["delivered_correct"]
            if f and not bool(t):
                lost += 1
            elif (not f) and bool(t):
                rescued += 1
            if t is not None:
                if f and t is False:
                    dl += 1
                elif (not f) and t:
                    dr += 1
        n_arm = len(by_arm[arm])
        p = mcnemar_exact(lost, rescued)
        pd = mcnemar_exact(dl, dr)
        print(f"  {arm:<16} lost={lost:>2} rescued={rescued:>2} "
              f"net={rescued - lost:+d}  eff={100*(rescued-lost)/n_arm:+.1f}pp"
              f"  p={p:.4f}  p_del={pd:.4f}")

    print("\nCross-arm check (confounded by independent first draws): "
          "none vs judge arm")
    for arm in JUDGE_ARMS:
        b = c = 0
        for tid, nr in by_arm["none"].items():
            ar = by_arm[arm][tid]
            n_ok = bool(nr["delivered_correct"])
            a_ok = bool(ar["delivered_correct"])
            if n_ok and not a_ok:
                b += 1
            elif (not n_ok) and a_ok:
                c += 1
        p = mcnemar_exact(b, c)
        print(f"  {arm:<16} none-only-right={b:>2}  arm-only-right={c:>2}"
              f"  exact p = {p:.4f}")

    sv = nv = sf = nf = 0
    for r in recs:
        if r["arm"] == "oracle_verdict" and len(r["attempts"]) >= 2:
            nv += 1
            sv += int(r["attempts"][1]["gt_pass"])
        if r["arm"] == "oracle_full" and len(r["attempts"]) >= 2:
            nf += 1
            sf += int(r["attempts"][1]["gt_pass"])
    p = fisher_exact(sv, nv - sv, sf, nf - sf)
    print(f"\nE3 feedback ablation (Fisher exact): "
          f"oracle_verdict {sv}/{nv} vs oracle_full {sf}/{nf}"
          f"  p = {p:.4f}")


if __name__ == "__main__":
    main()