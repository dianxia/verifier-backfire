# -*- coding: utf-8 -*-
"""Aggregate results: per-arm operating points (p, a, b), terminal
outcomes, per-round conditional success (repair factor proxy), and
predicted-vs-actual comparison of the theory.

Usage: python -m src.analyze [--path results/e1.jsonl]
"""
import argparse
import json
import math
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, center - half, center + half)


def load(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(
        _ROOT, "results", "e1.jsonl",
    ))
    args = ap.parse_args()
    recs = [r for r in load(args.path) if "error" not in r]

    per_arm = defaultdict(list)
    for r in recs:
        per_arm[r["arm"]].append(r)

    print(
        f"{'arm':<16}{'n':>4} {'p1':>12} {'a_hat':>12} {'b_hat':>12} "
        f"{'acc':>12} {'leak':>10} {'unres':>10} {'E[k]':>6} "
        f"{'pi_pred':>8}{'pi_obs':>8} {'E_pred':>7}"
    )

    summary = {}
    for arm, rows in per_arm.items():
        n = len(rows)
        # first-attempt generator success rate
        p1_num = sum(
            1 for r in rows
            if r["attempts"] and r["attempts"][0]["gt_pass"]
        )
        # pooled operating point over ALL attempts (verdict vs gt)
        acc_n = sum(1 for r in rows if r["attempts"][0].get("verdict"))
        correct_pool = [
            a["verdict"] for r in rows for a in r["attempts"]
            if a["gt_pass"]
        ]
        wrong_pool = [
            a["verdict"] for r in rows for a in r["attempts"]
            if not a["gt_pass"]
        ]
        n_acc_correct = sum(
            1 for v in correct_pool if v == "accept"
        )
        n_rej_wrong = sum(1 for v in wrong_pool if v == "reject")

        acc_delivered = sum(
            1 for r in rows if r["delivered_correct"]
        )
        leak = sum(1 for r in rows if r["leak"])
        unres = sum(1 for r in rows if r["unresolved"])
        ek = sum(r["n_attempts"] for r in rows) / n

        p1, p1lo, p1hi = wilson(p1_num, n)
        if arm == "none":
            # No verifier runs in the baseline arm: acceptance and
            # catch rates are undefined, not 1.0/0.0.
            a = alo = ahi = None
            b = blo = bhi = None
        else:
            if correct_pool:
                a, alo, ahi = wilson(n_acc_correct, len(correct_pool))
            else:
                a = alo = ahi = float("nan")
            if wrong_pool:
                b, blo, bhi = wilson(n_rej_wrong, len(wrong_pool))
            else:
                b = blo = bhi = float("nan")
        acc, acclo, acchi = wilson(acc_delivered, n)
        lk = leak / n
        un = unres / n

        # theory (only meaningful for looped arms with judges):
        pi_pred = pi_obs = float("nan")
        delivered = acc_delivered + leak
        if (
            a is not None and b is not None
            and correct_pool and wrong_pool and n > 1
        ):
            s = p1 * a
            ell = (1 - p1) * (1 - b)
            pi_pred = ell / (s + ell) if (s + ell) > 0 else float("nan")
        if delivered > 0:
            # pi_obs = P(wrong | delivered) = leak / delivered,
            # defined for every arm including the baseline.
            pi_obs = leak / delivered

        a_str = ("     --           "
                 if a is None else f"{a:>5.2f}[{alo:.2f},{ahi:.2f}]")
        b_str = ("     --           "
                 if b is None else f"{b:>5.2f}[{blo:.2f},{bhi:.2f}]")
        print(
            f"{arm:<16}{n:>4} "
            f"{p1:>5.2f}[{p1lo:.2f},{p1hi:.2f}] "
            f"{a_str} {b_str} "
            f"{acc:>5.2f}[{acclo:.2f},{acchi:.2f}] "
            f"{lk:>10.3f} {un:>10.3f} {ek:>6.2f} "
            f"{pi_pred:>8.3f}{pi_obs:>8.3f}"
        )

        # per-round conditional success p_k (repair factor proxy)
        pks = []
        for k in range(1, 6):
            denom = sum(
                1 for r in rows if len(r["attempts"]) >= k
            )
            if denom >= 10:
                num = sum(
                    1 for r in rows
                    if len(r["attempts"]) >= k
                    and r["attempts"][k - 1]["gt_pass"]
                )
                pks.append((k, num, denom))

        summary[arm] = {
            "n": n,
            "p1_hat": p1, "p1_ci": [p1lo, p1hi],
            "a_hat": a, "a_ci": [alo, ahi],
            "b_hat": b, "b_ci": [blo, bhi],
            "terminal_accuracy": acc, "acc_ci": [acclo, acchi],
            "leak_rate": lk, "unresolved_rate": un,
            "E_attempts": ek,
            "pi_predicted": pi_pred, "pi_observed_cond": pi_obs,
            "p_k": [
                {"k": k, "p_k": num / den, "n": den}
                for k, num, den in pks
            ],
            "n_attempts_pooled": len(correct_pool) + len(wrong_pool),
            "n_correct_pool": len(correct_pool),
            "n_wrong_pool": len(wrong_pool),
        }

    out_path = args.path.replace(".jsonl", "_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsummary -> {out_path}")

    # per-arm p_k detail (repair dynamics, E3)
    print("\nper-round conditional success p_k (attempts reached, "
          "n>=10):")
    for arm, s in summary.items():
        if s["p_k"]:
            line = " ".join(
                f"k{d['k']}:{d['p_k']:.2f}(n={d['n']})"
                for d in s["p_k"]
            )
            print(f"  {arm:<16} {line}")


if __name__ == "__main__":
    main()