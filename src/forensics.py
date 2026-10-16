# -*- coding: utf-8 -*-
"""Forensic attribution for the pooled-rate gap and the budget sweep.

- Leak composition: leaks split by first-attempt correctness---the
  conversion channel of Theorem 5 (falsely rejected correct first
  attempts) vs. the wrong-first-attempt channel.
- Degradation-only decomposition: loop models with constant (a, b),
  budget N=5, no instance-level clustering, under two retry-rate
  conventions (constant q = p2 vs. the measured per-round p_k); the
  attribution share is convention-dependent, which is the point.
- Overdispersion check: per-instance binomial Pearson chi2/df for
  misses among wrong attempts---directional evidence for clustering,
  not separable from degradation.
- Budget sweep: conditional-on-delivery error pi(N) by truncating
  every run at N rounds.

Requires results/e1_summary.json (run `python -m src.analyze` first).

Usage: python -m src.forensics [--path results/e1.jsonl]
"""
import argparse
import json
import os
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGE_ARMS = ["judge_hetero", "judge_self_iso", "judge_self_ctx"]


def load(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def pi_deg_model(p1: float, q: float, a: float, b: float, N: int = 5) -> float:
    """Conditional-on-delivery error of the degradation-only model:
    round 1 at rate p1, rounds 2..N at rate q, constant (a, b)."""
    s1, l1 = p1 * a, (1 - p1) * (1 - b)
    r1 = 1 - s1 - l1
    sq, lq = q * a, (1 - q) * (1 - b)
    rq = 1 - sq - lq
    good, bad, surv = s1, l1, r1
    for _ in range(2, N + 1):
        good += surv * sq
        bad += surv * lq
        surv *= rq
    return bad / (good + bad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(
        _ROOT, "results", "e1.jsonl",
    ))
    args = ap.parse_args()
    with open(os.path.join(_ROOT, "results", "e1_summary.json"),
              encoding="utf-8") as f:
        summary = json.load(f)
    recs = [r for r in load(args.path) if "error" not in r]

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_arm[r["arm"]][r["task_id"]] = r

    print("=== Leak composition (by first-attempt correctness) ===")
    for arm in JUDGE_ARMS:
        conv = wrong_first = 0
        for r in by_arm[arm].values():
            if r["leak"]:
                if r["attempts"][0]["gt_pass"]:
                    conv += 1      # Theorem-5 conversion channel
                else:
                    wrong_first += 1
        print(f"  {arm:<16} conversion={conv}  wrong-first={wrong_first}"
              f"  total={conv + wrong_first}")

    print("\n=== Degradation-only decomposition (constant (a,b), N=5) ===")
    print("  convention A: retries at constant q = p2 (per-round survival)")
    print("  convention B: retries at the measured per-round p_k")
    for arm in JUDGE_ARMS:
        s = summary[arm]
        p1, a, b = s["p1_hat"], s["a_hat"], s["b_hat"]
        q = next(d["p_k"] for d in s["p_k"] if d["k"] == 2)
        pks = {d["k"]: d["p_k"] for d in s["p_k"]}
        pi_pool = (1 - p1) * (1 - b) / (p1 * a + (1 - p1) * (1 - b))
        acc = sum(1 for r in by_arm[arm].values() if r["delivered_correct"])
        lk = sum(1 for r in by_arm[arm].values() if r["leak"])
        pi_obs = lk / (acc + lk)

        piA = pi_deg_model(p1, q, a, b)          # convention A
        good = bad = 0.0                          # convention B
        surv = 1.0
        for k in range(1, 6):
            p = pks.get(k, q)
            sk, lk_ = p * a, (1 - p) * (1 - b)
            good += surv * sk
            bad += surv * lk_
            surv *= 1 - sk - lk_
        piB = bad / (good + bad)

        fA = (piA - pi_pool) / (pi_obs - pi_pool)
        fB = (piB - pi_pool) / (pi_obs - pi_pool)
        print(f"  {arm:<16} pool={pi_pool:.4f}  "
              f"A(q=p2)={piA:.4f} ({100 * fA:.0f}%)  "
              f"B(p_k)={piB:.4f} ({100 * fB:.0f}%)  obs={pi_obs:.4f}")

    print("\n=== Overdispersion: per-instance misses among wrong attempts ===")
    print("  (tasks with >=1 wrong attempt; n_i wrong attempts, m_i of")
    print("   them accepted; binomial Pearson chi2 vs pooled miss rate)")
    for arm in JUDGE_ARMS:
        n_list, m_list = [], []
        for r in by_arm[arm].values():
            wrong = [att for att in r["attempts"] if not att["gt_pass"]]
            n = len(wrong)
            m = sum(1 for att in wrong if att["verdict"] == "accept")
            if n >= 1:
                n_list.append(n)
                m_list.append(m)
        N = sum(n_list)
        M = sum(m_list)
        pbar = M / N
        K = len(n_list)
        chi2 = sum((m - n * pbar) ** 2 / (n * pbar * (1 - pbar))
                   for n, m in zip(n_list, m_list))
        print(f"  {arm:<16} tasks={K}  wrong-attempts={N}  misses={M}"
              f"  miss-rate={pbar:.3f}  chi2/df={chi2 / (K - 1):.2f}")

    print("\n=== Budget sweep: pi(N) (runs truncated at N rounds) ===")
    for arm in JUDGE_ARMS:
        row = []
        for n_cap in range(1, 6):
            acc = lk = 0
            for r in by_arm[arm].values():
                atts = r["attempts"][:n_cap]
                if atts and atts[-1]["verdict"] == "accept":
                    if atts[-1]["gt_pass"]:
                        acc += 1
                    else:
                        lk += 1
            d = acc + lk
            row.append(lk / d if d else float("nan"))
        print(f"  {arm:<16} " + " ".join(f"{x:.3f}" for x in row))
    none_leak = sum(1 for r in by_arm["none"].values() if r["leak"])
    print(f"  none baseline    {none_leak / 164:.3f}")


if __name__ == "__main__":
    main()