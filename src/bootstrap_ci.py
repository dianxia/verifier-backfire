# -*- coding: utf-8 -*-
"""Task-level bootstrap CI for the pooled-rate leak-underprediction
ratio: observed conditional-on-delivery error divided by the
theory's prediction from pooled operating points. Per judge arm and
geometric mean across arms. Backs the paper's Section 4.3.

Requires results/e1_summary.json (run `python -m src.analyze`
first) for the fitted pooled rates; resamples the 164 tasks paired
across arms.

Usage: python -m src.bootstrap_ci [--path results/e1.jsonl]
  [--reps 10000] [--seed 20261016]
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARMS = ["judge_hetero", "judge_self_iso", "judge_self_ctx",
        "judge_deepseek_iso", "judge_glm_iso"]


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
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args()

    summary_path = args.path.replace(".jsonl", "_summary.json")
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    recs = [r for r in load(args.path) if "error" not in r]

    by_arm: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_arm[r["arm"]][r["task_id"]] = r
    _all_tasks = sorted(by_arm["none"].keys())
    tasks = sorted(set(_all_tasks).intersection(
        *[set(by_arm[a]) for a in ARMS]))
    if len(tasks) < len(_all_tasks):
        print(f"WARNING: {len(_all_tasks) - len(tasks)} tasks missing "
              "from some arms; excluded from the bootstrap")

    def pi_obs(arm: str, tids: list[str]) -> float:
        acc = lk = 0
        for t in tids:
            r = by_arm[arm][t]
            if r["delivered_correct"] is None:  # unresolved: excluded
                continue
            if r["delivered_correct"]:
                acc += 1
            else:
                lk += 1
        d = acc + lk
        return lk / d if d else float("nan")

    pi_pred = {}
    for arm in ARMS:
        s = summary[arm]
        p1, a, b = s["p1_hat"], s["a_hat"], s["b_hat"]
        pi_pred[arm] = (1 - p1) * (1 - b) / (p1 * a + (1 - p1) * (1 - b))

    random.seed(args.seed)
    per_arm: dict[str, list[float]] = {}
    gms = []
    skipped = 0
    for _ in range(args.reps):
        sample = [random.choice(tasks) for _ in tasks]
        vals = [pi_obs(arm, sample) / pi_pred[arm] for arm in ARMS]
        if any(v <= 0 for v in vals):
            skipped += 1  # zero-leak resample: log undefined
            continue
        for arm, v in zip(ARMS, vals):
            per_arm.setdefault(arm, []).append(v)
        gms.append(math.exp(sum(math.log(v) for v in vals) / len(ARMS)))
    if skipped:
        print(f"NOTE: {skipped}/{args.reps} resamples drew zero leaks for "
              "at least one arm; excluded from the geometric-mean CI")

    print("Pooled-rate underprediction ratio "
          "(pi_observed / pi_predicted):")
    for arm in ARMS:
        print(f"  {arm:<16} point = "
              f"{pi_obs(arm, tasks) / pi_pred[arm]:.2f}")
    gms.sort()
    k = len(gms)  # resamples surviving the zero-leak skip
    if k == 0:
        print("ERROR: all resamples skipped (zero leaks in every draw)")
        return
    ratios = [pi_obs(a, tasks) / pi_pred[a] for a in ARMS]
    gm0 = (math.exp(sum(math.log(v) for v in ratios) / len(ARMS))
           if all(v > 0 for v in ratios) else float("nan"))
    print(f"  geometric mean = {gm0:.2f}, "
          f"bootstrap 95% CI = [{gms[int(0.025 * k)]:.2f}, "
          f"{gms[int(0.975 * k)]:.2f}]")
    for arm in ARMS:
        v = sorted(per_arm[arm])
        print(f"  {arm:<16} boot95 = "
              f"[{v[int(0.025 * k)]:.2f}, {v[int(0.975 * k)]:.2f}]")


if __name__ == "__main__":
    main()