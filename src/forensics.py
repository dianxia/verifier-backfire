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
- Verdict-format fallbacks: judge-side budget exhaustions that the
  harness records as rejections, by arm and round, with the
  operating point recomputed among parsed verdicts.
- Acceptance persistence: per-task clustering of rejections of
  correct work among parsed verdicts (chi2/df, all-rejected share).
- Terminal backfire condition in both forms: the steady-state
  sufficient condition R_q < (1-a)/b and the exact finite-budget
  form A_5 < p1, with a task-level bootstrap CI for A_5 - p1.
- Acceptance-persistence permutation test: conditional null that
  shuffles verdicts across each arm's correct attempts (per-task
  counts preserved), with empirical p for the chi2/df and
  all-rejected statistics.
- Per-task compute cost: mean generator/judge tokens by arm
  (the paper's Appendix C).

Requires results/e1_summary.json (run `python -m src.analyze` first).

Usage: python -m src.forensics [--path results/e1.jsonl]
"""
import argparse
import json
import os
import random
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JUDGE_ARMS = ["judge_hetero", "judge_self_iso", "judge_self_ctx",
              "judge_deepseek_iso", "judge_glm_iso",
              "judge_self_iso_thinking"]

# The harness's fallback feedback for unparseable judge responses
# (verdict-budget exhaustion); attempts carrying this marker are
# harness rejections, not the judge's verdicts.
FB_MARKER = "could not be parsed"


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


def _clustering_stats(n_list: list[int], m_list: list[int]):
    """(chi2/df, all-rejected count, multi-correct task count) or None."""
    N, M, K = sum(n_list), sum(m_list), len(n_list)
    if K < 3 or M == 0 or M == N:
        return None
    pbar = M / N
    chi2 = sum((m - n * pbar) ** 2 / (n * pbar * (1 - pbar))
               for n, m in zip(n_list, m_list))
    multi = [(n, m) for n, m in zip(n_list, m_list) if n >= 2]
    allrej = sum(1 for n, m in multi if m == n)
    return chi2 / (K - 1), allrej, len(multi)


def _arm_params(recs_arm: dict) -> tuple | None:
    """Task-level (p1, a, b, q=p2) from an arm's records."""
    tasks = list(recs_arm.values())
    if not tasks:
        return None
    n = len(tasks)
    p1 = sum(1 for t in tasks if t["attempts"][0]["gt_pass"]) / n
    corr = [x for t in tasks for x in t["attempts"] if x["gt_pass"]]
    wrong = [x for t in tasks for x in t["attempts"] if not x["gt_pass"]]
    if not corr or not wrong:
        return None
    a = sum(1 for x in corr if x["verdict"] == "accept") / len(corr)
    b = sum(1 for x in wrong if x["verdict"] == "reject") / len(wrong)
    r2 = [t for t in tasks if len(t["attempts"]) >= 2]
    if not r2:
        return None
    q = sum(1 for t in r2 if t["attempts"][1]["gt_pass"]) / len(r2)
    return p1, a, b, q


def _a5_gap(p1: float, a: float, b: float, q: float, N: int = 5) -> float:
    """A_N - p1: the exact finite-budget backfire statistic."""
    s1, l1 = p1 * a, (1 - p1) * (1 - b)
    r1 = 1 - s1 - l1
    sq, lq = q * a, (1 - q) * (1 - b)
    rq = 1 - sq - lq
    good, surv = s1, r1
    for _ in range(2, N + 1):
        good += surv * sq
        surv *= rq
    return good - p1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.path.join(
        _ROOT, "results", "e1.jsonl",
    ))
    args = ap.parse_args()
    with open(args.path.replace(".jsonl", "_summary.json"),
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

    print("\n=== Pure-pool q sensitivity (correctly-rejected first attempts) ===")
    print("  q_correct: attempt-2 success among first attempts that were")
    print("  correct but falsely rejected -- the Theorem-5 sub-loop's own")
    print("  pool, vs. the mixed pool used in the paper (q = p2).")
    for arm in JUDGE_ARMS:
        s = summary[arm]
        p1, a, b = s["p1_hat"], s["a_hat"], s["b_hat"]
        pool = [
            r for r in by_arm[arm].values()
            if r["attempts"][0]["gt_pass"]
            and r["attempts"][0]["verdict"] == "reject"
            and len(r["attempts"]) >= 2
        ]
        n = len(pool)
        succ = sum(1 for r in pool if r["attempts"][1]["gt_pass"])
        q_c = succ / n if n else float("nan")
        qstar = (1 - b) * p1 / ((1 - p1) * a + (1 - b) * p1)
        piq = (1 - q_c) * (1 - b) / (q_c * a + (1 - q_c) * (1 - b))
        pred = p1 * (1 - a) * piq * s["n"]
        conv = sum(
            1 for r in by_arm[arm].values()
            if r["leak"] and r["attempts"][0]["gt_pass"]
        )
        print(f"  {arm:<16} q_correct={q_c:.2f} (n={n})  q*={qstar:.2f}  "
              f"below={q_c < qstar}  pred_conversions={pred:.1f}  "
              f"observed={conv}")

    print("\n=== Per-round verifier rates (a_k = accept|correct, "
          "b_k = reject|wrong; cells n>=5) ===")
    for arm in JUDGE_ARMS:
        st = defaultdict(lambda: [0, 0, 0, 0])
        for r in by_arm[arm].values():
            for a_ in r["attempts"]:
                d = st[a_["k"]]
                if a_["gt_pass"]:
                    d[1] += 1
                    d[0] += 1 if a_["verdict"] == "accept" else 0
                else:
                    d[3] += 1
                    d[2] += 1 if a_["verdict"] == "reject" else 0
        cells = []
        for k in sorted(st):
            acc, nc, rej, nw = st[k]
            s = f"k{k}:"
            if nc >= 5:
                s += f" a={acc/nc:.2f}({nc})"
            if nw >= 5:
                s += f" b={rej/nw:.2f}({nw})"
            if s != f"k{k}:":
                cells.append(s)
        print(f"  {arm:<16} " + "  ".join(cells))

    print("\n=== Verdict-format fallbacks (harness reject-on-unparseable) ===")
    print("  Judge-side verdict-budget exhaustions recorded as")
    print("  rejections by the harness; a_excl_fb = acceptance of")
    print("  correct work among parsed verdicts only.")
    for arm in JUDGE_ARMS:
        fb_c = fb_w = 0
        per_round = defaultdict(lambda: [0, 0])   # k -> [correct, wrong]
        parsed_ok = parsed_acc = 0
        for r in by_arm[arm].values():
            for att in r["attempts"]:
                fb = FB_MARKER in (att.get("feedback") or "")
                if fb and att["gt_pass"]:
                    fb_c += 1
                    per_round[att["k"]][0] += 1
                elif fb:
                    fb_w += 1
                    per_round[att["k"]][1] += 1
                if att["gt_pass"] and not fb:
                    parsed_ok += 1
                    parsed_acc += 1 if att["verdict"] == "accept" else 0
        a_excl = parsed_acc / parsed_ok if parsed_ok else float("nan")
        detail = " ".join(f"k{k}:{c}c/{w}w"
                          for k, (c, w) in sorted(per_round.items()))
        print(f"  {arm:<16} fallbacks={fb_c}c/{fb_w}w  "
              f"a_incl={summary[arm]['a_hat']:.2f}  a_excl_fb={a_excl:.2f}"
              f"  [{detail}]")

    print("\n=== Per-round a_k among parsed verdicts (fallbacks excluded) ===")
    print("  (acceptance of correct attempts, cells n>=5; the paper's")
    print("   tab:ak view: same-family arms have zero fallbacks, so only")
    print("   the cross-family cells differ from the loop view above)")
    for arm in JUDGE_ARMS:
        st = defaultdict(lambda: [0, 0])
        for r in by_arm[arm].values():
            for a_ in r["attempts"]:
                if (a_["gt_pass"]
                        and FB_MARKER not in (a_.get("feedback") or "")):
                    st[a_["k"]][1] += 1
                    st[a_["k"]][0] += 1 if a_["verdict"] == "accept" else 0
        cells = []
        for k in sorted(st):
            acc, n = st[k]
            if n >= 5:
                cells.append(f"k{k}: {acc / n:.2f}({n})")
        print(f"  {arm:<16} " + "  ".join(cells))

    print("\n=== Acceptance persistence: correct attempts, parsed verdicts ===")
    print("  Per-task clustering of rejections of correct work:")
    print("  tasks with >=1 parsed correct attempt; n_i correct attempts,")
    print("  m_i rejected; chi2/df vs pooled binomial; all-rejected share")
    print("  among tasks with >=2 correct attempts; conc = rejections")
    print("  falling on multi-rejection tasks.")
    for arm in JUDGE_ARMS:
        n_list, m_list = [], []
        for r in by_arm[arm].values():
            corr = [a_ for a_ in r["attempts"]
                    if a_["gt_pass"]
                    and FB_MARKER not in (a_.get("feedback") or "")]
            if not corr:
                continue
            n_list.append(len(corr))
            m_list.append(sum(1 for a_ in corr if a_["verdict"] == "reject"))
        N, M, K = sum(n_list), sum(m_list), len(n_list)
        if K < 3 or M == 0:
            print(f"  {arm:<16} tasks={K} rejections={M} (degenerate)")
            continue
        pbar = M / N
        chi2 = sum((m - n * pbar) ** 2 / (n * pbar * (1 - pbar))
                   for n, m in zip(n_list, m_list))
        multi = [(n, m) for n, m in zip(n_list, m_list) if n >= 2]
        allrej = sum(1 for n, m in multi if m == n)
        rej_multi = sum(m for n, m in zip(n_list, m_list) if m >= 2)
        print(f"  {arm:<16} tasks={K} corr={N} rej={M} rate={pbar:.2f}  "
              f"chi2/df={chi2 / (K - 1):.2f}  all-rej={allrej}/{len(multi)}  "
              f"conc={rej_multi}/{M}")

    print("\n=== Persistence: conditional permutation test (10k) ===")
    print("  Verdicts shuffled across each arm's parsed correct attempts")
    print("  (per-task counts preserved); one-sided empirical p.")
    rng = random.Random(20261022)
    for arm in JUDGE_ARMS:
        per_task = []
        for r in by_arm[arm].values():
            corr = [a_ for a_ in r["attempts"]
                    if a_["gt_pass"]
                    and FB_MARKER not in (a_.get("feedback") or "")]
            if corr:
                per_task.append(
                    [1 if a_["verdict"] == "reject" else 0 for a_ in corr])
        n_list = [len(x) for x in per_task]
        obs = _clustering_stats(n_list, [sum(x) for x in per_task])
        if obs is None:
            print(f"  {arm:<16} degenerate")
            continue
        pool = [x for lst in per_task for x in lst]
        c_chi = c_ar = 0
        REPS = 10000
        for _ in range(REPS):
            rng.shuffle(pool)
            idx = 0
            sim = []
            for n_ in n_list:
                sim.append(sum(pool[idx:idx + n_]))
                idx += n_
            st = _clustering_stats(n_list, sim)
            if st is None:
                continue
            if st[0] >= obs[0]:
                c_chi += 1
            if st[1] >= obs[1]:
                c_ar += 1
        print(f"  {arm:<16} chi2/df={obs[0]:.2f}  p={c_chi / REPS:.4f}  "
              f"all-rej={obs[1]}/{obs[2]}  p={c_ar / REPS:.4f}")

    print("\n=== Terminal backfire condition (q = p2) ===")
    print("  steady state: R_q < (1-a)/b (sufficient-only)")
    print("  finite N=5:   A_5 < p1 (the paper's primary prediction)")
    for arm in JUDGE_ARMS:
        s = summary[arm]
        p1, a, b = s["p1_hat"], s["a_hat"], s["b_hat"]
        q = next(d["p_k"] for d in s["p_k"] if d["k"] == 2)
        Rq = (a / (1 - b)) * (q * (1 - p1)) / (p1 * (1 - q))
        thr = (1 - a) / b
        s1, l1 = p1 * a, (1 - p1) * (1 - b)
        r1 = 1 - s1 - l1
        sq, lq = q * a, (1 - q) * (1 - b)
        rq = 1 - sq - lq
        good, surv = s1, r1
        for _ in range(2, 6):
            good += surv * sq
            surv *= rq
        A5 = good
        lost = sum(
            1 for r in by_arm[arm].values()
            if r["attempts"][0]["gt_pass"]
            and not bool(r["delivered_correct"]))
        rescued = sum(
            1 for r in by_arm[arm].values()
            if not r["attempts"][0]["gt_pass"]
            and bool(r["delivered_correct"]))
        print(f"  {arm:<16} q={q:.2f}  A5-p1={A5 - p1:+.4f}  "
              f"pred_A5={A5 < p1}  Rq={Rq:.2f}  (1-a)/b={thr:.2f}  "
              f"pred_ss={Rq < thr}  net={rescued - lost:+d}")

    print("\n=== A_5 - p1: task-level bootstrap CI (10k resamples, seed 20261022) ===")
    rng = random.Random(20261022)
    for arm in JUDGE_ARMS:
        tasks = list(by_arm[arm].values())
        pr = _arm_params(by_arm[arm])
        if pr is None:
            print(f"  {arm:<16} degenerate")
            continue
        obs = _a5_gap(*pr)
        vals = []
        for _ in range(10000):
            sample = [rng.choice(tasks) for _ in tasks]
            rs = {t["task_id"]: t for t in sample}
            pr2 = _arm_params(rs)
            if pr2 is None:
                continue
            vals.append(_a5_gap(*pr2))
        vals.sort()
        if not vals:
            print(f"  {arm:<16} degenerate")
            continue
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals))]
        pneg = sum(1 for v in vals if v < 0) / len(vals)
        print(f"  {arm:<16} A5-p1={obs:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  "
              f"P(boot<0)={pneg:.3f}")

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
    print(f"  none baseline    "
          f"{none_leak / len(by_arm['none']):.3f}")

    print("\n=== Per-task compute cost (mean tokens by arm) ===")
    print(f"  {'arm':<24} {'E[k]':>5} {'gen':>6} {'judge':>7} {'total':>7}")
    for arm in sorted(by_arm.keys()):
        rs = list(by_arm[arm].values())
        if not rs:
            continue
        n = len(rs)
        gen = sum(sum(a_.get("gen_tokens", 0) for a_ in r["attempts"])
                  for r in rs) / n
        jud = sum(sum(a_.get("judge_tokens", 0) for a_ in r["attempts"])
                  for r in rs) / n
        ek = sum(r["n_attempts"] for r in rs) / n
        print(f"  {arm:<24} {ek:5.2f} {gen:6.0f} {jud:7.0f} {gen + jud:7.0f}")

    print("\n=== C3 fallback-delivery convention (deliver last attempt ===")
    print("=== when the budget exhausts) ===")
    print(f"  {'arm':<24} {'C1acc':>6} {'C3acc':>6} {'C1eff%':>7} "
          f"{'C3eff%':>7}  unres(last-correct)")
    for arm in sorted(by_arm.keys()):
        if arm == "none":
            continue
        rs = list(by_arm[arm].values())
        n = len(rs)
        if not n:
            continue
        c1 = sum(1 for r in rs if r["delivered_correct"])
        unres = [r for r in rs if not r["resolved"]]
        lc = sum(1 for r in unres if r["attempts"][-1]["gt_pass"])
        lost1 = sum(1 for r in rs if r["attempts"][0]["gt_pass"]
                    and not bool(r["delivered_correct"]))
        resc1 = sum(1 for r in rs if not r["attempts"][0]["gt_pass"]
                    and bool(r["delivered_correct"]))

        def c3ok(r: dict) -> bool:
            return (bool(r["delivered_correct"])
                    or (not r["resolved"]
                        and r["attempts"][-1]["gt_pass"]))

        lost3 = sum(1 for r in rs
                    if r["attempts"][0]["gt_pass"] and not c3ok(r))
        resc3 = sum(1 for r in rs
                    if not r["attempts"][0]["gt_pass"] and c3ok(r))
        print(f"  {arm:<24} {c1 / n:6.3f} {(c1 + lc) / n:6.3f} "
              f"{(resc1 - lost1) / n * 100:+6.1f} "
              f"{(resc3 - lost3) / n * 100:+6.1f}  {lc}/{len(unres)}")


if __name__ == "__main__":
    main()