# verifier-backfire

Artifact for the paper *"When Verifiers Backfire: A Cost–Correctness Theory of
Generation–Verification–Repair Loops for LLMs"* (Chen, 2026).

A 164 × 6 study on HumanEval showing when test-time verification helps
(oracle unit-test verification: +7pp accuracy, leak 9.1% → 0.0%, at +27%
attempts) and when it backfires (LLM-judge verification with pooled operating
points of a≈0.78, b≈0.75–0.86 lands *below* the no-verification baseline),
together with the theory that predicts both outcomes from measured parameters
alone — including the repair-degradation flip threshold q\*.

## Repository layout

| Path | Content |
|---|---|
| `paper/main.tex` | Full paper source (self-contained LaTeX) |
| `src/` | Experiment code: GVR loop, 6 verifier arms, sandbox oracle, driver, analysis |
| `results/e1.jsonl` | All 984 raw runs — per-attempt verdicts, ground truth, feedback text, token counts |
| `results/e1_summary.json` | Aggregated table (the paper's Table 1/2 numbers) |
| `data/humaneval.jsonl` | Benchmark (HumanEval, 164 problems) |

## Reproduce

Requirements: Python ≥ 3.10, `httpx` (the only external dependency), a
[DashScope API key](https://dashscope.console.aliyun.com/).

```bash
# 1) install the single dependency
pip install httpx

# 2) put your key in a .env at the repo root (or export DASHSCOPE_API_KEY)
echo "DASHSCOPE_API_KEY=sk-..." > .env

# 3) run the full experiment (164 problems × 6 arms, ~1.6M tokens total)
python -m src.run_exp --arms all --limit 164 --threads 8

# 4) reproduce the paper's tables
python -m src.analyze

# 5) significance tests, attribution, and bootstrap CIs (paper §4)
python -m src.paired_tests   # within-arm McNemar (primary) + cross-arm + Fisher (E3)
python -m src.forensics      # leak composition, degradation attribution, pi(N) budget sweep
python -m src.bootstrap_ci   # task-level bootstrap CI for the ≈3.4× ratio
```

Notes:

- Checkpointing is built in — the driver skips `(arm, task)` pairs already in
  `results/e1.jsonl`, so an interrupted run simply resumes.
- Unit tests (the oracle arm) execute model-generated code in a local
  subprocess with a 6-second timeout; a timeout counts as a failed test
  (rare on HumanEval — none of the 984 committed runs timed out). Run on a
  machine you are comfortable executing LLM-generated Python on (or wrap
  in a container).
- The committed `results/e1.jsonl` was produced with `qwen-plus` as the
  generator (temperature 1.0) and `qwen-max`/`qwen-plus` as judges, budget
  N=5, in October 2026.

## Key experiments and what they show

| Arm | Verifier | Expected finding |
|---|---|---|
| `none` | — | Baseline: deliver the first attempt (p̂₁ ≈ 0.91) |
| `oracle_full` | unit tests + traceback feedback | Accuracy 0.98, leak 0 — the theory's clean case (b=1) |
| `oracle_verdict` | unit tests, verdict only | E3 ablation: repair success 0.19 vs 0.33 — feedback actionability (δ > 0) |
| `judge_hetero` | qwen-max judge, isolated context | Backfires below baseline; conversion rate matches the flip-theorem prediction |
| `judge_self_iso` | same-model judge, isolated context | Worst arm (0.82): false rejections + degraded retries (q≈0.27 < q\*≈0.75) |
| `judge_self_ctx` | same-model judge, sees its own reasoning | Honest negative result: no verifier-context collapse (b̂ 0.86 vs 0.74) |

## Citation

```bibtex
@misc{chen2026verifiers,
  title={When Verifiers Backfire: A Cost--Correctness Theory of
         Generation--Verification--Repair Loops for LLMs},
  author={Guidong Chen},
  year={2026},
  note={arXiv preprint (to appear)}
}
```

## License

MIT — see [LICENSE](LICENSE). The HumanEval benchmark is used under its
original license (Chen et al., 2021).