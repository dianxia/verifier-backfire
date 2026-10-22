# verifier-backfire

Artifact for the paper *"When Verifiers Backfire: A Cost–Correctness Theory of
Generation–Verification–Repair Loops for LLMs"* (Chen, 2026).

Two benchmark studies — 164 × 9 on HumanEval and 247 × 9 on MBPP — showing when
test-time verification helps (oracle unit-test verification: leak → 0, never
loses a correct solution) and when it backfires: **same-family** LLM judges lose
correct work (isolated self-judge −15 on HumanEval, p=.0007; cross-capability
qwen-max judge −10 on both benchmarks), while neither **cross-family** judge
(deepseek-v4-pro, glm-5.2) shows a significant negative effect and glm-5.2
trends positive on MBPP (+8, nominal p=.039). The mechanism is not broken
repair — retries succeed at q=.47–.95 — but per-round **acceptance collapse**:
same-family judges accept correct work at .81–.86 on round 1 and near zero by
round 4, although the judges cannot see the round, and their rejections of
correct work cluster by task (the isolated self-judge rejects every correct
attempt on 19/32 multi-correct HumanEval tasks; the cross-family judges show
no clustering on either benchmark). This is instance-persistent
verifier–generator correlation, measured directly via silent oracle
instrumentation of every attempt; the finite-budget backfire condition
A₅ < p̂₁ signs all twelve judge–benchmark cells. A same-family
reasoning-judge control (qwen-plus with thinking enabled, the cross-family
verdict budget) dissociates the family/capability confound: the collapse
signature persists within the family (a_k .87→.45→.20 on HumanEval) while
the significant losses are confined to the non-reasoning same-family
judges (control net −3/−4, both n.s.).

## Repository layout

| Path | Content |
|---|---|
| ~~`paper/`~~ | Superseded pre-rerun paper (v1.0.0) — preserved in the Zenodo v1 archive ([DOI 10.5281/zenodo.23463247](https://doi.org/10.5281/zenodo.23463247)) and in git history; its headline numbers reflect the completion-truncation artifact described in the notes below |
| `paper_tmlr/` | Paper source (TMLR format, official tmlr.sty template) |
| `src/` | Experiment code: GVR loop, 9 verifier arms, sandbox oracle, paired-ablation driver, analysis |
| `results/e1.jsonl` | E1: all 1,476 HumanEval runs (9 arms) — per-attempt verdicts, ground truth, parse health, tokens |
| `results/e1_summary.json` | E1 aggregated per-arm stats |
| `results/e2_mbpp.jsonl` | E2: all 2,221 MBPP runs (9 arms; transient network-error records kept with their successful retries) |
| `results/e2_mbpp_summary.json` | E2 aggregated per-arm stats |
| `results/e3_paired_e1.jsonl` | Paired feedback ablation, HumanEval (8 seeds × 2 modes) |
| `results/e3_paired_e2.jsonl` | Paired feedback ablation, MBPP (64 seeds × 2 modes) |
| `results/e1_cap1024.jsonl` | 1,024-token-cap pilot (6 arms): retry-truncation evidence |
| `results/e2_mbpp_cap1024.jsonl` | 1,024-token-cap pilot, MBPP |
| `data/humaneval.jsonl` | Benchmark (HumanEval, 164 problems) |
| `data/mbpp.jsonl` | Benchmark (MBPP official test split, task IDs 11–510; 247 problems) |

## Reproduce

Requirements: Python ≥ 3.10, `httpx` (the only external dependency), a
[DashScope-compatible API key](https://dashscope.console.aliyun.com/).

```bash
# 1) install the single dependency
pip install httpx

# 2) put your key in a .env at the repo root (or export DASHSCOPE_API_KEY)
echo "DASHSCOPE_API_KEY=sk-..." > .env

# 3) run the full experiments (E1: 164 × 9 HumanEval, ~4.9M tokens;
#    E2: 247 × 9 MBPP, ~5.8M tokens)
python -m src.run_exp --arms all --limit 164 --threads 8
python -m src.run_exp --arms all --limit 247 --data data/mbpp.jsonl --out results/e2_mbpp.jsonl

# 4) reproduce the paper's tables
python -m src.analyze --path results/e1.jsonl
python -m src.analyze --path results/e2_mbpp.jsonl

# 5) significance tests, attribution, and bootstrap CIs (paper §4)
python -m src.paired_tests --path results/e1.jsonl    # within-arm McNemar (primary) + Fisher
python -m src.paired_tests --path results/e2_mbpp.jsonl
python -m src.forensics --path results/e1.jsonl        # leak composition, per-round a_k/b_k, verdict-format fallbacks, acceptance persistence, A_5 < p1 condition, pi(N) sweep
python -m src.forensics --path results/e2_mbpp.jsonl
python -m src.bootstrap_ci --path results/e1.jsonl     # pooled-rate underprediction CI
python -m src.bootstrap_ci --path results/e2_mbpp.jsonl
python -m src.pilot_truncation                         # reproduce the pilot's 36–68% round-2 truncation rates

# 6) paired feedback ablation (shared failing code, McNemar)
python -m src.paired_e3 --path results/e1.jsonl --data data/humaneval.jsonl --out results/e3_paired_e1.jsonl
python -m src.paired_e3 --path results/e2_mbpp.jsonl --data data/mbpp.jsonl --out results/e3_paired_e2.jsonl
```

Notes:

- The committed logs were produced with `qwen-plus` as the generator
  (temperature 1.0, completion cap 8192) and `qwen-max`/`qwen-plus`/
  `deepseek-v4-pro`/`glm-5.2` plus a qwen-plus reasoning control
  (`enable_thinking`, 4,096-token verdict budget) as judges, budget N=5,
  via a DashScope-compatible API (October 2026); bootstrap seed 20261016.
- The completion cap matters: the earlier pilot (`results/*_cap1024.jsonl`)
  ran with a 1,024-token cap that truncated 36–68% of round-2 completions by
  arm on HumanEval (30–45% on MBPP; first attempts 4–6% / 0–2%) into
  unparsable code, masquerading as repair degradation — reproduce with
  `python -m src.pilot_truncation`. The reported runs use 8,192 tokens;
  every attempt records a parse-health flag.
- The verdict budget matters too: the Qwen judges' 512-token verdict cap was
  never binding (zero format failures), while the reasoning judges
  (deepseek-v4-pro, glm-5.2; 4,096 tokens) occasionally exhausted their budget
  mid-thought, and the harness records those events as rejections (38/28 for
  glm, 4/7 for deepseek on HumanEval/MBPP) — an asymmetry that biases against
  the cross-family arms, which nonetheless dominate; the thinking control's
  4,096-token budget never bound either (zero fallbacks; two MBPP tasks lost
  to repeated API read timeouts, n=245/247); `src/forensics.py` counts
  these fallbacks and reports acceptance both with and without them.
- Checkpointing is built in — the driver skips `(arm, task)` pairs already
  in the output file, so an interrupted run simply resumes; failed jobs
  are retried on resume.
- Unit tests (the oracle arm) execute model-generated code in a local
  subprocess with a 6-second timeout; a timeout counts as a failed test.
  Run on a machine you are comfortable executing LLM-generated Python on
  (or wrap in a container).

## Key experiments and what they show

| Arm | Verifier | Expected finding |
|---|---|---|
| `none` | — | Baseline: deliver the first attempt (p̂₁ ≈ .95 HumanEval / .74 MBPP) |
| `oracle_full` | unit tests + traceback | Leak → 0, never loses a correct solution; +20 rescues on MBPP (p<10⁻⁴) |
| `oracle_verdict` | unit tests, verdict only | Same; paired-E3 seeds come from these arms' first attempts |
| `judge_hetero` | qwen-max judge, isolated | Backfires on both benchmarks (−10, p=.006); the one arm where the steady-state flip condition R_q < (1−a)/b holds |
| `judge_self_iso` | same-model judge, isolated | Worst arm on HumanEval (−15, p=.0007): acceptance collapse churns correct work to unresolved |
| `judge_self_ctx` | same-model judge, sees its own reasoning | No context-induced collapse; per-round acceptance still collapses |
| `judge_deepseek_iso` | deepseek-v4-pro judge (cross-family) | Near-oracle on HumanEval (acc .97); no significant negative effect |
| `judge_glm_iso` | glm-5.2 judge (cross-family) | Helps on MBPP (+8, nominal p=.039) |
| `judge_self_iso_thinking` | qwen-plus judge, reasoning mode (enable_thinking, 4,096) | Same-family reasoning control: collapse signature persists (a_k .87→.45→.20 HumanEval) yet net −3/−4, n.s. on both benchmarks |

## Citation

<!-- After cutting the v2 release: replace the doi below with the Zenodo *Concept DOI*
     (shown on the Zenodo record page) so citations always resolve to the latest version. -->
```bibtex
@misc{chen2026verifiers,
  title={When Verifiers Backfire: A Cost--Correctness Theory of
         Generation--Verification--Repair Loops for LLMs},
  author={Guidong Chen},
  year={2026},
  note={Preprint. Code and full logs: https://github.com/dianxia/verifier-backfire},
  doi={10.5281/zenodo.23463247}
}
```

## License

MIT — see [LICENSE](LICENSE). The HumanEval and MBPP benchmarks are used
under their original licenses.