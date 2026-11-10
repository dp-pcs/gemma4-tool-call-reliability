# Missing Edit Arguments and Misleading Test Status: Lessons from a Gemma 4 Coding Agent

**Subtitle:** We document rejected edit calls with missing arguments and shell pipelines that conceal test failure. The first mechanism remains unresolved; correcting the second did not improve public scores. Architecture and measurement lessons follow.

---

## Abstract

Building a Gemma 4–based agent for Kaggle's Gemma 4 Developer Agent competition, we found two tool-interface failure classes that aggregate resolution rates alone do not explain. The first: 710 of 2,515 `edit_file` calls (28.2%) arrived at the harness missing their `old_string` argument. We falsify the attractive "naive colon-splitter parser" explanation using the actual Apache-2.0 open-source vLLM Gemma 4 parser code (which uses explicit string delimiters, not colon parsing) plus 14,127 `run_command` controls with no unexpected or missing argument keys. The root cause remains open. The second: the agent's verification step pipes pytest output through `tail`, and bash reports `tail`'s exit code — not pytest's — so **failing tests can look like passes** to the agent; 85 of the agent's pytest calls across 65 of 285 runs were affected. The corrected prompt scored +9/114 locally (run-level p = 0.022; task-block p = 0.125), but did not improve scores on the public leaderboard (three draws: 9, 8, 8/58 versus 10, 10 for the uncorrected prompt). We also report three independent architecture and calibration lessons from the same project.

---

## 1. Introduction

Agentic coding systems depend on structured tool-calling interfaces both to act (edit files) and to verify (check whether tests pass). These interfaces fail in different ways: missing edit arguments trigger explicit tool errors, while shell pipelines can return a misleading success status. Aggregate resolution scores alone do not reveal either failure mode or how the agent responds.

Sections 3–4 investigate missing edit arguments and masked test status. The former remains unexplained; fixing the latter did not improve public scores. Sections 5–7 examine runtime calibration, shared-budget orchestration and self-distillation latency. Together, these results show why benchmark interpretation requires tool outcomes, termination reasons and deployment conditions alongside resolution counts.

---

## 2. Setting

The principal analyses use Gemma 4 31B IT (W4A16) via vLLM with the `gemma4` tool-call and reasoning parsers, built on Google's ADK via swegemma. Per-task budget: 28 tool calls and 4–8 minutes. Tools: `read_file`, `run_command`, `edit_file`, `write_file`, `submit_patch`.

Two corpora: Sections 3–4's bug analysis covers 7 configurations × up to 3 samples × 95 development tasks = 1,140 runs, on a 4×A100 rig at N=5 concurrent tasks per GPU. Sections 5–7 use 38 of the same 95 development tasks, 3 samples each (114 runs per arm); Section 5 compares N=5 with N=8 (decode-matched to the scorer's L4 rate) and its leaderboard comparisons use screens run at N=5 and N=2; Section 6 uses N=1 (single-task concurrency for the ensemble test); Section 7 uses N=5 (pre-calibration). The Section 4 pipefix screen ran at N=2 workers per GPU (8 total across 4 replicas). Neither corpus is the hidden scoring set; the competition publicly shows 58 of roughly 120 hidden tasks from repositories disjoint from the four public training ones.

---

## 3. An Unexplained Argument-Loss Failure in `edit_file`

### 3.1 The symptom

`edit_file` requires `old_string` (exact source text to match) and `new_string` (its replacement). Across 2,515 calls in the 1,140-run corpus:

| Category | Count | % of 2,515 |
|---|---|---|
| Both present | 1,777 | 70.7% |
| **`old_string` absent, `new_string` present** | **710** | **28.2%** |
| Both absent | 25 | 1.0% |
| `new_string` absent only | 3 | 0.1% |

We also found 602 back-to-back identical `edit_file`/`write_file` calls — a repetition signal that does not by itself establish whether the preceding call was rejected. Per-arm, the loss rate spans 16.1%–41.8%, with no single configuration knob explaining the spread.

### 3.2 Falsifying the naive hypothesis

Our first hypothesis was a parser-level defect: something treating colon characters inside argument values as key-value separators.

This is testable because the vLLM Gemma 4 tool-call parser is open-source (Apache-2.0, `vllm/tool_parsers/gemma4_tool_parser.py`). It uses a **custom wire format** in which string values are delimited by a reserved multi-character token sequence (`<|"|>...<|"|>`), not by colons. A correctly-emitted string value containing any number of colon characters should parse intact.

We extracted the three pure parsing functions — which depend on nothing beyond Python stdlib — and ran them directly in `scripts/paper/repro_colon_split.py`: colon-rich delimited strings parsed correctly in all test cases. The specific bare-value and unclosed-delimiter fixtures truncate values or add spurious keys while retaining `old_string`. They do not reproduce the observed missing-key symptom; they do not rule out other generation/parser interactions.

The control group independently confirms this: across all 14,127 `run_command` calls in the same corpus, 2,299 contained three or more colon characters (type hints, URLs, heredoc code) and **zero** showed any spurious or missing argument key. Uniform colon splitting would also affect these colon-dense `run_command` calls. We observe no such key corruption.

### 3.3 What the evidence says and does not say

A universal colon-split mechanism is contradicted by these controls. Generation, streaming parser behavior, and later schema/serialization handling remain possible contributors to the `old_string`-absent symptom. Our traces preserve only post-parse argument dicts, not raw model output, so we cannot reproduce the exact mechanism; we state it as an open, falsifiable question. The falsification of the colon-splitter hypothesis stands independently.

### 3.4 Mitigation and a note on the denominator

We routed most edits through `run_command` with a Python heredoc, avoiding `edit_file` for any multi-line or colon-containing match text. In the `T1_toolfix` arm, this policy produced only 63 `edit_file` calls across 41 of 285 runs; of those, 3 (4.8%) lost `old_string`. This denominator is a selected subset by construction (short, colon-free targets), not a random sample. The finding is that routing edits away from the interface avoids the failure class, not that `edit_file` became more reliable on the calls it still received.

---

## 4. Shell Pipelines Can Mask Failing Tests

### 4.1 The symptom

The agent's verification step instructs the agent to run tests as:
```
timeout 150 python -m pytest tests/path.py -x -q -k keyword 2>&1 | tail -25
```
In bash, the exit status of a pipeline is the exit status of its **last command** — `tail`, which almost always exits 0 regardless of what pytest reported. Across 1,140 runs in the same corpus, we found **85 pytest calls piped through `tail` across 65 of 285 `T1_toolfix` runs** (22.8% of runs had at least one affected call). No other arm showed this pattern, because only `T1_toolfix`'s prompt included this specific command template.

### 4.2 The mechanism

This is a scaffold-level bug in our own prompt, not a defect in pytest, bash, or the harness: we instructed the agent to run a command whose exit code it cannot trust. When pytest reports failing tests (exit code 1) or a collection error (exit code 2), the pipeline masks both as success. The agent, seeing no error exit code, may conclude its fix is working and proceed toward submission with a patch that fails the actual hidden tests.

### 4.3 The fix and what the data show

```
timeout 150 python -m pytest tests/path.py -x -q -k keyword > /tmp/t.log 2>&1; ec=$?; tail -25 /tmp/t.log; echo "EXIT:$ec"
```
The measured fix saves pytest’s status immediately, then displays it as `EXIT:1` on failure. It does not propagate that status: the final `echo` still exits zero. Automation consuming process status should also end with `exit "$ec"`; that additional change was not evaluated here.

A paired screen on familiar repositories (FastAPI, Rich) measured the local outcome: 38 development tasks × 3 samples = 114 runs per arm. The corrected arm resolved 44/114 (38.6%) vs. 35/114 (30.7%) for the original — W11/L2/T101, net +9, run-level sign-test p = 0.022. Keeping each task’s three repeats together gives task-block p = 0.125 (five task wins, one loss; `results/next_steps_20261106/pipefix_recovered_audit.json`). Explicit timeouts, including subsequently resolved patches, were 12 vs. 16 and empty patches decreased (24 vs. 27).

On the public leaderboard, the gain did not transfer. Three submissions of the corrected prompt scored 9, 8, and 8 of the 58 public tasks (mean 8.3), versus 10 and 10 for the uncorrected prompt (the same two draws used for T1@8). The local difference did not justify a public-task forecast; the observed public difference was −1.7 (mean). Recovered traces show policy adoption: direct pytest/unittest-to-tail/head pipelines fell from 30 to zero calls; 44 candidate calls captured status in a variable. This does not establish test relevance or a causal resolution gain. Selection noise, repository differences and runtime differences remain viable explanations for the public results. The mechanism for the transfer failure is an open question (see Section 5 for the broader context of this pattern).

---

## 5. Lesson A — Familiar-Repository Screens Can Mis-Rank Prompt Changes

At N=5 concurrent tasks per GPU (A100), one arm (`T3_5m`) resolved 37/114 (32.5%), mean task duration 284 s. The competition scorer's hardware (4×L4) has a measured single-stream decode rate of 34.3 tok/s, roughly 79% of our local rate. Recalibrating to N=8 workers per GPU, where per-stream decode (33.6 tok/s) matches the scorer's 34.3 tok/s, dropped the same arm to 28/114 (24.6%), mean 302 s, with 79/114 runs now hitting the 5-minute cap. An 8-percentage-point swing from changing one concurrency parameter, without changing the agent prompt.

`duration_seconds` includes setup, agent execution, and verification. Historical last-trace timestamps do not identify those phase boundaries, so subtracting their residual cannot establish agent-only runtime or a 12-hour margin. The corrected 1,140-run accounting retains 238 explicit timeouts (29 resolved), 33 turn-budget stops (9 resolved), 6 tool-budget stops, and 863 runs without an explicit termination marker. Unknown termination is not normal completion. T1@8 has two recorded 0.17 public scores; these establish completion of those draws, not a measured runtime margin. Matching decode rate alone does not establish scorer parity; setup, prefill contention, concurrency, and repository differences remain uncontrolled.

The deeper finding comes from the leaderboard data. Each time a prompt configuration won the local paired screen, it scored lower than the unchanged baseline on the public leaderboard:
- T3\_dynamic@8: net +7 locally (40 vs. 33/114), 0.13 vs. 0.17 on the LB
- T3\_5m: net +4 locally (37 vs. 33/114), 0.10 vs. 0.17
- t1\_pipefix: net +9 locally (44 vs. 35/114, task-block p = 0.125), mean 8.3 vs. 10/58

Each gap is small (2–4 tasks), and all three go the same way. For the two configurations compared here, T1 and pipefix, repeat submissions varied by at most one task; this is not a bound on variation for other configurations. Repository differences are one hypothesis: local screens used FastAPI and Rich, whereas the hidden tasks use disjoint repositories. Selection noise and runtime differences are alternatives; these observations do not isolate their contributions. A controlled unfamiliar-repository proxy could strengthen confirmation, but must first pass reference and negative controls.

---

## 6. Lesson B — A Multi-Agent Ensemble Can Be Disabled by One Runaway Sub-Agent

We built a parallel best-of-N ensemble (three independent solvers feeding a selector stage that calls `submit_patch`) to explore whether sampling diversity helps. Each solver was prompt-budgeted to 7 tool calls; the selector to 3. The screen ran at N=1 (single-task concurrency) to avoid the confound of GPU contention.

At single-task concurrency, the ensemble resolved 2/114 (1.8%) versus 29/114 (25.4%) for the single-agent baseline (T3\_5m\_tc1536) — W1/L27, sign-test p ≈ 2.2×10⁻⁷ (`results/p3_screen_20261023/gates_analysis_p3_vs_tc1536.json`). The cause was structural: ADK's `ParallelAgent` blocks until every sub-agent completes, and the harness enforces one session-wide budget (not per-sub-agent). In one inspected trace, one solver made ~20 tool calls against its 7-call prompt budget, running to 311 elapsed seconds while siblings finished around 120–125 s, exhausting time before the selector could run. The `locate` stage also overran its stated 2-call limit (using 3–4 calls), reflecting the same absence of enforced stage budgets. Mean tool calls per run rose from 17.0 (baseline) to 56.8; the primary failure shift was in empty patches (1 → 32) and wrong patches (6 → 0): the selector that holds `submit_patch` rarely ran, so most runs either submitted nothing or were stale partial captures.

A prompt-stated call limit is not an enforced limit. Multi-agent designs whose only commit path sits behind a stage that can be starved by a sibling need either framework-level per-sub-agent enforcement (unavailable here) or a non-blocking fallback path.

---

## 7. Lesson C — Self-Distillation Fine-Tuning Can Be Defeated by Inference Latency

We ran four LoRA self-distillation attempts on the agent's own resolved trajectories, each addressing a failure mode found in the prior attempt, against a pre-registered bar (promote if net wins ≥ +9; hold if ≤ −9, `results/harvest_20261015/GATES.md`), paired 38-task × 3-sample screen vs. untuned baseline:

| Variant | Key change | Result | Net (sign-test p) |
|---|---|---|---|
| v1 | Full thinking supervised, 90 examples, lr 1e-4 | 29/114 vs. 39/114 | −10 (p = 0.041) |
| v2 | Thinking masked from loss | 19/114 vs. 41/114 | −22 (p < 10⁻⁵) |
| v3 | Thinking supervised, truncated, lr 5e-5 | 31/114 vs. 35/114 | −4 (p = 0.52) |
| v4 | Efficiency-filtered training data | 32/114 vs. 33/114 | −1 (p = 1.0) |

None cleared the bar. The v1 result (net −10, p = 0.041, `results/harvest_20261015/GATES.md`) is based on 38 tasks × 3 samples; under a task-block permutation sign-flip, the same data yield p = 0.074 — not independently significant, so treat it as suggestive. v1 showed higher per-call token counts (305→370), longer per-call duration (14.8→17.6 s), and more empty-at-budget runs (38→55), all from the same source.

A confirming signal: v3's packaged submission (Kaggle ref 57412209) failed outright with "exceeded the allowed runtime" — a latency failure under the same budget the untuned baseline consistently passes. An adapter that improves per-call formatting can still lose a time-budgeted competition if it increases per-call latency. We observe this cost and treat latency as the most plausible cause, without claiming to have isolated it from other explanations.

---

## 8. Reproducibility

The paper's supporting code — detection scripts, aggregate evidence, the parser-testing repro script — is released in a public repository (`dp-pcs/gemma4-tool-call-reliability`) that contains no organizer-provided data. A companion Kaggle notebook (https://www.kaggle.com/code/latentgenius/gemma-4-agent-tool-interface-failures-evidence) reproduces the key evidence cells and links to that repository. The notebook contains no task statements, reference patches, repository snapshots or model traces.

---

## 9. Related Work

**SWE-bench.** Jimenez et al. (2024) established GitHub-issue resolution as an agent benchmark. Our paper is a case study in how much measured variance in such benchmarks can come from harness behavior, evaluation selection and runtime differences alongside model capability.

**Tool-calling reliability.** Gorilla (Patil et al., 2023) identified inaccurate argument generation as a core obstacle. The Berkeley Function-Calling Leaderboard systematically tracks call-level accuracy. Noorain et al. (2026) find that per-call error rates compound across multi-step tasks; our 28.2% argument-loss rate on a 31B model shows that argument-level failure is not limited to small models.

**Agent benchmarking methodology.** Zhang et al. (2026) document scaffold-attribution and shape-scoring confounds. Our three observed local/public rank reversals do not establish a distinct causal mechanism, but reinforce the need to confirm apparent local improvements.

**Multi-agent orchestration.** The session-wide-budget starvation we document in Section 6 is a concrete design constraint for this ADK-based submission architecture.

---

## 10. Limitations

Later eight-task pilots rejected T1@9 (3/8 versus 2/8; its only win finished within eight minutes) and E4B final-edit verification (0/8 in both arms). A subsequent frozen 38-task E4B scaffold screen on one L4 resolved 3/38 versus 6/38 (zero paired wins, three losses; p=0.25), failing its net ≥+3 gate. Candidate/baseline budgets differed: six/eight minutes and 50/28 calls. No transfer stage ran. These screens establish neither equivalence nor an isolated scaffold effect.

The principal analyses use Gemma 4 31B; later screens also cover E4B. All use swegemma/ADK and Python issue-resolution tasks. Section 3's mechanism investigation falsifies one hypothesis but leaves the actual cause open. Section 4's bug is from one prompt variant; other agents using a different verification command are not affected. The non-transfer finding in Section 5 rests on three public draws against a two-draw baseline. For T1 and pipefix, repeat submissions on the fixed 58-task public set varied by at most 1 task (T1@8: 10, 10; t1\_pipefix: 9, 8, 8); a binomial task-sampling calculation addresses hypothetical draws of tasks, not repeat-run variance or a measured private-set uncertainty. We have not established that the private set changes between runs. Section 6 is one ensemble design. Section 7's v1 p-value (0.041) does not clear a task-block permutation threshold (0.074), and all four variants share a training pipeline.

---

## 11. Conclusion

We documented missing edit arguments and misleading verification status. The first mechanism remains open; correcting the second changed agent commands but has not demonstrated a reliable scoring gain. The architecture and timing experiments show why termination, patch grading, runtime and task-level uncertainty must be reported separately. Familiar local screens provide development evidence; they do not by themselves establish generalization or identify the cause of public-score differences.

Code, evidence, and the executable parser tests are at https://github.com/dp-pcs/gemma4-tool-call-reliability.

---

## References

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*. arXiv:2310.06770.
- Noorain, A., Mohanty, S., Banerjee, A., & Dasgupta, A. (2026). Invocation-Level Reliability of Tool-Using Agents. arXiv:2608.26189.
- Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.
- Zhang, Y., Motaali, S., Dinh, V. P., Piroutiniya, A., et al. (2026). The Double Measurement Confound in Agent Benchmarks: De-Scaffolding, Ground-Truth Scoring, and Reliability Beyond the Mean. arXiv:2609.09218.
- Berkeley Function-Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html
- Google. (2026). Agent Development Kit (ADK). https://google.github.io/adk-docs/
