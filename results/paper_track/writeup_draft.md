# When Tool-Interfaces Lie: Two Silent Failures and Three Measurement Lessons from a Gemma 4 Coding Agent

**Subtitle:** We found two independent tool-interface bugs that report success when the agent has failed. We ruled out an attractive explanation for the first, and provide a complete executable fix for the second. Three infrastructure lessons follow.

---

## Abstract

Building a Gemma 4–based agent for the Kaggle Gemma 4 Developer Agent competition, we found two independent, silent tool-interface failures — failures that give the agent no actionable error signal and are invisible in aggregate resolution-rate metrics. The first: 710 of 2,515 `edit_file` calls (28.2%) arrived at the harness missing their `old_string` argument. We falsify the attractive "naive colon-splitter parser" explanation using the actual Apache-2.0 open-source vLLM Gemma 4 parser code (which uses explicit string delimiters, not colon parsing) plus a clean control group of 14,127 `run_command` calls (colon-dense, zero corruption). The root cause remains open. The second: the agent's verification step pipes pytest output through `tail`, and bash reports `tail`'s exit code — not pytest's — so **failing tests can look like passes** to the agent; 85 of the agent's pytest calls across 65 of 285 runs were affected. A single command change (redirect + capture `$?`) is the complete fix. We also report three independent calibration and methodology lessons from the same project.

---

## 1. Introduction

Agentic coding systems depend on structured tool-calling interfaces both to act (edit files) and to verify (check whether tests pass). Failures in those interfaces can be silent: the agent receives no error, assumes success, submits a patch, and the downstream scorer sees the actual failure. From the outside, the run looks identical to a reasoning failure.

This paper documents two such failures found while building an agent for the Google–Gemma 4 Developer Agent competition (Kaggle). Section 3 covers a measured, large-scale argument-loss failure whose mechanism we cannot fully explain despite a careful investigation that falsified one hypothesis with executable code. Section 4 covers an exit-code masking bug whose mechanism is precise and whose fix is a one-line command change.

Sections 5–7 report three further, independent lessons from the same project: benchmark-rig calibration is harder than matching one throughput number; shared-budget multi-agent frameworks can make concurrency actively harmful; and self-distillation fine-tuning can be systematically defeated by inference-latency cost under a wall-clock budget.

---

## 2. Setting

Agents use Gemma 4 31B IT (W4A16) via vLLM with the `gemma4` tool-call and reasoning parsers, built on Google's ADK via swegemma. Per-task budget: 28 tool calls and 4–8 minutes. Tools: `read_file`, `run_command`, `edit_file`, `write_file`, `submit_patch`.

Two corpora: Sections 3–4's bug analysis covers 7 configurations × up to 3 samples × 95 development tasks = 1,140 runs, on a 4×A100 rig. Sections 5–7 use 38 tasks selected from the same 95-task development set, 3 samples each (114 runs per arm), on the same rig with concurrency adjusted to approximate the scorer's L4 decode rate. Neither corpus is the hidden scoring set; the competition publicly shows 58 of roughly 120 hidden tasks from repositories disjoint from the four public training ones.

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

We also found 602 back-to-back identical `edit_file`/`write_file` calls — the agent retrying an already-rejected call unchanged. Per-arm, the `old_string`-loss rate spans 16.1%–41.8%; no single configuration knob explains the spread and we make no causal claim from it.

Two preserved traces illustrate the shape: in `fastapi_14978`, a call targeting a multi-line Python parameter block arrived with `new_string` intact but ending in the literal fragment `` `,old_string: ``, with spurious top-level keys `embed`, `\n        media_type`, `\n        alias`. In `fastapi_14258`, a call targeting a docstring with a Markdown link produced garbage keys `) -> None` and `](https`.

### 3.2 Falsifying the naive hypothesis

Our first hypothesis was a parser-level defect: something treating colon characters inside argument values as key-value separators.

This is testable because the vLLM Gemma 4 tool-call parser is open-source (Apache-2.0, `vllm/tool_parsers/gemma4_tool_parser.py`). It uses a **custom wire format** in which string values are delimited by a reserved multi-character token sequence (`<|"|>...<|"|>`), not by colons. A correctly-emitted string value containing any number of colon characters should parse intact.

We extracted the three pure parsing functions (`_parse_gemma4_args`, `_parse_gemma4_value`, `_parse_gemma4_array`) — which depend on nothing beyond Python stdlib — and ran them directly in `scripts/paper/repro_colon_split.py`:
- A colon-rich Python function signature as a `STRING_DELIM`-delimited string value: **parsed correctly**, `old_string` intact.
- A Markdown URL inside a delimited string: **parsed correctly**.
- A bare (undelimited) colon-containing string: produces a truncated value and some spurious keys, but does **not** reproduce the "old_string absent" symptom observed in 710 real calls.
- An unclosed delimiter: the parser consumes the rest of the input as the key's value; again, `old_string` remains present.

The control group independently confirms this: across all 14,127 `run_command` calls in the same corpus, 2,299 contained three or more colon characters (type hints, URLs, heredoc code) and **zero** showed any spurious or missing argument key. If the parser split on colons uniformly, `run_command` would fail at the same rate as `edit_file`. It does not.

### 3.3 What the evidence says and does not say

The parser is not the cause. The `old_string`-absent symptom arises from something earlier in the pipeline — either a model-generation failure to correctly emit the `STRING_DELIM` closing token after a long, verbatim, character-exact source-copy, or a serialization step specific to `edit_file`'s argument schema that does not affect `run_command`. We have not reproduced the exact corruption mechanism from first principles: our traces preserve only the post-parse argument dicts, not the raw model token output. We state this as an open, falsifiable question; the falsification of the colon-splitter hypothesis stands independently.

### 3.4 Mitigation and a note on the denominator

We routed most edits through `run_command` with a Python heredoc, entirely avoiding `edit_file` for any multi-line or colon-containing match text. In the `T1_toolfix` arm, the prompt explicitly reserves `edit_file` for short, colon-free targets; this policy produced only 63 `edit_file` calls across 41 of 285 runs. Of those 63 calls, 3 (4.8%) lost `old_string`. This denominator is a selected subset by construction — calls the policy judged safe to route through `edit_file` — not a random sample of all edits. The finding is that routing edits away from the `edit_file` interface avoids the failure class, not that `edit_file` became more reliable on the calls it did receive.

---

## 4. A Second Silent Bug: Pipefail Masks Failing Tests

### 4.1 The symptom

The agent's verification step instructs the agent to run tests as:
```
timeout 150 python -m pytest tests/path.py -x -q -k keyword 2>&1 | tail -25
```
In bash, the exit status of a pipeline is the exit status of its **last command** — `tail`, which almost always exits 0 regardless of what pytest reported. Across 1,140 runs in the same corpus, we found **85 pytest calls piped through `tail` across 65 of 285 `T1_toolfix` runs** (22.8% of runs had at least one affected call). No other arm showed this pattern, because only `T1_toolfix`'s prompt included this specific command template.

### 4.2 The mechanism

When pytest reports failing tests (exit code 1) or a collection error (exit code 2), the pipeline masks both as success. The agent, seeing no error exit code, may conclude its fix is working and proceed toward submission with a patch that fails the actual hidden tests.

### 4.3 The fix

```
timeout 150 python -m pytest tests/path.py -x -q -k keyword > /tmp/t.log 2>&1; ec=$?; tail -25 /tmp/t.log; echo "EXIT:$ec"
```
Redirect to a file, capture `$?` immediately after pytest (before any subsequent command can overwrite it), then show the tail and print the saved exit code explicitly. The agent now sees `EXIT:1` when tests fail.

---

## 5. Lesson A — Calibrating an Agent Benchmark Is Harder Than Matching One Throughput Number

At N=5 concurrent tasks per GPU (A100), our best arm (`T3_5m`) resolved 37/114 (32.5%), mean task duration 284 s. The competition scorer's hardware (4×L4) has a measured single-stream decode rate of 34.3 tok/s, roughly 79% of our local rate. Recalibrating to N=8 concurrent tasks — matching the throughput ratio — dropped the same arm to 28/114 (24.6%), mean 302 s, with 79/114 runs now hitting the 5-minute cap. An 8-percentage-point swing from changing one concurrency parameter, without changing the agent prompt.

One calibration caveat is worth noting: the `duration_seconds` field our matrix runner saves includes both the agent loop and the grading/verification step (Phase 2). A separate analysis of 1,140 runs found that grading adds roughly 25–30 s to a typical 300–450 s task (about 7–9%), making duration-based extrapolations slightly high as estimates of pure agent time. This does not change the conclusion about concurrency sensitivity, but it does affect any precise claim about rig-vs-scorer timing margin.

The practical lesson: agent benchmarks under a wall-clock budget are more sensitive to concurrency than to raw token throughput, because concurrency determines how many runs hit the timeout boundary rather than completing. Absolute resolution-rate estimates varied by several percentage points between rig settings. Relative rankings across arms proved more stable across both local settings and on the public leaderboard, suggesting relative comparisons are the more defensible signal from a local rig.

---

## 6. Lesson B — A Multi-Agent Ensemble Can Be Disabled by One Runaway Sub-Agent

We built a parallel best-of-N ensemble (three independent solvers feeding a selector stage that calls `submit_patch`) to explore whether sampling diversity helps. Each solver was prompt-budgeted to 7 tool calls; the selector to 3.

At corrected single-task concurrency, the ensemble resolved 2/114 (1.8%) versus 29/114 (25.4%) for the single-agent baseline — sign-test p ≈ 2×10⁻⁷. The cause was structural: ADK's `ParallelAgent` blocks until every sub-agent completes, and the harness enforces one session-wide budget (not per-sub-agent). In one inspected trace, one solver made ~20 tool calls against its 7-call prompt budget, running to 311 elapsed seconds while siblings finished around 120–125 s, exhausting time before the selector could run. Mean tool calls per run rose from 15.7 to 56.8; failures shifted almost entirely to timeout (70/114) and empty patch (32/114).

A prompt-stated call limit is not an enforced limit. Multi-agent designs whose only commit path sits behind a stage that can be starved by a sibling need either framework-level per-sub-agent enforcement (unavailable here) or a non-blocking fallback path.

---

## 7. Lesson C — Self-Distillation Fine-Tuning Can Be Defeated by Inference Latency

We ran four LoRA self-distillation attempts on the agent's own resolved trajectories, each addressing a failure mode found in the prior attempt, against a pre-registered bar (promote if net wins ≥ +9; hold if ≤ −9, paired 38-task × 3-sample screen vs. untuned baseline):

| Variant | Key change | Result | Net (sign-test p) |
|---|---|---|---|
| v1 | Full thinking supervised, 90 examples, lr 1e-4 | 29/114 vs. 39/114 | −10 (p = 0.041) |
| v2 | Thinking masked from loss | 19/114 vs. 41/114 | −22 (p ≈ 0) |
| v3 | Thinking supervised, truncated, lr 5e-5 | 31/114 vs. 35/114 | −4 (p = 0.52) |
| v4 | Efficiency-filtered training data | 32/114 vs. 33/114 | −1 (p = 1.0) |

None cleared the bar. The v1 result (net −10, p = 0.041) is based on 38 development tasks with 3 samples each; under a task-block permutation sign-flip, the same data yield p = 0.074 — still directional but not independently significant, so this result should be treated as suggestive rather than definitive. The trend across all four variants, however, is consistent: higher per-call token counts (v1: 305→370 tokens/call, 14.8→17.6 s/call) and more frequent timeout-without-patch runs (v1: empty-at-budget 38→55) appear across variants, and the most latency-affected runs show systematic losses.

A separate confirming signal: v3's packaged submission (Kaggle ref 57412209) failed outright with "exceeded the allowed runtime" — a latency failure under the same 8-minute budget the untuned baseline consistently passes. An adapter that improves per-call formatting can still lose a time-budgeted competition if it increases per-call latency. We observe this cost; we identify latency as the most plausible proximate cause; we do not claim to have isolated it from other potential explanations.

---

## 8. Reproducibility and Competition-Rule Constraints

Our development work lives in a private repository (`dp-pcs/gemma4-swe-agent`) that currently tracks some organizer-provided files (task list, sample submission bundle) — these are subject to competition rule 2.4.b, which restricts redistributing Competition Data to non-participants, and should be cleaned up before that repository is shared publicly.

The paper's supporting code — detection scripts, aggregate evidence, the parser-testing repro script — is released separately in a public repository (`dp-pcs/gemma4-tool-call-reliability`) that contains no organizer-provided data. A companion Kaggle notebook (`results/paper_track/kaggle_notebook.py`) reproduces the key evidence cells and links to that repository.

---

## 9. Related Work

**SWE-bench.** Jimenez et al. (2024) established GitHub-issue resolution as an agent benchmark. Our paper is a case study in how much measured variance in such benchmarks can come from harness-level bugs rather than model capability.

**Tool-calling reliability.** Gorilla (Patil et al., 2023) identified inaccurate argument generation as a core obstacle. The Berkeley Function-Calling Leaderboard systematically tracks call-level accuracy. Noorain et al. (2026) find that per-call error rates compound across multi-step tasks and are substantially higher for smaller or weaker models; our 28.2% argument-loss rate on a 31B model is consistent with that size-dependent pattern on a single model.

**Agent benchmarking methodology.** Zhang et al. (2026) document scaffold-attribution and shape-scoring confounds. Our Section 5 finding (concurrency-driven timeout amplification) is a distinct mechanism but shares the implication that absolute benchmark numbers can diverge from deployment performance under apparently small methodological choices.

**Multi-agent orchestration.** The session-wide-budget starvation we document in Section 6, to our knowledge, has not been previously reported in a live competition setting; we flag it as a concrete design constraint for ADK-based submissions.

---

## 10. Limitations

All findings are from one model family (Gemma 4 31B), one harness (swegemma/ADK), and Python issue-resolution tasks. Section 3's mechanism investigation falsifies one hypothesis but leaves the actual cause open; reproduction from raw token output would be needed to resolve it. Section 4's bug is from one prompt variant; other agents using a different verification command are not affected. Section 5 calibration findings are from two rig settings and a small number of public leaderboard data points (±2.5-task SD per draw). Section 6 is one ensemble design. Section 7's v1 p-value (0.041) does not clear a task-block permutation threshold (0.074), and all four variants share a training pipeline; other fine-tuning approaches may not encounter the same latency trade-off.

---

## 11. Conclusion

We identified and documented two independent tool-interface bugs — one whose mechanism we falsified but could not fully resolve, and one whose mechanism is precise and whose fix is a single line. Three calibration and architecture findings round out the paper. The consistent theme: in fixed-budget agent competitions, infrastructure defects and measurement methodology choices produce effects at least as large as the agent improvements typically under study.

Code, evidence, and the executable parser tests are at https://github.com/dp-pcs/gemma4-tool-call-reliability.

---

## References

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*. arXiv:2310.06770.
- Noorain, A., Mohanty, S., Banerjee, A., & Dasgupta, A. (2026). Invocation-Level Reliability of Tool-Using Agents. arXiv:2608.26189.
- Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.
- Zhang, Y., Motaali, S., Dinh, V. P., Piroutiniya, A., et al. (2026). The Double Measurement Confound in Agent Benchmarks. arXiv:2609.09218.
- Berkeley Function-Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html
- Google. (2026). Agent Development Kit (ADK). https://google.github.io/adk-docs/
