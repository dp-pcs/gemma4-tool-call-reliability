# When Edits Vanish: An Unexplained Argument-Loss Failure and Three Measurement Lessons from a Gemma 4 Coding Agent

**Subtitle:** A third of file-edit tool calls lost their match target. We ruled out the obvious cause, and learned hard lessons about benchmark calibration, multi-agent budget sharing, and self-distillation fine-tuning.

---

## Abstract

Building a Gemma 4–based agent for the Kaggle Gemma 4 Developer Agent competition, we found that 710 of 2,515 `edit_file` calls (28.2%) arrived at the harness missing their `old_string` argument while `new_string` was present, causing automatic rejection. We first attributed this to a naive colon-split serialization defect. Inspecting Gemma 4's actual open-source vLLM parser shows it uses an explicit string-delimiter format that safely handles embedded colons; a control group of 14,127 `run_command` calls — colon-dense but zero-corruption — falsifies the naive hypothesis. The true cause remains open. A heredoc-based prompt mitigation cut the failure rate to 4.8%, validating the problem's existence even while its mechanism is unresolved. We report this alongside three independent lessons: local benchmark calibration can still diverge from the real scorer after matching decode throughput; a parallel-solver ensemble can be entirely disabled by one runaway sub-agent when one session-wide budget is shared; and four successive self-distillation LoRA fine-tuning attempts never cleared a pre-registered improvement bar, each time for the same mundane reason — added inference latency lost more tasks to the wall-clock cap than the tuning gained.

---

## 1. Introduction

Agentic coding systems depend on a tool-calling interface between the language model and the environment. Failures in that interface look, from the outside, identical to failures of reasoning — both end in a rejected call or an empty patch — and are often silent in standard logs. This paper documents one such failure found while building an agent for the Google–Gemma 4 Developer Agent competition (Kaggle), which evaluates agents resolving GitHub issues from a fixed harness (swegemma/ADK) under hard time and tool-call budgets.

We report four findings from 1,140+ preserved agent runs. The headline finding (Section 3) is a measured, reproducible, and currently *unexplained* loss of a required argument in 28% of edit attempts — notable both for its rate and because our first hypothesis for the cause was falsified by a clean control group and the actual parser source, and we report that falsification explicitly. Sections 4–6 report three independent lessons: benchmark-rig calibration is harder than matching one throughput number; shared-budget multi-agent frameworks can make concurrency actively harmful; and self-distillation fine-tuning, even when behaviorally effective, can lose a fixed-time competition purely on latency.

---

## 2. Setting

Agents use Gemma 4 31B IT (W4A16) via vLLM with the `gemma4` tool-call and reasoning parsers, built on Google's ADK via swegemma. Per-task budget: 28 tool calls and 4–8 minutes. Tools: `read_file`, `run_command`, `edit_file`, `write_file`, `submit_patch`.

Two corpora: Section 3's corruption analysis covers 7 configurations × up to 3 samples × 95 development tasks = 1,140 runs, on a single A100. Sections 4–6 use 38 held-out tasks (23 FastAPI, 15 Rich, excluded from the earlier screen), 3 samples each (114 runs per arm), on a 4×A100 rig with concurrency throttled to approximate the scorer's L4 decode rate. Neither corpus is the hidden scoring set; the competition publicly shows 58 of roughly 120 hidden tasks from repositories disjoint from the four public ones.

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

We also found 602 back-to-back identical `edit_file`/`write_file` calls — the model retrying an already-rejected call unchanged. Per-arm, the `old_string`-loss rate spans 16.1%–41.8% across six non-mitigation variants; no single configuration knob explains the spread, so we make no causal claim from it.

Two concrete examples from preserved traces illustrate the shape: in `fastapi_14978`, a call targeting a multi-line Python parameter block arrived with `new_string` intact but its value ending in the literal fragment `` `,old_string: ``, with spurious top-level keys `embed`, `\n        media_type`, `\n        alias` — one per line of the intended match text. In `fastapi_14258`, a call targeting a docstring with a Markdown link produced garbage keys `) -> None` and `](https` — breaking at colons inside the intended value.

### 3.2 Falsifying the naive hypothesis

Our first hypothesis was a parser-level defect: something splitting argument text on literal colons, treating Python type annotations, URLs, and dict literals inside `old_string` as new key-start markers.

This is testable. Gemma 4's vLLM tool-call parser (`vllm/tool_parsers/gemma4_tool_parser.py`, Apache-2.0) uses a custom, non-JSON wire format in which **string values are delimited by a reserved token sequence** (`<|"|>...<|"|>`), not by colon boundaries. A correctly formed string value containing any number of colons should survive parsing intact.

Control check: across all 14,127 `run_command` calls in the same corpus, 2,299 contained three or more colon characters in the command string (type hints, URLs, heredoc code), and **zero** showed any spurious or missing argument key. A naive colon-splitter applied uniformly to argument text would corrupt `run_command`'s `command` argument at the same rate as `edit_file`'s. It does not. The naive hypothesis is false.

### 3.3 What the evidence says and does not say

Given the parser's delimiter-based design, a properly-emitted value cannot be split by embedded colons. The corruption must therefore originate earlier — most likely in how the model generates `old_string`/`new_string` values specifically. These are the only arguments in the tool set requiring long, verbatim, character-exact reproduction of previously read source text, rather than freely generated content. We hypothesize a generation-time failure to correctly close the string delimiter during a long, structure-rich copy, but cannot confirm this: our traces record only the post-parse argument dictionaries, not the model's raw pre-parse token output. We report this as an open, falsifiable question. `scripts/paper/repro_colon_split.py` demonstrates the corruption class that *a* naive colon-based parser could exhibit — illustrative for practitioners — with the explicit caveat that it does not describe Gemma 4's actual parser behavior.

### 3.4 Mitigation

We routed most edits through `run_command` with a Python heredoc instead of calling `edit_file` directly. In the mitigation arm (`T1_toolfix`), only 63 `edit_file` calls were made across 41 of 285 runs; of those, 3 (4.8%) lost `old_string` — down from 16.1%–41.8% without the mitigation, regardless of which specific mechanism is ultimately responsible.

---

## 4. Lesson A — Calibrating an Agent Benchmark Is Harder Than Matching One Throughput Number

At N=5 concurrent tasks per GPU on development hardware (A100), our best arm (`T3_5m`) resolved 37/114 (32.5%), mean wall-time 284.4 s. The competition's scorer hardware (4×L4) has a measured single-stream decode rate of 34.3 tok/s, roughly 79% of ours. Re-running at N=8 to approximate that ratio dropped the same arm to 28/114 (24.6%), mean 301.8 s, with 79/114 runs now timing out — an 8-point swing from concurrency alone.

However, this "calibrated" rig still diverges from the real scorer. The same N=8 configuration (`T1@8`) has a mean wall-time of 444.8 s; extrapolated over ~120 sequential hidden tasks, that implies over 14 hours of agent time — yet the actual submission at this configuration completed within the real 12-hour budget and scored 0.17 on the public leaderboard. Local concurrent requests contend for prefill in a way a sequential, single-stream, prefix-caching scorer does not, adding latency beyond what decode-rate calibration captures.

The practical conclusion: even a careful calibration effort may leave an unmodeled latency gap that moves absolute resolution-rate estimates substantially. Relative rankings between configurations proved more stable: our best arm outperformed weaker arms consistently across the two local rigs and on the public leaderboard, even as absolute estimates varied by several percentage points between rigs.

---

## 5. Lesson B — A Multi-Agent Ensemble Can Be Disabled by One Runaway Sub-Agent

We built a parallel best-of-N ensemble (three independent solvers feeding a selector stage that calls `submit_patch`) to explore whether sampling diversity helps. Each solver was prompt-budgeted to 7 tool calls; the selector to 3.

At corrected single-task concurrency, the ensemble resolved 2/114 (1.8%) versus 29/114 (25.4%) for the single-agent baseline — sign-test p ≈ 2×10⁻⁷. The cause was structural: ADK's `ParallelAgent` blocks until every sub-agent completes, and the harness enforces one session-wide budget, not per-sub-agent limits. In one inspected trace, solver_c made ~20 tool calls against its 7-call prompt budget (running to 311 elapsed seconds while siblings finished around 120–125 s), exhausting time before the selector could run. Mean tool calls per run rose from 15.7 to 56.8; failures shifted to timeout (70/114) and empty patch (32/114).

A prompt-level call limit is not an enforced limit. Multi-agent designs whose only commit path sits behind a stage that can be starved by a sibling need either framework-level per-sub-agent enforcement (unavailable here) or a non-blocking fallback path.

---

## 6. Lesson C — Self-Distillation Fine-Tuning Can Lose on Latency Even When Behavior Improves

We ran four LoRA self-distillation attempts (rejection-sampling on the agent's own resolved trajectories), each addressing a specific failure mode found in the prior attempt, evaluated against a pre-registered bar (promote only if net wins ≥ +9; hold if ≤ −9, paired 38-task × 3-sample screen vs. untuned baseline):

| Variant | Key design change | Result | Net (p) |
|---|---|---|---|
| v1 | Full thinking supervised, 90 examples, lr 1e-4 | 29/114 vs. 39/114 | −10 (0.041) |
| v2 | Thinking masked from loss | 19/114 vs. 41/114 | −22 (≈0) |
| v3 | Thinking supervised, truncated, lr 5e-5 | 31/114 vs. 35/114 | −4 (0.52) |
| v4 | Efficiency-filtered training data | 32/114 vs. 33/114 | −1 (1.0) |

None cleared the bar. Each variant had a diagnosable specific failure: v1 became more hesitant (tokens/call 305→370, seconds/call 14.8→17.6, empty-at-budget 38→55); v2 fell into generation loops due to loss-masking removing the "stop thinking" signal; v3 and v4 fixed both of those problems but still ran slightly slower per task than the untrained baseline.

A separate confirming data point: v3's packaged submission to the real scorer (Kaggle ref 57412209) failed outright with "exceeded the allowed runtime" — a latency failure, not a quality failure — under the same 8-minute budget the untuned baseline passes. A fine-tuned model that is neutral or even improved on resolution rate can still be strictly worse in a fixed-wall-clock evaluation if it adds per-call latency. Inference speed is a first-class acceptance criterion for time-budgeted agent competitions.

---

## 7. Reproducibility and Competition-Rule Constraints

This competition's rules forbid redistributing organizer-provided Competition Data to non-participants (rule 2.4.b). Our repository currently tracks organizer-provided files (the public task list and sample submission bundle), which should be removed or handled per the competition's rules before the repository link is published with this paper. Our analysis scripts, aggregate evidence file, and offline demonstration script do not depend on any organizer-provided data and are appropriate to share independently of that cleanup.

---

## 8. Related Work

**SWE-bench.** Jimenez et al. (2024) established GitHub-issue resolution with hidden test suites as an agent benchmark. Our paper is a case study in how much measured variance in such benchmarks can come from harness infrastructure rather than model capability.

**Tool-calling reliability.** Gorilla (Patil et al., 2023) identified inaccurate argument generation as a core obstacle to tool-using LLMs. The Berkeley Function-Calling Leaderboard tracks call-level accuracy across APIs. Noorain et al. (2026) find that weaker models exhibit substantially higher per-call error rates that compound across multi-step tasks; our 28.2% argument-loss rate on a 31B model is consistent with that size-dependent pattern, though we test only one model.

**Agent benchmarking methodology.** Zhang et al. (2026) document scaffold-attribution and shape-scoring confounds in agent benchmarks. Our Section 4 finding — concurrency-driven latency divergence between local and deployed evaluation — is a distinct mechanism but shares the broader implication that absolute benchmark numbers can mislead even under careful methodology.

**Multi-agent orchestration.** The session-wide budget starvation we document in Section 5 is, to our knowledge, not previously reported in a live competition setting; we flag it as a concrete design constraint for ADK-based multi-agent submissions.

---

## 9. Limitations

All findings are from one model family, one harness, and Python issue-resolution tasks. Section 3's mechanism investigation rules out one hypothesis but leaves the cause open. Section 4 is based on two local rigs and a small number of public leaderboard data points (estimated ±2.5-task SD per draw). Section 5 is one ensemble design on one task set. Section 6's four LoRA variants share a training and evaluation pipeline; other fine-tuning paradigms (RL with environment-derived rewards, larger datasets, stronger base models) may not hit the same latency-budget interaction.

---

## 10. Conclusion

We report a tool-call argument-loss failure affecting 28% of file edits, falsify our own initial explanation using the actual parser source and a control group, and leave the mechanism as an open, well-specified question. Three further findings — a calibrated local benchmark still diverging from the real scorer, a multi-agent ensemble disabled by shared-budget starvation, and four fine-tuning attempts lost purely to inference latency — share a common theme: measurement and infrastructure effects in fixed-budget agent competitions are often larger than the effects of the interventions they are used to evaluate. Code and evidence are available in the accompanying repository (subject to the data-cleanup caveat in Section 7).

---

## References

- Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR 2024*. arXiv:2310.06770.
- Noorain, A., Mohanty, S., Banerjee, A., & Dasgupta, A. (2026). Invocation-Level Reliability of Tool-Using Agents. arXiv:2608.26189.
- Patil, S. G., Zhang, T., Wang, X., & Gonzalez, J. E. (2023). Gorilla: Large Language Model Connected with Massive APIs. arXiv:2305.15334.
- Zhang, Y., Motaali, S., Dinh, V. P., Piroutiniya, A., et al. (2026). The Double Measurement Confound in Agent Benchmarks. arXiv:2609.09218.
- Berkeley Function-Calling Leaderboard. https://gorilla.cs.berkeley.edu/leaderboard.html
- Google. (2026). Agent Development Kit (ADK). https://google.github.io/adk-docs/
