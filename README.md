# Gemma 4 Tool-Call Reliability: Evidence and Reproduction

Supporting materials for **Missing Edit Arguments and Misleading Test Status: Lessons from a Gemma 4 Coding Agent**, prepared for the [Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper).

The paper documents missing edit arguments, misleading shell-pipeline exit status, and limits of local evaluation. Missing arguments triggered explicit tool errors; their cause remains unresolved. The pipefix local gain did not improve public scores. Publication of these materials does not establish submission of the paper Writeup.

## Evidence

- [Paper](results/paper_track/writeup_draft.md): the complete research report, under 3,000 words.
- [Public Kaggle notebook](https://www.kaggle.com/code/latentgenius/gemma-4-agent-tool-interface-failures-evidence): offline aggregate checks and actual-parser controls; no network or GPU required.
- `scripts/paper/repro_colon_split.py`: preserved pure functions from vLLM's Apache-2.0 Gemma 4 parser. Colon-rich delimited strings survive; tested malformed strings do not reproduce absent `old_string`. These fixtures do not exclude other parser/generation interactions.
- `results/paper_track/tool_call_corruption_summary.json`: aggregate counts from 1,140 historical model runs; 710/2,515 edit calls lack only `old_string`.
- `results/paper_track/measurement_repair_summary.json`: corrected termination accounting. Unknown termination is not normal completion; trace residuals are not measured grading time.
- `results/paper_track/pipefix_summary.json`: 44/114 versus 35/114; run-level p=.02246 and task-block p=.125. Public outcomes were 9,8,8/58 versus T1's 10,10/58.

- `results/paper_track/scaffold_screen_summary.json`: later E4B screen, candidate 3/38 versus baseline 6/38, zero paired wins and three losses. Unequal budgets prevent an isolated scaffold-effect claim; the frozen advancement gate failed.

## Reproduction

```bash
# Requires only Python 3 standard library; no evaluation dataset:
python3 -I scripts/paper/repro_colon_split.py
python3 -I results/paper_track/kaggle_notebook.py

# Requires the original evaluation traces, which are not distributed here:
python3 -I scripts/paper/analyze_tool_call_corruption.py OUT_DIR [OUT_DIR ...]
python3 -I scripts/paper/measure_repair.py OUT_DIR [OUT_DIR ...]
```

Offline checks verify the published arithmetic and parser fixtures. They do not independently recreate the model runs or patch-grading verdicts. The 38-task screens reuse familiar development tasks; neither those results nor the FreshSWE controls are hidden-score estimates.

## Data and license

No organizer task statements, task source snapshots, reference/test patches, model traces, weights or credentials are included. Trace-based scripts require a participant's own inputs. Aggregate outcomes and our analysis code are released under Apache-2.0; the included parser functions retain attribution to vLLM contributors. See [LICENSE](LICENSE).

Updated November 10, 2026 with assistance from OpenAI GPT-6 (Codex).
