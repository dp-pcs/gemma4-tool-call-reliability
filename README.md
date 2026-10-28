# Gemma 4 Tool-Call Reliability: Evidence and Reproduction

Supporting code and evidence for a Kaggle Paper Track submission ("When Edits Vanish: An
Unexplained Argument-Loss Failure and Three Measurement Lessons from a Gemma 4 Coding Agent"),
submitted to the [Google - The Gemma 4 Developer Agent Paper Track](https://www.kaggle.com/competitions/gemma-4-developer-agent-paper)
competition.

## What this is

While building a Gemma 4-based coding agent for the companion
[Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) competition,
we found that 28.2% of `edit_file` tool calls in a 1,140-run evaluation corpus arrived at the
harness missing a required argument. This repository contains:

- `scripts/paper/analyze_tool_call_corruption.py` — the aggregation script that produces the
  headline evidence numbers from preserved agent evaluation traces.
- `scripts/paper/repro_colon_split.py` — a standalone, offline, stdlib-only demonstration of the
  general "naive colon-split argument parsing" corruption class. This does **not** reproduce
  Gemma 4's actual tool-call parser (which we confirmed is NOT a naive colon-splitter — see the
  writeup, Section 3.2) — it illustrates the risk class for other practitioners.
- `scripts/eval_matrix/mechanism_check.py` — the underlying per-run malformed-call detection
  logic used across our broader evaluation work.
- `results/paper_track/tool_call_corruption_summary.json` — the exact aggregate numbers cited in
  the paper, reproducible by re-running the analysis script against evaluation traces.
- `results/paper_track/writeup_draft.md` — the full paper text.

## What this is NOT

This repository does **not** include any Kaggle-provided Competition Data (the task dataset,
harness configuration, sample submission, or Docker sandbox images), per the competition rules'
restriction on redistributing Competition Data to non-participants. The aggregation and
mechanism-check scripts operate on evaluation traces that any competition participant can
regenerate locally using the organizer-provided harness and task data — we do not redistribute
that data here.

## Running the scripts

Both `scripts/paper/repro_colon_split.py` and `scripts/eval_matrix/mechanism_check.py` /
`scripts/paper/analyze_tool_call_corruption.py` are stdlib-only Python 3.

```bash
# Offline demonstration, no data or dependencies needed:
python3 -I scripts/paper/repro_colon_split.py

# Aggregation script (requires your own evaluation trace directories, not included here):
python3 -I scripts/paper/analyze_tool_call_corruption.py OUT_DIR [OUT_DIR ...]
```

## License

Code in this repository is released under the Apache License 2.0 (see `LICENSE`). The vLLM
tool-call parser source referenced in the paper (`gemma4_tool_parser.py`) is part of the vLLM
project and separately licensed Apache-2.0 by its own contributors; it is not included here.
