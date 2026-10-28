# =============================================================================
# HOW TO USE AS A KAGGLE NOTEBOOK
# =============================================================================
# 1. Go to kaggle.com/code → "New Notebook" → "File" → "Import Notebook"
# 2. Upload this file (kaggle_notebook.py).  Kaggle recognises "# %%" cells.
# 3. Set the notebook to PUBLIC before saving (required for the Paper Track
#    "Project Links" field).
# 4. Run all cells; confirm output matches the printed evidence below.
# 5. Copy the notebook URL into the Paper Track submission's Project Links.
#
# No GPU, no internet access, no competition data required.
# Everything runs with Python stdlib (json, textwrap).
# =============================================================================

# %% [markdown]
# # Supporting Evidence — "When Edits Vanish"
#
# **Gemma 4 Developer Agent Paper Track submission**
#
# This notebook provides two pieces of standalone, reproducible evidence for
# the paper *"When Edits Vanish: An Unexplained Argument-Loss Failure and
# Three Measurement Lessons from a Gemma 4 Coding Agent"*:
#
# 1. **Aggregate corruption counts** from 1,140 preserved evaluation runs —
#    the headline numbers cited in Section 3 of the paper.
# 2. **An offline illustration** of the "naive colon-split" argument-corruption
#    *class* — useful for practitioners building tool-calling agents, even
#    though we confirmed this is **not** how Gemma 4's actual parser works
#    (see the companion repo for the full investigation).
#
# **No competition data is used or embedded here.**  The counts were derived
# from evaluation traces any competition participant can regenerate using the
# organiser-provided harness; the aggregation script is in the companion repo.
#
# Companion repo (scripts + evidence + full paper):
# https://github.com/dp-pcs/gemma4-tool-call-reliability

# %% [markdown]
# ## 1. Aggregate Evidence
#
# The table below shows how 2,515 `edit_file` calls across 1,140 evaluation
# runs (seven prompt/configuration variants, up to three samples, 95 tasks)
# split by which required arguments arrived at the harness intact.
#
# | Field | Value | Meaning |
# |---|---|---|
# | `runs` | 1,140 | Total agent evaluation runs in the corpus |
# | `edit_file_total` | 2,515 | Total `edit_file` calls made |
# | `edit_file_ok_args_present` | 1,777 (70.7%) | Both `old_string` and `new_string` present |
# | **`edit_file_missing_old_string_only`** | **710 (28.2%)** | **`new_string` present, `old_string` absent — the headline figure** |
# | `edit_file_missing_both` | 25 (1.0%) | Both arguments absent |
# | `edit_file_missing_new_string_only` | 3 (0.1%) | `old_string` present, `new_string` absent |
# | `edit_file_any_missing_required_arg` | 738 (29.3%) | Any required argument absent (all three above combined) |
# | `back_to_back_identical_edit_or_write_calls` | 602 | Consecutive identical calls — model retrying a rejected call unchanged |
#
# The 710-call (28.2%) figure is the cleanest signal: `new_string` arrived
# correctly, so the call was well-intentioned, but `old_string` (the match
# target) was lost in transit.  All 710 received the harness error
# *"mandatory input parameters are not present: old_string"*.
#
# With the T1 heredoc-first mitigation (routing edits through `run_command`
# with a Python heredoc instead of `edit_file`), this dropped to 3/63 = **4.8%**
# of the `edit_file` calls that *were* still made — validating the problem
# regardless of its unresolved root cause.

# %%
import json, textwrap

SUMMARY = {
  "source_dirs": [
    "results/eval_matrix_20261014/out/s0",
    "results/eval_matrix_20261015/out/s0",
    "results/eval_matrix_20261015/out/s1",
    "results/eval_matrix_20261015/out/s2",
  ],
  "methodology": (
      "Per-call classification of edit_file arguments by presence/absence of "
      "old_string and new_string keys after JSON parsing of the harness-recorded "
      "tool_calls[].arguments field; back-to-back duplicate detection compares "
      "consecutive edit_file/write_file calls (same function name + identical "
      "parsed arguments) with no other tool call between them."
  ),
  "runs": 1140,
  "edit_file_total": 2515,
  "edit_file_ok_args_present": 1777,
  "edit_file_missing_old_string_only": 710,
  "edit_file_missing_old_string_only_pct": 28.2,
  "edit_file_missing_both": 25,
  "edit_file_missing_new_string_only": 3,
  "edit_file_any_missing_required_arg": 738,
  "edit_file_any_missing_required_arg_pct": 29.3,
  "back_to_back_identical_edit_or_write_calls": 602,
}

ef = SUMMARY["edit_file_total"]
print(f"Corpus: {SUMMARY['runs']} runs, {ef} edit_file calls\n")
print(f"  Both args present:            {SUMMARY['edit_file_ok_args_present']:>5}  ({100*SUMMARY['edit_file_ok_args_present']/ef:.1f}%)")
print(f"  old_string absent (new ok):   {SUMMARY['edit_file_missing_old_string_only']:>5}  ({SUMMARY['edit_file_missing_old_string_only_pct']:.1f}%)  ← headline")
print(f"  Both absent:                  {SUMMARY['edit_file_missing_both']:>5}  ({100*SUMMARY['edit_file_missing_both']/ef:.1f}%)")
print(f"  new_string absent (old ok):   {SUMMARY['edit_file_missing_new_string_only']:>5}  ({100*SUMMARY['edit_file_missing_new_string_only']/ef:.1f}%)")
print(f"  Any arg missing (total):      {SUMMARY['edit_file_any_missing_required_arg']:>5}  ({SUMMARY['edit_file_any_missing_required_arg_pct']:.1f}%)")
print(f"\n  Back-to-back identical retries: {SUMMARY['back_to_back_identical_edit_or_write_calls']}")
print(f"\nMitigation (T1 heredoc-first): 3/63 edit_file calls missing old_string = 4.8%")
print(f"  (vs {SUMMARY['edit_file_missing_old_string_only_pct']}% without mitigation)")

# %% [markdown]
# ## 2. Illustration — What a Naive Colon-Split Parser Would Do
#
# > **Important caveat:** This is NOT a reproduction of Gemma 4's actual
# > tool-call parser.  Gemma 4's vLLM parser (`gemma4_tool_parser.py`,
# > Apache-2.0) uses an explicit `<|"|>...<|"|>` string-delimiter format that
# > safely handles embedded colons — and a control check on 14,127
# > `run_command` calls in the same 1,140-run corpus (2,299 of them
# > colon-heavy) found **zero** stray or missing argument keys.  The naive
# > colon-split hypothesis was **falsified** by our own data.  The true
# > mechanism remains an open question (see Section 3.3 of the paper).
#
# This cell illustrates the corruption *class* that a naive colon-space
# splitter would produce — useful context for any practitioner building a
# tool-calling agent over a custom serialization layer.

# %%
def flat_encode(args: dict) -> str:
    """Encode args as a flat 'key: value\\nkey: value' text (no value-escaping)."""
    return "\n".join(f"{k}: {v}" for k, v in args.items())


def naive_parse(text: str) -> dict:
    """Parse by splitting each line on the first ': ' — the simplest strategy
    that would corrupt type-annotation-containing values."""
    parsed = {}
    for line in text.splitlines():
        if ": " in line:
            key, _, rest = line.partition(": ")
            key = key.strip()
            if key:
                parsed[key] = rest
    return parsed


INTENDED = {
    "filepath": "mymodule/utils.py",
    # Multi-line value mirrors real failures: each "param: type" line is a colon-bearing line
    # that the naive parser misreads as a new top-level key when the value has embedded newlines.
    "old_string": (
        "    def connect(\n"
        "        host: str,\n"
        "        port: int,\n"
        "        timeout: float = 5.0,\n"
        "        retries: int | None = None,\n"
        "    ) -> bool:"
    ),
    "new_string": (
        "    def connect(\n"
        "        host: str,\n"
        "        port: int,\n"
        "        timeout: float = 10.0,\n"
        "        retries: int | None = None,\n"
        "    ) -> bool:"
    ),
}

print("=== Intended arguments ===")
print(json.dumps(INTENDED, indent=2))
print()

encoded = flat_encode(INTENDED)
print("=== Flat encoding (no colon-escaping) ===")
print(encoded)
print()

corrupted = naive_parse(encoded)
print("=== Naive colon-space parse of that encoding ===")
print(json.dumps(corrupted, indent=2))
print()

old_ok = corrupted.get("old_string") == INTENDED["old_string"]
print("=== Diagnosis ===")
print(f"old_string present : {'old_string' in corrupted}")
print(f"old_string intact  : {old_ok}")
if not old_ok:
    spurious = [k for k in corrupted if k not in INTENDED]
    got = corrupted.get("old_string", "<absent>")
    print(f"old_string value   : {repr(got)}")
    print(f"Spurious keys from type-annotation colons: {spurious}")
    if "old_string" not in corrupted:
        print("Harness reply: 'mandatory input parameters are not present: old_string'")
    else:
        print("Harness reply: edit fails (old_string truncated; no match found in source file)")
    print()
    print("In our real corpus: 710/2,515 edit_file calls had old_string ENTIRELY ABSENT")
    print("(new_string present but old_string key missing from the parsed argument dict).")

print()
print(textwrap.dedent("""\
    Observed in real traces (schema only — actual FastAPI code not shown):
      Intended : old_string = '<param block with embed: bool, media_type: str, alias: str>'
      Captured : embed, \\n        media_type, \\n        alias  (one per colon-bearing line)
      old_string: absent — harness error confirmed in preserved trace fastapi_14978
      NOTE: this matches the symptom; the mechanism is an open question (see paper §3.3).
"""))

# %% [markdown]
# ## Citation and Links
#
# **Paper:**
# > "When Edits Vanish: An Unexplained Argument-Loss Failure and Three
# > Measurement Lessons from a Gemma 4 Coding Agent."
# > Submitted to the Google – Gemma 4 Developer Agent Paper Track (Kaggle),
# > November 2026.
#
# **Companion repository** (aggregation scripts, evidence JSON, full paper text):
# https://github.com/dp-pcs/gemma4-tool-call-reliability
#
# **Scripts:**
# - `scripts/paper/analyze_tool_call_corruption.py` — reproduces the aggregate
#   counts from evaluation traces
# - `scripts/paper/repro_colon_split.py` — the standalone corruption-class demo
#   above, with full source comments
# - `scripts/eval_matrix/mechanism_check.py` — the per-run malformed-call
#   detection methodology used across the broader evaluation work
#
# **License:** Apache 2.0.
