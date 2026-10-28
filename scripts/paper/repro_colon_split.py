# Standalone, offline illustration of a "naive colon-split" argument corruption CLASS --
# NOT a reproduction of Gemma 4's actual tool-call parser, which we have since confirmed does
# NOT work this way (see below).
#
# Our 1,140 preserved evaluation traces show edit_file calls targeting Python source with type
# annotations, URLs, or dict literals in old_string/new_string suffer high argument loss: 710 of
# 2,515 calls (28.2%) had new_string present but old_string specifically absent (a further 25
# lost both, 3 lost only new_string; 738/2,515 = 29.3% lost at least one required argument for
# any reason). This dropped to 4.8% (3/63) of edit_file calls with the T1 heredoc-first
# mitigation, regardless of root cause.
#
# We INITIALLY hypothesized the parser splits argument text on literal colons. This is FALSE:
# Gemma 4's actual vLLM tool-call parser (results/history_off_screen_20261009/vllm_source/vllm/
# tool_parsers/gemma4_tool_parser.py, Apache-2.0, open source) uses an explicit string-delimiter
# format (<|"|>...<|"|>) that safely handles embedded colons -- and a control check on 14,127
# run_command calls in the same corpus (2,299 of them colon-heavy) found ZERO equivalent
# corruption. The naive-splitter hypothesis is falsified; the true mechanism is an open question
# (see results/paper_track/writeup_draft.md Section 3.3), plausibly a model-generation-side
# failure specific to long verbatim-reproduction arguments rather than a parser defect.
#
# This script exists only to illustrate, for other practitioners, what a naive colon-based
# parser COULD do to a colon-containing string value -- a real risk class in other systems --
# not to assert this is what Gemma 4's parser does.
#
# Aggregate evidence: results/paper_track/tool_call_corruption_summary.json
# Per-run detection methodology: scripts/eval_matrix/mechanism_check.py
# Offline, stdlib-only, no model weights or private data needed.

import json

def flat_encode(args: dict) -> str:
    """Encode args as a flat 'key: value\\nkey: value' string (no escaping of colons in values).
    This is the format we hypothesize the model emits when it cannot reliably escape colons."""
    return "\n".join(f"{k}: {v}" for k, v in args.items())


def naive_parse(text: str) -> dict:
    """Parse a flat colon-space-delimited arg block by splitting each line on the first ': '.
    This is the simplest parser that would corrupt type-annotation-containing values."""
    parsed = {}
    for line in text.splitlines():
        if ": " in line:
            key, _, rest = line.partition(": ")
            # Take the key name from the last word on its own line (trim leading whitespace)
            key = key.strip()
            if key:
                parsed[key] = rest
    return parsed


# ── Intended args for a call editing a function signature ──────────────────────────────────────

INTENDED = {
    "filepath": "mymodule/utils.py",
    "old_string": (
        "def connect(host: str, port: int, timeout: float = 5.0, retries: int | None = None) -> bool:"
    ),
    "new_string": (
        "def connect(host: str, port: int, timeout: float = 10.0, retries: int | None = None) -> bool:"
    ),
}

def main() -> None:
    print("=== Intended tool-call arguments ===")
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

    print("=== Diagnosis ===")
    old_ok = corrupted.get("old_string") == INTENDED["old_string"]
    print(f"old_string present: {'old_string' in corrupted}")
    print(f"old_string intact:  {old_ok}")
    if not old_ok:
        print("Observed corruption: 'old_string' is truncated or absent.")
        print("Spurious keys created from type-annotation colons:",
              [k for k in corrupted if k not in INTENDED])
        print("Harness would reply: mandatory input parameters are not present: old_string")
    else:
        print("(No corruption on this example — increase colon density to trigger.)")

    print()
    print("Real example (from a preserved trace, reformatted; actual FastAPI code shown as schema):")
    print("  Intended key  : old_string = '<function param block with embed: bool, media_type: str, alias: str>'")
    print("  Captured keys : embed, \\n        media_type, \\n        alias  (one per colon-bearing line)")
    print("  Captured value: old_string absent; harness error confirmed in preserved trace")
    print("  NOTE: this matches the SYMPTOM, not a confirmed mechanism -- see script header.")
    print("  Source trace  : results/eval_matrix_20261014/out/s0/A_recipe/fastapi_14978/traces/")


if __name__ == "__main__":
    main()
