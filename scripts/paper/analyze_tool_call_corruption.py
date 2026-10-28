"""Aggregate edit_file/write_file argument-corruption signatures across preserved eval-matrix
traces (1,140 runs from results/eval_matrix_20261014/out/s0 and
results/eval_matrix_20261015/out/{s0,s1,s2}).

Reports three distinct, non-overlapping categories for edit_file calls, rather than one blended
"malformed" bucket, so the paper can cite a single well-defined number instead of mixing
definitions:
  - missing_old_string_only: new_string present, old_string absent (the cleanest symptom of the
    argument-loss failure discussed in results/paper_track/writeup_draft.md Section 3 — if the
    call were simply malformed/truncated for an unrelated reason, new_string would usually be
    damaged too). NOTE: the proximate cause is an open question, not confirmed to be a parser
    defect — see Section 3.2-3.3 of the writeup for a falsified "naive colon-split parser"
    hypothesis, ruled out via the real open-source gemma4 parser source and a run_command
    control group.
  - missing_both: neither old_string nor new_string present
  - missing_new_string_only: old_string present, new_string absent
  - any_missing (= sum of the above three): matches scripts/eval_matrix/mechanism_check.py's
    broader "edit_malformed" definition (any required argument absent), used elsewhere in this
    project for arm-level screening

Also counts back-to-back identical edit_file/write_file calls (same function + arguments, no
other tool call in between) — the retry-after-rejection signature.

Usage: python3 -I analyze_tool_call_corruption.py OUT_DIR [OUT_DIR ...] > summary.json

Note: the raw trace JSON files under results/*/out/*/*/*/traces/*.json are gitignored (large,
regenerable by re-running the matrix) and are not part of this public repo. This script and its
output summary are the citable, committed artifacts.
"""

import collections
import glob
import json
import sys


def classify_edit_call(args: dict) -> str:
    has_old = "old_string" in args
    has_new = "new_string" in args
    if has_old and has_new:
        return "ok_args_present"
    if has_new and not has_old:
        return "missing_old_string_only"
    if has_old and not has_new:
        return "missing_new_string_only"
    return "missing_both"


def parse_args(tc: dict) -> dict:
    args = tc.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"_raw": args}
    return args


def count_dir(base: str, totals: collections.Counter) -> None:
    for arm_path in sorted(glob.glob(f"{base}/*")):
        for tp in glob.glob(f"{arm_path}/*/traces/*.json"):
            try:
                steps = json.load(open(tp))["steps"]
            except (OSError, ValueError, KeyError):
                continue
            totals["runs"] += 1
            prev_call_sig = None
            for s in steps:
                for tc in s.get("tool_calls") or []:
                    fn = tc.get("function_name")
                    args = parse_args(tc)

                    if fn in ("edit_file", "write_file"):
                        sig = (fn, json.dumps(args, sort_keys=True))
                        if sig == prev_call_sig:
                            totals["back_to_back_identical_edit_or_write"] += 1
                        prev_call_sig = sig
                    else:
                        prev_call_sig = None

                    if fn == "edit_file":
                        totals["edit_file_total"] += 1
                        totals[classify_edit_call(args)] += 1


def main() -> None:
    totals: collections.Counter = collections.Counter()
    for base in sys.argv[1:]:
        count_dir(base, totals)

    ef = totals["edit_file_total"]
    any_missing = (
        totals["missing_old_string_only"]
        + totals["missing_both"]
        + totals["missing_new_string_only"]
    )

    def pct(num, den):
        return round(100 * num / den, 1) if den else 0.0

    out = {
        "source_dirs": sys.argv[1:],
        "methodology": (
            "Per-call classification of edit_file arguments by presence/absence of old_string "
            "and new_string keys after JSON parsing of the harness-recorded tool_calls[].arguments "
            "field; back-to-back duplicate detection compares consecutive edit_file/write_file "
            "calls (same function name + identical parsed arguments) with no other tool call "
            "between them."
        ),
        "runs": totals["runs"],
        "edit_file_total": ef,
        "edit_file_ok_args_present": totals["ok_args_present"],
        "edit_file_missing_old_string_only": totals["missing_old_string_only"],
        "edit_file_missing_old_string_only_pct": pct(totals["missing_old_string_only"], ef),
        "edit_file_missing_both": totals["missing_both"],
        "edit_file_missing_new_string_only": totals["missing_new_string_only"],
        "edit_file_any_missing_required_arg": any_missing,
        "edit_file_any_missing_required_arg_pct": pct(any_missing, ef),
        "back_to_back_identical_edit_or_write_calls": totals["back_to_back_identical_edit_or_write"],
        "notes": [
            "missing_old_string_only (new_string present, old_string absent) is the cleanest "
            "symptom of the edit_file argument-loss failure (see writeup Section 3): the call "
            "carries a replacement but the match target was lost. Mechanism is an open question "
            "(the naive colon-split parser hypothesis was falsified; see Section 3.2-3.3).",
            "edit_file_any_missing_required_arg matches the broader 'malformed' definition used "
            "by scripts/eval_matrix/mechanism_check.py elsewhere in this project (any required "
            "argument absent, for any reason) -- do not conflate the two percentages when citing.",
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
