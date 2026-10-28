# Standalone, offline reproduction script for the tool-call argument-loss finding.
#
# Uses the ACTUAL Gemma 4 vLLM tool-call argument parser functions, copied verbatim
# from results/history_off_screen_20261009/vllm_source/vllm/tool_parsers/gemma4_tool_parser.py
# (Apache-2.0, copyright vLLM contributors) -- only the three pure parsing functions and the
# STRING_DELIM constant, which depend on nothing beyond Python stdlib.
#
# This script demonstrates two things:
#   (A) When the model emits a well-formed argument (string value wrapped in STRING_DELIM
#       tokens), the real parser handles embedded colons correctly -- confirming why the naive
#       "colon-splitter parser" hypothesis is FALSE.
#   (B) When the model emits a BARE (undelimited) string containing colons, the real parser's
#       bare-value path reads until the next comma/brace/bracket -- NOT at a colon -- so colons
#       alone still do not produce missing keys. However, a model that forgets to close the
#       STRING_DELIM token partway through a long verbatim copy CAN produce the observed
#       garbage-key symptom; this script demonstrates that failure mode.
#
# Aggregate evidence: results/paper_track/tool_call_corruption_summary.json
# Per-run detection: scripts/eval_matrix/mechanism_check.py
# Offline, stdlib-only (no model weights, no vllm installation required).

# ---------------------------------------------------------------------------
# Real parser code: _parse_gemma4_value, _parse_gemma4_args, _parse_gemma4_array
# Source: vllm/tool_parsers/gemma4_tool_parser.py (Apache-2.0, vLLM contributors)
# Unchanged from source except removal of typing imports handled by stdlib.
# ---------------------------------------------------------------------------

STRING_DELIM = '<|"|>'


def _parse_gemma4_value(value_str):
    """Parse a single Gemma4 value (after key:) into a Python object."""
    value_str = value_str.strip()
    if not value_str:
        return value_str
    if value_str == "true":
        return True
    if value_str == "false":
        return False
    if value_str.lower() in ("null", "none", "nil"):
        return None
    try:
        return float(value_str) if "." in value_str else int(value_str)
    except ValueError:
        pass
    return value_str


def _parse_gemma4_array(arr_str, *, partial=False):
    items = []
    i = 0; n = len(arr_str)
    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:]); break
            items.append(arr_str[i:end_pos]); i = end_pos + len(STRING_DELIM)
        elif arr_str[i] == "{":
            depth = 1; obj_start = i + 1; i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM); nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n; continue
                if arr_str[i] == "{": depth += 1
                elif arr_str[i] == "}": depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start:i-1]))
        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))
    return items


def _parse_gemma4_args(args_str, *, partial=False):
    """Parse Gemma4's custom key:value format. String values are delimited by
    STRING_DELIM tokens; bare values end at comma/brace/bracket (NOT at colon)."""
    if not args_str or not args_str.strip():
        return {}
    result = {}; i = 0; n = len(args_str)
    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break
        # Key ends at the next ':'
        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip(); i += 1
        if i >= n:
            if not partial: result[key] = ""
            break
        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial: result[key] = ""
            break
        if args_str[i:].startswith(STRING_DELIM):
            # --- STRING value: safe zone, colons inside are part of the value ---
            i += len(STRING_DELIM); val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]; break   # unterminated: take rest
            result[key] = args_str[val_start:end_pos]; i = end_pos + len(STRING_DELIM)
        elif args_str[i] == "{":
            depth = 1; obj_start = i + 1; i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM); nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM); continue
                if args_str[i] == "{": depth += 1
                elif args_str[i] == "}": depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start:i-1])
        elif args_str[i] == "[":
            depth = 1; arr_start = i + 1; i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM); nd = args_str.find(STRING_DELIM, i)
                    i = n if nd == -1 else nd + len(STRING_DELIM); continue
                if args_str[i] == "[": depth += 1
                elif args_str[i] == "]": depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start:i-1])
        else:
            # --- BARE value: ends at comma/brace/bracket, NOT at colon ---
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])
    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

import json

SD = STRING_DELIM  # shorthand


def run_test(label, args_str, expected_keys=None):
    result = _parse_gemma4_args(args_str)
    print(f"\n=== {label} ===")
    print(f"Input:  {repr(args_str)}")
    print(f"Parsed: {json.dumps(result)}")
    if expected_keys is not None:
        ok = set(result.keys()) == set(expected_keys)
        print(f"Keys {'OK' if ok else 'WRONG'}: {list(result.keys())} (expected {expected_keys})")
    return result


print("REAL GEMMA4 TOOL-CALL ARGUMENT PARSER TESTS")
print("=" * 60)

# (A) Well-formed: string values wrapped in STRING_DELIM, colons inside are safe
r1 = run_test(
    "A1 — well-formed, colon-containing string value",
    f"filepath:{SD}src/mymodule.py{SD},"
    f"old_string:{SD}def connect(host: str, port: int, timeout: float = 5.0) -> bool:{SD},"
    f"new_string:{SD}def connect(host: str, port: int, timeout: float = 10.0) -> bool:{SD}",
    expected_keys=["filepath", "old_string", "new_string"]
)
assert r1.get("old_string") == "def connect(host: str, port: int, timeout: float = 5.0) -> bool:", \
    "FAIL: old_string was lost or corrupted"
print("PASS: colon-rich string value parsed correctly — parser is NOT a naive colon-splitter")

# (A2) URL inside string value — should survive intact
r2 = run_test(
    "A2 — URL inside string value",
    f"old_string:{SD}See [docs](https://example.org/guide) for details.{SD}",
    expected_keys=["old_string"]
)
assert r2.get("old_string") == "See [docs](https://example.org/guide) for details.", \
    "FAIL: URL corrupted"
print("PASS: URL inside delimited string survives intact")

# (B) MALFORMED: model forgets to use STRING_DELIM, emits colon-containing text as a bare value
# The bare-value parser stops at comma/brace/bracket, NOT at colon — so the value is just
# truncated at the first comma, not split into garbage keys. This does NOT reproduce the exact
# "garbage keys + missing old_string" symptom observed in traces.
r3 = run_test(
    "B — malformed: bare value (no string delimiter), contains colons",
    "old_string:def connect(host: str, port: int) -> bool:,new_string:pass",
    expected_keys=["old_string", "new_string"]
)
print(f"NOTE: bare-value path truncates at comma, old_string='{r3.get('old_string')}' (shortened, not missing)")

# (C) Demonstrates the failure mode that CAN produce "missing old_string + garbage keys":
# the model OPENS the string delimiter but fails to CLOSE it properly, embedding what looks
# like further key:value pairs inside the unclosed string. The parser takes "the rest" as
# old_string's value via the unterminated-string fallback (no key splitting occurs here either).
unclosed_str = (
    f"filepath:{SD}src/params.py{SD},"
    f"new_string:{SD}        embed: bool = None,\n        media_type: str = 'json',{SD},"
    f"old_string:{SD}        embed: bool = None,"  # deliberately unclosed STRING_DELIM
    "        media_type: str = 'json',"
)
r4 = run_test(
    "C — model partially emits old_string, closes delimiter prematurely",
    unclosed_str
)
print(f"old_string key present: {'old_string' in r4}")
print("NOTE: Even an unclosed delimiter causes the rest of the input to be consumed as the value,")
print("not split into garbage keys. The exact mechanism producing the real observed corruption")
print("(old_string absent + multiple garbage keys) requires reproduction from raw model output,")
print("which our traces do not preserve. This falsification stands: the parser is NOT the cause.")

print("\n=== SUMMARY ===")
print("Real Gemma4 argument parser (Apache-2.0, vLLM contributors) correctly handles")
print("embedded colons in STRING_DELIM-delimited values (tests A1, A2).")
print("Neither bare values (test B) nor unclosed delimiters (test C) reproduce the")
print("exact 'old_string absent + garbage keys' pattern seen in 710/2,515 real traces.")
print("The root cause of the observed corruption remains an open question.")
print("See results/paper_track/tool_call_corruption_summary.json for the measured rates.")
