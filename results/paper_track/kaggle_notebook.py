# # Gemma 4: Missing Edit Arguments and Misleading Test Status
#
# Supporting evidence for **Missing Edit Arguments and Misleading Test Status: Lessons from a Gemma 4 Coding Agent**.
#
# This notebook checks published aggregate arithmetic and executes preserved pure functions from the actual vLLM Gemma 4 parser (Apache-2.0). It does not rerun the original model evaluations or grade patches. Missing `old_string` triggered explicit tool errors; the mechanism producing those arguments remains unknown. Shell pipelines separately masked failing test exit codes.
#
# [Paper, scripts, provenance and aggregate evidence](https://github.com/dp-pcs/gemma4-tool-call-reliability).
#
# No task statements, task source, reference patches or model traces are embedded. No model, network or GPU is required. A published notebook is supporting material, not a paper submission receipt.

# ## 1. Aggregate evidence
#
# Counts are from 1,140 familiar-development-task runs. The 3/63 T1 result concerns a selected subset of calls; it is not an unbiased improvement in the interface itself.

import json
SUMMARY = json.loads('{"source_dirs": ["results/eval_matrix_20261014/out/s0", "results/eval_matrix_20261015/out/s0", "results/eval_matrix_20261015/out/s1", "results/eval_matrix_20261015/out/s2"], "methodology": "Per-call classification of edit_file arguments by presence/absence of old_string and new_string keys after JSON parsing of the harness-recorded tool_calls[].arguments field; back-to-back duplicate detection compares consecutive edit_file/write_file calls (same function name + identical parsed arguments) with no other tool call between them.", "runs": 1140, "edit_file_total": 2515, "edit_file_ok_args_present": 1777, "edit_file_missing_old_string_only": 710, "edit_file_missing_old_string_only_pct": 28.2, "edit_file_missing_both": 25, "edit_file_missing_new_string_only": 3, "edit_file_any_missing_required_arg": 738, "edit_file_any_missing_required_arg_pct": 29.3, "back_to_back_identical_edit_or_write_calls": 602, "notes": ["missing_old_string_only (new_string present, old_string absent) is the cleanest symptom of the edit_file argument-loss failure (see writeup Section 3): the call carries a replacement but the match target was lost. Mechanism is an open question (the naive colon-split parser hypothesis was falsified; see Section 3.2-3.3).", "edit_file_any_missing_required_arg matches the broader \'malformed\' definition used by scripts/eval_matrix/mechanism_check.py elsewhere in this project (any required argument absent, for any reason) -- do not conflate the two percentages when citing."]}')
print(json.dumps(SUMMARY, indent=2))
assert 1777 + 710 + 25 + 3 == 2515
print("Missing old_string only:", round(710 / 2515 * 100, 2), "%")

# ## 2. Real-parser controls
#
# Well-formed colon-rich strings survive. The bare-value and unclosed-delimiter fixtures do not reproduce an absent `old_string`; these checks reject universal colon splitting, not every generation/parser interaction.

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
#       STRING_DELIM token produces a different malformed value in the fixture below.
#       These fixtures do not reproduce the observed absent-old_string symptom.
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

# (C) Tests an unclosed delimiter; old_string remains present in this fixture:
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
    "C — model opens old_string delimiter but never closes it",
    unclosed_str
)
print(f"old_string key present: {'old_string' in r4}")
print("NOTE: Even an unclosed delimiter causes the rest of the input to be consumed as the value,")
print("not split into garbage keys. The exact mechanism producing the real observed corruption")
print("(old_string absent + multiple garbage keys) requires reproduction from raw model output,")
print("which our traces do not preserve. These controls reject universal colon splitting, not every parser interaction.")

print("\n=== SUMMARY ===")
print("Real Gemma4 argument parser (Apache-2.0, vLLM contributors) correctly handles")
print("embedded colons in STRING_DELIM-delimited values (tests A1, A2).")
print("Neither bare values (test B) nor unclosed delimiters (test C) reproduce the")
print("exact 'old_string absent + garbage keys' pattern seen in 710/2,515 real traces.")
print("The root cause of the observed corruption remains an open question.")
print("See results/paper_track/tool_call_corruption_summary.json for the measured rates.")

# ## 3. Keep termination and grading separate
#
# A resolved patch remains resolved even after an agent timeout. No explicit termination marker means unknown, not normal completion. Last-trace timing is not a measured agent/verification boundary.

MEASUREMENT = json.loads('{"source_dirs": ["results/eval_matrix_20261014/out/s0", "results/eval_matrix_20261015/out/s0", "results/eval_matrix_20261015/out/s1", "results/eval_matrix_20261015/out/s2"], "note": "Unknown is not normal completion. Last-trace elapsed and its residual are proxies, not measured phases.", "totals": {"runs": 1140, "resolved": 441, "resolution_rate": 0.3868421052631579, "patched": 919, "termination_counts": {"timeout": 238, "tool_budget": 6, "turn_budget": 33, "unknown": 863}, "resolved_by_termination": {"timeout": 29, "turn_budget": 9, "unknown": 403}, "timeout_total": 238, "timeout_resolved": 29, "timeout_unresolved_patched": 106, "timeout_unresolved_empty": 103, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 347.22, "mean_last_trace_elapsed_s": 319.179, "mean_duration_minus_last_trace_s": 28.041, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 85, "runs_with_pipefail_pytest": 65}, "per_arm": {"A_recipe": {"runs": 190, "resolved": 72, "resolution_rate": 0.37894736842105264, "patched": 162, "termination_counts": {"timeout": 39, "turn_budget": 14, "unknown": 137}, "resolved_by_termination": {"timeout": 2, "turn_budget": 3, "unknown": 67}, "timeout_total": 39, "timeout_resolved": 2, "timeout_unresolved_patched": 21, "timeout_unresolved_empty": 16, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 331.66, "mean_last_trace_elapsed_s": 301.673, "mean_duration_minus_last_trace_s": 29.987, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}, "B_temp1": {"runs": 95, "resolved": 35, "resolution_rate": 0.3684210526315789, "patched": 77, "termination_counts": {"timeout": 16, "turn_budget": 6, "unknown": 73}, "resolved_by_termination": {"timeout": 1, "turn_budget": 1, "unknown": 33}, "timeout_total": 16, "timeout_resolved": 1, "timeout_unresolved_patched": 8, "timeout_unresolved_empty": 7, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 326.318, "mean_last_trace_elapsed_s": 301.678, "mean_duration_minus_last_trace_s": 24.64, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}, "C_think2k": {"runs": 95, "resolved": 38, "resolution_rate": 0.4, "patched": 83, "termination_counts": {"timeout": 10, "turn_budget": 8, "unknown": 77}, "resolved_by_termination": {"timeout": 2, "turn_budget": 4, "unknown": 32}, "timeout_total": 10, "timeout_resolved": 2, "timeout_unresolved_patched": 3, "timeout_unresolved_empty": 5, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 309.976, "mean_last_trace_elapsed_s": 287.11, "mean_duration_minus_last_trace_s": 22.866, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}, "D_editrecover": {"runs": 95, "resolved": 39, "resolution_rate": 0.4105263157894737, "patched": 82, "termination_counts": {"timeout": 13, "turn_budget": 4, "unknown": 78}, "resolved_by_termination": {"timeout": 2, "unknown": 37}, "timeout_total": 13, "timeout_resolved": 2, "timeout_unresolved_patched": 7, "timeout_unresolved_empty": 4, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 318.187, "mean_last_trace_elapsed_s": 292.809, "mean_duration_minus_last_trace_s": 25.378, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}, "T1_toolfix": {"runs": 285, "resolved": 112, "resolution_rate": 0.3929824561403509, "patched": 217, "termination_counts": {"timeout": 59, "tool_budget": 1, "unknown": 225}, "resolved_by_termination": {"timeout": 8, "unknown": 104}, "timeout_total": 59, "timeout_resolved": 8, "timeout_unresolved_patched": 24, "timeout_unresolved_empty": 27, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 356.081, "mean_last_trace_elapsed_s": 330.651, "mean_duration_minus_last_trace_s": 25.43, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 85, "runs_with_pipefail_pytest": 65}, "V2_budgetaware": {"runs": 285, "resolved": 107, "resolution_rate": 0.37543859649122807, "patched": 219, "termination_counts": {"timeout": 72, "tool_budget": 5, "unknown": 208}, "resolved_by_termination": {"timeout": 8, "unknown": 99}, "timeout_total": 72, "timeout_resolved": 8, "timeout_unresolved_patched": 30, "timeout_unresolved_empty": 34, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 367.577, "mean_last_trace_elapsed_s": 336.643, "mean_duration_minus_last_trace_s": 30.934, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}, "V3_honghanhhh": {"runs": 95, "resolved": 38, "resolution_rate": 0.4, "patched": 79, "termination_counts": {"timeout": 29, "turn_budget": 1, "unknown": 65}, "resolved_by_termination": {"timeout": 6, "turn_budget": 1, "unknown": 31}, "timeout_total": 29, "timeout_resolved": 6, "timeout_unresolved_patched": 13, "timeout_unresolved_empty": 10, "normal_total": 0, "normal_resolved": 0, "mean_duration_s": 377.867, "mean_last_trace_elapsed_s": 343.329, "mean_duration_minus_last_trace_s": 34.539, "mean_agent_loop_s": null, "mean_grading_s": null, "pipefail_pytest_calls": 0, "runs_with_pipefail_pytest": 0}}}')
t = MEASUREMENT["totals"]
assert sum(t["termination_counts"].values()) == 1140
assert t["resolved"] == 441 and t["timeout_resolved"] == 29
print(json.dumps(t, indent=2))

# ## 4. Pipefix: task-block uncertainty and public outcomes
#
# Local paired totals are 44/114 versus 35/114, but repeats share tasks. The task-block sign-flip result is 0.125; the run-level discordance result is 0.02246. Three public pipefix draws were 9, 8, 8/58 versus T1 10, 10/58. Local improvement did not demonstrate a scoring gain.

import itertools, math
PAIRED = json.loads('{"author": "OpenAI GPT-6 (Codex)", "complete_expected_legacy_rows": 228, "limitations": ["Legacy manifest did not enforce artifact/runtime immutability.", "38 familiar development tasks; three samples per task.", "Pattern adoption is not proof of semantic test relevance or of a causal score gain."], "workers_per_endpoint": 2, "resolved": {"T1_toolfix_orig": 35, "t1_pipefix": 44}, "run_wins": 11, "run_losses": 2, "run_sign_p": 0.0224609375, "task_block_p": 0.125, "task_wins": 5, "task_losses": 1, "task_net_deltas": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]}')
deltas = [x for x in PAIRED['task_net_deltas'] if x]
observed = abs(sum(deltas))
permutations = [abs(sum(d*s for d,s in zip(deltas,signs))) for signs in itertools.product((-1,1), repeat=len(deltas))]
block_p = sum(x >= observed for x in permutations) / len(permutations)
run_p = 2 * sum(math.comb(13,i) for i in range(3)) / 2**13
assert block_p == .125 and run_p == .0224609375
print({'task_block_p':block_p,'run_level_p':run_p,'local_resolved':[35,44],'public_T1':[10,10],'public_pipefix':[9,8,8]})

# ## 5. Later E4B scaffold screen: failed investment gate
#
# One frozen local repeat on 38 development tasks produced candidate 3/38 versus baseline 6/38. Zero paired wins, three losses (two-sided p=.25). Candidate/baseline time budgets were 6/8 minutes, call limits 50/28. This comparison does not isolate a scaffold effect. The net>=+3 gate failed, so no repeat 2 or transfer stage ran. Only aggregate arithmetic is checked here; no evaluation data or traces are included.

SCAFFOLD = {'author_model': 'OpenAI GPT-6 (Codex)', 'description': 'Frozen E4B scaffold investment screen; aggregate-only release.', 'tasks': 38, 'repeats': 1, 'jobs': 76, 'model': 'gemma-4-e4b-it-qat-w4a16-ct', 'runtime': 'single NVIDIA L4; four workers; text-only; vLLM 0.19.1; 32,768-token context', 'resolved': {'baseline': 6, 'candidate': 3}, 'paired': {'wins': 0, 'losses': 3, 'ties': 35, 'net': -3, 'two_sided_sign_p': 0.25}, 'budgets': {'baseline': {'command_seconds': 240, 'tool_calls': 28, 'agent_minutes': 8, 'turns': 80}, 'candidate': {'command_seconds': 150, 'tool_calls': 50, 'agent_minutes': 6, 'turns': 80}}, 'gate': {'required_net': 3, 'passed': False, 'repeat2_run': False, 'transfer_run': False}, 'limitations': ['One sample per familiar development task.', 'Complete recipes differ in prompt, tools and budgets; no isolated scaffold-effect claim.', 'Local runtime is not scorer parity; no public-score estimate.'], 'source_manifest_sha256': 'da64cf1b9f4af993aa8b3d0a251b7e406c0fb9dc6466ff79c2800a76ce966464'}
assert SCAFFOLD["jobs"] == 2*SCAFFOLD["tasks"]
assert sum(SCAFFOLD["paired"][k] for k in ["wins","losses","ties"]) == SCAFFOLD["tasks"]
assert SCAFFOLD["resolved"]["candidate"]-SCAFFOLD["resolved"]["baseline"] == -3
assert 2/2**3 == SCAFFOLD["paired"]["two_sided_sign_p"] == .25
assert not SCAFFOLD["gate"]["passed"]
print(json.dumps(SCAFFOLD,indent=2))
