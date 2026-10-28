"""Per-arm tool-use mechanism counts from matrix traces (one sample round).

Usage: python3 -I mechanism_check.py OUT_DIR/sN [ARM ...]
"""

import collections
import glob
import json
import statistics
import sys


def main() -> None:
    base = sys.argv[1]
    arms = sys.argv[2:] or sorted(p.rsplit("/", 1)[-1] for p in glob.glob(f"{base}/*"))
    for arm in arms:
        c = collections.Counter()
        first_edit = []
        runs = 0
        for tp in glob.glob(f"{base}/{arm}/*/traces/*.json"):
            runs += 1
            steps = json.load(open(tp))["steps"]
            landed = None
            call_idx = 0
            for s in steps:
                for tc in s.get("tool_calls") or []:
                    fn = tc.get("function_name")
                    args = tc.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {"_raw": args}
                    obs = (s.get("observation") or {}).get("content", "") or ""
                    c[fn] += 1
                    if fn not in ("submit_patch", "get_status"):
                        call_idx += 1
                    if fn == "edit_file":
                        if "mandatory input parameters" in obs or "old_string" not in args:
                            c["edit_malformed"] += 1
                        elif "BudgetExceeded" in obs:
                            c["edit_after_budget"] += 1
                        elif "not found" in obs.lower() or "no match" in obs.lower() or "multiple" in obs.lower():
                            c["edit_nomatch"] += 1
                        else:
                            c["edit_ok"] += 1
                            if landed is None:
                                landed = call_idx
                    elif fn == "write_file":
                        if "Path traversal" in obs:
                            c["write_rejected"] += 1
                        elif "mandatory" in obs:
                            c["write_malformed"] += 1
                        elif "BudgetExceeded" not in obs:
                            c["write_ok"] += 1
                    elif fn == "run_command":
                        cmd = str(args.get("command", ""))
                        if "write_text" in cmd or "<<'PY'" in cmd or '<<"PY"' in cmd or "<< 'PY'" in cmd:
                            c["heredoc_edit"] += 1
                            if landed is None and "BudgetExceeded" not in obs and '"exit_code": 0' in obs:
                                landed = call_idx
                        if "git status" in cmd and "git diff" in cmd:
                            c["final_check"] += 1
                        if "/tmp/" in cmd and ("cat >" in cmd or "<<" in cmd):
                            c["scratch_via_run_command"] += 1
            if landed is not None:
                first_edit.append(landed)
        ef = c["edit_file"]
        pct = 100 * c["edit_malformed"] / ef if ef else 0
        graph = c["search_similar_code"] + c["get_code_neighbors"] + c["get_code_subgraph"]
        med = statistics.median(first_edit) if first_edit else 0
        print(
            f"{arm:<15} runs {runs}: edit_file {ef} (ok {c['edit_ok']}, malformed {c['edit_malformed']} = {pct:.0f}%, "
            f"nomatch {c['edit_nomatch']}, after-budget {c['edit_after_budget']}) | write_file {c['write_file']} "
            f"(ok {c['write_ok']}, rejected {c['write_rejected']}) | heredoc edits {c['heredoc_edit']} | "
            f"run_command {c['run_command']} read_file {c['read_file']} get_status {c['get_status']} graph {graph} | "
            f"final git status+diff {c['final_check']} | runs with a landed edit {len(first_edit)}/{runs}, "
            f"median first-edit at call {med:.0f}"
        )


if __name__ == "__main__":
    main()
