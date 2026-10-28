"""Measurement repair: per-run termination reason + agent-loop vs grading time split.

Reads job_summary.json, run.log, and trace files (if available) for all arms/samples
in the given OUT_DIR paths. Produces a JSON summary with correct breakdown of:
  - timeout (from run.log "agent timed out") vs normal completion
  - resolved (from job_summary), patched (patch_bytes > 0)
  - agent_loop_s (trace final step elapsed_s), grading_s (duration_s - agent_loop_s)

Usage: python3 -I measure_repair.py OUT_DIR [OUT_DIR ...] > measurement_repair.json
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path


def get_trace_agent_elapsed(task_dir: Path) -> float | None:
    """Return the elapsed_s of the last agent step from the trace, if available."""
    trace_files = list((task_dir / "traces").glob("*.json")) if (task_dir / "traces").exists() else []
    if not trace_files:
        return None
    try:
        trace = json.loads(trace_files[0].read_text())
        steps = trace.get("steps", [])
        # Find last step with an elapsed_s in its extra field (ATIF format)
        for step in reversed(steps):
            elapsed = (step.get("extra") or {}).get("elapsed_s")
            if elapsed is not None:
                return float(elapsed)
    except Exception:
        pass
    return None


def classify_run_log(run_log_path: Path) -> str:
    """Return 'timeout', 'session_timeout', or 'normal' based on run.log content."""
    try:
        text = run_log_path.read_text(errors="replace")
        if "agent timed out" in text.lower():
            return "timeout"
        if "session timeout" in text.lower() or "max_time_minutes" in text.lower() and "exceeded" in text.lower():
            return "session_timeout"
    except Exception:
        pass
    return "normal"


def count_pipefail_pytest(task_dir: Path) -> int:
    """Count pytest-piped-to-tail calls (pipefail bug) in the trace."""
    trace_files = list((task_dir / "traces").glob("*.json")) if (task_dir / "traces").exists() else []
    if not trace_files:
        return 0
    try:
        trace = json.loads(trace_files[0].read_text())
        count = 0
        for step in trace.get("steps", []):
            for tc in step.get("tool_calls") or []:
                if tc.get("function_name") != "run_command":
                    continue
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                cmd = str(args.get("command", ""))
                if "pytest" in cmd and "| tail" in cmd:
                    count += 1
        return count
    except Exception:
        return 0


def process_dirs(bases: list[str]) -> dict:
    per_arm: dict = {}
    totals = collections.Counter()

    for base in bases:
        for arm_path in sorted(glob.glob(f"{base}/*")):
            arm = Path(arm_path).name
            arm_data = per_arm.setdefault(arm, {
                "runs": 0, "resolved": 0, "patched": 0,
                "timeout_resolved": 0, "timeout_unresolved_patched": 0, "timeout_unresolved_empty": 0,
                "normal_resolved": 0, "normal_failed": 0,
                "duration_s_total": 0.0, "agent_loop_s_total": 0.0, "grading_s_total": 0.0,
                "agent_loop_s_count": 0, "pipefail_pytest_calls": 0, "runs_with_pipefail": 0,
            })

            for task_dir in Path(arm_path).iterdir():
                if not task_dir.is_dir():
                    continue
                summary_file = task_dir / "job_summary.json"
                run_log_file = task_dir / "run.log"
                if not summary_file.exists():
                    continue

                try:
                    summary = json.loads(summary_file.read_text())
                except Exception:
                    continue

                resolved = bool(summary.get("resolved"))
                patch_bytes = summary.get("patch_bytes", 0) or 0
                patched = patch_bytes > 0
                duration_s = (summary.get("metrics") or {}).get("duration_seconds") or 0.0

                termination = classify_run_log(run_log_file) if run_log_file.exists() else "unknown"
                agent_loop_s = get_trace_agent_elapsed(task_dir)
                grading_s = (duration_s - agent_loop_s) if (agent_loop_s is not None and duration_s) else None

                pipefail_count = count_pipefail_pytest(task_dir)

                arm_data["runs"] += 1
                totals["runs"] += 1
                if resolved:
                    arm_data["resolved"] += 1
                    totals["resolved"] += 1
                if patched:
                    arm_data["patched"] += 1
                    totals["patched"] += 1
                if duration_s:
                    arm_data["duration_s_total"] += duration_s
                if agent_loop_s is not None:
                    arm_data["agent_loop_s_total"] += agent_loop_s
                    arm_data["agent_loop_s_count"] += 1
                    if grading_s is not None:
                        arm_data["grading_s_total"] += grading_s
                if pipefail_count > 0:
                    arm_data["pipefail_pytest_calls"] += pipefail_count
                    arm_data["runs_with_pipefail"] += 1
                    totals["pipefail_pytest_calls"] += pipefail_count
                    totals["runs_with_pipefail"] += 1

                if termination == "timeout":
                    totals["timeout"] += 1
                    if resolved:
                        arm_data["timeout_resolved"] += 1
                        totals["timeout_resolved"] += 1
                    elif patched:
                        arm_data["timeout_unresolved_patched"] += 1
                        totals["timeout_unresolved_patched"] += 1
                    else:
                        arm_data["timeout_unresolved_empty"] += 1
                        totals["timeout_unresolved_empty"] += 1
                elif termination == "normal":
                    totals["normal"] += 1
                    if resolved:
                        arm_data["normal_resolved"] += 1
                        totals["normal_resolved"] += 1
                    else:
                        arm_data["normal_failed"] += 1
                        totals["normal_failed"] += 1

    def summarize_arm(d: dict) -> dict:
        n = d["runs"] or 1
        al_n = d["agent_loop_s_count"] or 1
        return {
            "runs": d["runs"],
            "resolved": d["resolved"],
            "resolution_rate": round(d["resolved"] / n, 3),
            "patched": d["patched"],
            "timeout_resolved": d["timeout_resolved"],
            "timeout_unresolved_patched": d["timeout_unresolved_patched"],
            "timeout_unresolved_empty": d["timeout_unresolved_empty"],
            "normal_resolved": d["normal_resolved"],
            "normal_failed": d["normal_failed"],
            "mean_duration_s": round(d["duration_s_total"] / n, 1),
            "mean_agent_loop_s": round(d["agent_loop_s_total"] / al_n, 1) if d["agent_loop_s_count"] else None,
            "mean_grading_s": round(d["grading_s_total"] / al_n, 1) if d["agent_loop_s_count"] else None,
            "pipefail_pytest_calls": d["pipefail_pytest_calls"],
            "runs_with_pipefail_pytest": d["runs_with_pipefail"],
        }

    n_total = totals["runs"] or 1
    al_n = sum(1 for base in bases for arm_path in glob.glob(f"{base}/*")
               for task_dir in Path(arm_path).iterdir()
               if task_dir.is_dir() and (task_dir / "traces").exists()) or 1

    return {
        "source_dirs": bases,
        "note": (
            "This summary's timeout-recovery rate is specific to the source_dirs listed above "
            "(the 1,140-run prompt-comparison corpus from 2026-10-14/15). A separate, broader "
            "corpus (harvest2 + other eval-matrix runs, 742 timed-out-with-patch cases) showed a "
            "higher resolution-among-timeouts rate (~35%, 262/742). Both are real measurements of "
            "the same phenomenon on different run populations; do not treat either as the single "
            "'true' rate without citing which corpus it came from."
        ),
        "totals": {
            "runs": totals["runs"],
            "resolved": totals["resolved"],
            "resolution_rate": round(totals["resolved"] / n_total, 3),
            "timeout_total": totals["timeout"],
            "timeout_resolved": totals["timeout_resolved"],
            "timeout_resolved_rate": round(totals["timeout_resolved"] / max(totals["timeout"], 1), 3),
            "timeout_unresolved_patched": totals["timeout_unresolved_patched"],
            "timeout_unresolved_empty": totals["timeout_unresolved_empty"],
            "normal_total": totals["normal"],
            "normal_resolved": totals["normal_resolved"],
            "pipefail_pytest_calls": totals["pipefail_pytest_calls"],
            "runs_with_pipefail_pytest": totals["runs_with_pipefail"],
        },
        "per_arm": {arm: summarize_arm(d) for arm, d in sorted(per_arm.items())},
    }


def main():
    result = process_dirs(sys.argv[1:])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
