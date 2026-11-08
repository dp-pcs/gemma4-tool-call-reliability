"""Report termination independently from grading, without inventing phase times.

Usage: python3 -I scripts/paper/measure_repair.py OUT/s0 [OUT/s1 ...]
Historical last-trace time and its residual are explicitly named proxies.
Only phase_timings from an instrumented runner supply measured phase durations.
"""
import collections
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval_matrix'))
from evidence import classify_termination


def get_trace(task_dir):
    files = sorted((task_dir / 'traces').glob('*.json'))
    if not files:
        return {}
    try:
        return json.loads(files[0].read_text())
    except (OSError, ValueError):
        return {}


def get_trace_agent_elapsed(task_dir):
    for step in reversed(get_trace(task_dir).get('steps', [])):
        value = (step.get('extra') or {}).get('elapsed_s')
        if isinstance(value, (int, float)):
            return float(value)
    return None


def classify_run_log(path):
    try:
        return classify_termination(path.read_text(errors='replace'))
    except OSError:
        return 'unknown'


def count_pipefail_pytest(task_dir):
    count = 0
    for step in get_trace(task_dir).get('steps', []):
        for call in step.get('tool_calls') or []:
            if call.get('function_name') != 'run_command':
                continue
            args = call.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    continue
            command = str(args.get('command', ''))
            count += 'pytest' in command and '| tail' in command
    return count


def process_dirs(bases):
    per_arm = collections.defaultdict(list)
    seen = set()
    for base in bases:
        for summary_path in sorted(Path(base).glob('*/*/job_summary.json')):
            if summary_path.resolve() in seen:
                raise ValueError(f'Duplicate input: {summary_path}')
            seen.add(summary_path.resolve())
            summary = json.loads(summary_path.read_text())
            directory = summary_path.parent
            termination = summary.get('termination') or classify_run_log(directory / 'run.log')
            duration = (summary.get('metrics') or {}).get('duration_seconds')
            last_trace = get_trace_agent_elapsed(directory)
            per_arm[summary_path.parent.parent.name].append({
                'resolved': bool(summary.get('resolved')), 'patched': (summary.get('patch_bytes') or 0) > 0,
                'termination': termination, 'duration': duration,
                'last_trace': last_trace,
                'residual': duration - last_trace if duration is not None and last_trace is not None else None,
                'phase_timings': summary.get('phase_timings') or {},
                'pipe_calls': count_pipefail_pytest(directory),
            })

    def summarize(rows):
        def mean(values):
            values = [x for x in values if x is not None]
            return round(sum(values) / len(values), 3) if values else None
        counts = collections.Counter(r['termination'] for r in rows)
        resolved_counts = collections.Counter(r['termination'] for r in rows if r['resolved'])
        timed = [r for r in rows if r['termination'] == 'timeout']
        return {
            'runs': len(rows), 'resolved': sum(r['resolved'] for r in rows),
            'resolution_rate': sum(r['resolved'] for r in rows) / len(rows) if rows else None,
            'patched': sum(r['patched'] for r in rows),
            'termination_counts': dict(sorted(counts.items())),
            'resolved_by_termination': dict(sorted(resolved_counts.items())),
            'timeout_total': len(timed), 'timeout_resolved': sum(r['resolved'] for r in timed),
            'timeout_unresolved_patched': sum(not r['resolved'] and r['patched'] for r in timed),
            'timeout_unresolved_empty': sum(not r['resolved'] and not r['patched'] for r in timed),
            'normal_total': counts['normal'], 'normal_resolved': resolved_counts['normal'],
            'mean_duration_s': mean(r['duration'] for r in rows),
            'mean_last_trace_elapsed_s': mean(r['last_trace'] for r in rows),
            'mean_duration_minus_last_trace_s': mean(r['residual'] for r in rows),
            'mean_agent_loop_s': mean(r['phase_timings'].get('agent_loop_seconds') for r in rows),
            'mean_grading_s': mean(r['phase_timings'].get('verification_seconds') for r in rows),
            'pipefail_pytest_calls': sum(r['pipe_calls'] for r in rows),
            'runs_with_pipefail_pytest': sum(r['pipe_calls'] > 0 for r in rows),
        }
    all_rows = [row for rows in per_arm.values() for row in rows]
    return {
        'source_dirs': bases,
        'note': 'Unknown is not normal completion. Last-trace elapsed and its residual are proxies, not measured phases.',
        'totals': summarize(all_rows),
        'per_arm': {arm: summarize(rows) for arm, rows in sorted(per_arm.items())},
    }


if __name__ == '__main__':
    print(json.dumps(process_dirs(sys.argv[1:]), indent=2))
