"""Shared, dependency-free evaluation identity and outcome accounting."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(path):
    root = Path(path)
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.pyc')}


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write('\n')
        fh.flush()
        os.fsync(fh.fileno())
        temporary = fh.name
    os.replace(temporary, path)


def classify_termination(text, *, completed=False):
    text = (text or '').lower()
    # Match actual messages, never incidental configuration keys elsewhere in a log.
    if re.search(r'agent timed out|agent exceeded session timeout', text):
        return 'timeout'
    if re.search(r'(?:exceeded|exhausted).{0,35}(?:tool calls?|tool.call budget)', text):
        return 'tool_budget'
    if re.search(r'(?:exceeded|exhausted).{0,40}(?:turns?|llm.calls?)', text):
        return 'turn_budget'
    if 'sandbox execution error' in text or 'traceback (most recent call last)' in text:
        return 'execution_error'
    return 'normal' if completed and not text.strip() else 'unknown'


def valid_summary(path, manifest_sha256, job_key):
    try:
        row = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if (not isinstance(row, dict) or row.get('schema_version') != 2 or row.get('complete') is not True
            or type(row.get('resolved')) is not bool
            or row.get('manifest_sha256') != manifest_sha256
            or row.get('job_key') != job_key
            or row.get('task_id') != job_key.rsplit('/', 1)[-1]
            or row.get('outcome') not in {'graded', 'agent_error'}):
        return None
    return row


def expected_jobs(manifest):
    return [f's{s}/{a}/{t}' for s in manifest['sample_ids']
            for t in manifest['tasks'] for a in manifest['arms']]


def read_complete_matrix(out):
    out = Path(out)
    manifest = json.loads((out / 'matrix.json').read_text())
    if manifest.get('schema_version') != 2:
        raise ValueError('Historical manifest lacks enforced provenance; use an explicitly audited legacy report.')
    sha = digest(manifest)
    rows, missing = {}, []
    for key in expected_jobs(manifest):
        row = valid_summary(out / key / 'job_summary.json', sha, key)
        if row is None:
            missing.append(key)
        else:
            sample, arm, task = key.split('/')
            rows[(sample, arm, task)] = row
    extras = {str(p.parent.relative_to(out)) for p in out.glob('s*/*/*/job_summary.json')} - set(expected_jobs(manifest))
    if missing or extras:
        raise ValueError(f'Incomplete/inconsistent matrix: missing_or_invalid={missing}; unexpected={sorted(extras)}')
    return manifest, rows
