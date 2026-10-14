"""Fail-closed input discovery shared by snapshot audits and pairing."""
import hashlib
import json
import os
from pathlib import Path


def snapshot_paths(directory):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError('Snapshot directory does not exist or is not a directory: %s. '
                                'Use the dataset summary.json args.input_dir; do not use a placeholder.' % root)
    seen = set()
    paths = []
    def fail(error):
        raise error
    # Collections may contain symlinked worker directories. Visit each real
    # directory once, avoiding symlink cycles and duplicate aliases.
    for current, dirs, files in os.walk(root, followlinks=True, onerror=fail):
        real = Path(current).resolve()
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        paths.extend(Path(current)/name for name in sorted(files) if name.endswith('.json'))
    if not paths:
        raise ValueError('No .json snapshots found under: %s. Check the original collection directory.' % root)
    return sorted(paths)


def manifest_jobs(path):
    path = Path(path)
    if path.name.endswith('.partial.jsonl'):
        raise ValueError('Partial pairing is diagnostic evidence, not an evaluation manifest: %s' % path)
    data = path.read_bytes()
    report_path = Path(str(path)+'.audit.json')
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report.get('passed') is not True:
            raise ValueError('Pairing failed/incomplete: %s. Fix missing/ambiguous sources before native/spawn.' % report_path)
        if report.get('manifest_sha256') != hashlib.sha256(data).hexdigest():
            raise ValueError('Manifest hash differs from pairing evidence: %s' % path)
    jobs = [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    if not jobs:
        raise ValueError('Empty manifest: %s. Rebuild pairing from an existing snapshot directory.' % path)
    return jobs
