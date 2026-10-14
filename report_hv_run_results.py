#!/usr/bin/env python3
"""Read existing audit ledgers; report execution failures separately from model metrics."""
import argparse
from collections import Counter
import datetime
import json
from pathlib import Path

from run_hv_mainline_batch import analyze, read_jsonl


def collect(result_dir):
    experiment = result_dir / 'experiment'
    if not experiment.is_dir():
        raise FileNotFoundError('Missing experiment directory: ' + str(experiment))
    state_path = experiment / 'batch_status.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.is_file() else {}
    stages = {}
    names = set(state.get('stages', {}))
    names.update(p.name for p in experiment.iterdir() if p.is_dir() and (p/'rows.jsonl').is_file())
    for name in sorted(names):
        folder = experiment/name
        rows_path, jobs_path = folder/'rows.jsonl', folder/'jobs.jsonl'
        if not rows_path.is_file():
            stages[name] = {'status': state.get('stages', {}).get(name, {}), 'ledger_present': False}
            continue
        rows = read_jsonl(rows_path)
        jobs = read_jsonl(jobs_path) if jobs_path.is_file() else []
        summary_path = folder/'summary.json'
        summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
        skipped = []
        failed = []
        for i, row in enumerate(rows):
            errors = row.get('errors') or []
            if any('Skipped after replay isolation/cleanup failure' in str(e) for e in errors):
                skipped.append(i)
            elif errors:
                failed.append(i)
        planned = max(len(jobs), len(rows), summary.get('planned_jobs', 0))
        first = failed[0] if failed else None
        stages[name] = {
            'status': state.get('stages', {}).get(name, {}), 'ledger_present': True,
            'planned': planned, 'written': len(rows), 'unwritten': planned-len(rows),
            'skipped_after_abort': len(skipped), 'executed_with_errors': len(failed),
            'rows_without_execution_errors': len(rows)-len(skipped)-len(failed),
            'error_counts': dict(Counter(str(e) for r in rows for e in r.get('errors', []))),
            'first_executed_error': None if first is None else {
                'sequence': first+1, 'job': jobs[first] if first<len(jobs) else None,
                'row': rows[first]},
            'meaning': 'Execution counts are not actor validity or model success rates.'}
    report = {'result_dir': str(result_dir), 'batch_status': state, 'execution': stages,
              'scope': 'Diagnostic paired subset; skipped/unwritten records are unknown, not model negatives.'}
    try:
        report['metrics'] = analyze(experiment)
    except Exception as exc:
        report['metrics_error'] = '%s: %s' % (type(exc).__name__, exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.result_dir)
    output = args.result_dir / ('results_diagnosis_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for name, stage in report['execution'].items():
        if not stage['ledger_present']:
            print(name + ': no row ledger; inspect batch stage status')
            continue
        print('%s: planned=%d written=%d no_execution_errors=%d executed_errors=%d skipped=%d unwritten=%d' %
              (name, stage['planned'], stage['written'], stage['rows_without_execution_errors'],
               stage['executed_with_errors'], stage['skipped_after_abort'], stage['unwritten']))
        first = stage['first_executed_error']
        if first:
            print('  FIRST ERROR sequence=%d record=%s: %s' %
                  (first['sequence'], (first['job'] or {}).get('record_index'), first['row'].get('errors')))
    if 'metrics_error' in report:
        print('METRICS_ERROR=' + report['metrics_error'])
    print('REPORT=' + str(output))


if __name__ == '__main__':
    main()
