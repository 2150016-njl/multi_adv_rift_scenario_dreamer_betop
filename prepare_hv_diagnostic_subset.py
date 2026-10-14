#!/usr/bin/env python3
"""Resume verified unique pairs for diagnosis; never certify full evaluation."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_subset(report,data):
    if report.get('passed') is not False:
        raise ValueError('Expected a failed/incomplete pairing report')
    if hashlib.sha256(data).hexdigest()!=report.get('partial_manifest_sha256'):
        raise ValueError('Partial manifest SHA differs from evidence')
    jobs=[json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    indices=[j['record_index'] for j in jobs]
    if not jobs or len(jobs)!=report['written_jobs'] or len(set(indices))!=len(indices):
        raise ValueError('Empty, duplicate or inconsistent unique pair ledger')
    selected=report['selected_dataset_indices']
    excluded=report['missing_record_indices']+report['ambiguous_record_indices']
    if len(set(selected))!=report['requested_records'] or len(set(excluded))!=len(excluded) or \
            set(indices)&set(excluded) or set(indices)|set(excluded)!=set(selected):
        raise ValueError('Unique/excluded indices do not partition original selection')
    if any(not j.get('snapshot_sha256') or not j.get('snapshot_path') for j in jobs):
        raise ValueError('Missing snapshot provenance')
    return jobs,{
        'evaluation_scope':'unambiguous_diagnostic_subset','full_evaluation_eligible':False,
        'original_requested_records':report['requested_records'],
        'unique_pair_records':len(jobs),'excluded_original_indices':excluded,
        'excluded_missing_indices':report['missing_record_indices'],
        'excluded_ambiguous_indices':report['ambiguous_record_indices'],
        'original_selection':report['selection'],
        'dataset_sha256':report['dataset_sha256'],'generated_sha256':report['generated_sha256'],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-report',type=Path,required=True)
    p.add_argument('--output-manifest',type=Path,required=True)
    a=p.parse_args();report_bytes=a.pair_report.read_bytes();report=json.loads(report_bytes)
    # Bundle relocation is supported: the partial ledger is beside its report.
    partial=a.pair_report.parent/Path(report['partial_manifest_path']).name
    data=partial.read_bytes();jobs,scope=validate_subset(report,data)
    sidecar=Path(str(a.output_manifest)+'.audit.json')
    if a.output_manifest.exists() or sidecar.exists():raise FileExistsError('Refuse to overwrite evidence')
    scope.update(passed=True,requested_records=len(jobs),written_jobs=len(jobs),
                 manifest_sha256=hashlib.sha256(data).hexdigest(),
                 selected_dataset_indices=[j['record_index'] for j in jobs],
                 source_pair_report=str(a.pair_report.resolve()),
                 source_pair_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                 source_partial_sha256=hashlib.sha256(data).hexdigest())
    a.output_manifest.parent.mkdir(parents=True,exist_ok=True)
    a.output_manifest.write_bytes(data)
    sidecar.write_text(json.dumps(scope,indent=2),encoding='utf-8')
    print(json.dumps(scope,indent=2))


if __name__=='__main__':main()
