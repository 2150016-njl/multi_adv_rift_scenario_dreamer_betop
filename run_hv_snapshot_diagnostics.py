#!/usr/bin/env python3
"""Run a fixed-prefix snapshot comparison, keeping commands and console logs."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hv_audit_inputs import snapshot_paths


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-jsonl', type=Path, default=root/'data/hv_init/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl')
    p.add_argument('--generated-jsonl', type=Path, default=root/'data/hv_init/eval/v2_v3_same_v3test_full_20261006/v2_raw_s1_generated.jsonl')
    p.add_argument('--raw', type=Path, help='Default: dataset summary.json args.input_dir')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-records', type=int, default=100)
    p.add_argument('--native-records', type=int, default=20)
    p.add_argument('--with-native', action='store_true')
    p.add_argument('--with-spawn', action='store_true', help='Requires a dedicated empty CARLA server; includes native checks')
    p.add_argument('--with-physics', action='store_true', help='Separate multi-tick physics probe after raw spawn')
    p.add_argument('--probe-ticks',type=int,default=20)
    p.add_argument('--selection',choices=('prefix','stratified'),default='prefix')
    p.add_argument('--diagnostic-pair-report',type=Path,
                   help='Resume unique pairs only, explicitly marked incomplete diagnostic subset; no collection rescan')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=31000)
    p.add_argument('--timeout',type=float,default=300.,help='CARLA RPC timeout after service readiness')
    a = p.parse_args()
    if a.max_records <= 0 or not 0 < a.native_records <= a.max_records:
        p.error('Require 0 < native-records <= max-records')
    for path in (a.dataset_jsonl, a.generated_jsonl):
        if not path.is_file():
            raise FileNotFoundError(path)
    if a.raw is None:
        summary = a.dataset_jsonl.parent/'summary.json'
        original = Path(json.loads(summary.read_text(encoding='utf-8'))['args']['input_dir'])
        a.raw = original if original.is_dir() else root/'data/hv_init'/original.name
    paths = snapshot_paths(a.raw)
    print('RAW:', a.raw, 'JSON files:', len(paths), flush=True)
    out = a.output_dir or root/'data/hv_init/eval'/('snapshot_audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out = out.resolve()
    try:
        out.relative_to(a.raw.resolve())
    except ValueError:
        pass
    else:
        raise ValueError('Output must be outside snapshot input tree')
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    print('RESULT_DIR='+str(out), flush=True)
    config = root/'RIFT/rift/cbv/recognition/config/rule.yaml'
    meta = dict(vars(a), python=sys.executable, raw_json_files=len(paths), result_dir=out)
    (out/'run.json').write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
    def run(name, script, args):
        cmd = [sys.executable, '-u', str(root/script)] + [str(x) for x in args]
        with (out/'logs'/(name+'.log')).open('w', encoding='utf-8') as log:
            command = shlex.join(cmd)
            print(command, flush=True)
            log.write(command+'\n'); log.flush()
            process = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace')
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        if code:
            raise SystemExit('Stage %s failed (exit %d). Inspect/upload %s' % (name, code, out))
    audit = 'audit_hv_snapshot_experiment.py'
    if a.with_native or a.with_spawn or a.with_physics:
        run('native_environment','check_hv_native_environment.py',[])
    if not a.diagnostic_pair_report:
        run('pool', audit, ['--input-dir', a.raw, '--mode', 'snapshot', '--max-records', a.max_records,
                            '--output-dir', out/'pool100'])
    if a.with_native:
        raw_common=['--input-dir',a.raw,'--max-records',a.max_records,'--host',a.host,'--port',a.port,
                    '--timeout',a.timeout,
                    '--load-maps','--recognition-config',config,'--recognition-mode','eval',
                    '--sampling-resolution','1.0','--enrich-map']
        run('pool_native',audit,raw_common+['--mode','native','--output-dir',out/'pool_native100'])
    manifest = out/'v2_pairs100.jsonl'
    if a.diagnostic_pair_report:
        run('diagnostic_subset','prepare_hv_diagnostic_subset.py',[
            '--pair-report',a.diagnostic_pair_report,'--output-manifest',manifest])
    else:
        run('pair', 'build_hv_snapshot_audit_manifest.py', ['--dataset-jsonl', a.dataset_jsonl,
            '--generated-jsonl', a.generated_jsonl, '--snapshot-dir', a.raw,
            '--max-records', a.max_records,'--selection',a.selection,'--output-manifest', manifest])
    run('pair_snapshot', audit, ['--manifest', manifest, '--mode', 'snapshot', '--max-records', a.max_records,
                                 '--output-dir', out/'v2_snapshot100'])
    common = ['--manifest', manifest, '--max-records', a.native_records, '--host', a.host,
              '--port', a.port,'--timeout',a.timeout, '--load-maps', '--recognition-config', config,
              '--recognition-mode', 'eval', '--sampling-resolution', '1.0','--enrich-map']
    for mode, enabled in [('native', a.with_native), ('spawn', a.with_spawn),('physics',a.with_physics)]:
        if enabled:
            run(mode, audit, common+['--mode', mode,'--probe-ticks',a.probe_ticks,
                                     '--output-dir', out/('v2_'+mode+'20')])
    print('DONE. Upload this entire result directory: '+str(out), flush=True)


if __name__ == '__main__':
    main()
