#!/usr/bin/env python3
"""V7 first round: frozen protocol, map alignment, preliminary approach diagnosis."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scenario-dreamer'))
sys.path.insert(0,str(ROOT/'RIFT'))
from v7.io import initialize_run,read_jsonl,write_json,write_jsonl
from v7.pipeline import build,audit_dataset


def main():
    pre=argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config',type=Path)
    known,_=pre.parse_known_args()
    base=json.loads(known.config.read_text(encoding='utf-8')) if known.config else {}
    if not isinstance(base,dict):raise ValueError('Config must be an object')
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    allowed={'command'}
    for name in ('build','audit-map','audit-approaches','calibrate'):
        p=sub.add_parser(name)
        p.add_argument('--output-dir',type=Path,required=True)
        p.add_argument('--config',type=Path)
        p.add_argument('--seed',type=int,default=0)
        p.add_argument('--max-records',type=int,default=0)
        p.add_argument('--expected-input-sha256',type=Path)
        if name=='build':
            p.add_argument('--root',type=Path,default=ROOT)
            p.add_argument('--experiment',type=Path,required=True)
            p.add_argument('--dataset-jsonl',type=Path,required=True)
            p.add_argument('--snapshot-root',type=Path)
            p.add_argument('--record-indices',type=lambda s:[int(i) for i in s.split(',')])
            p.add_argument('--allow-ledger-reconstruction',action='store_true')
            p.add_argument(
                '--calibrated-map-dir',
                type=Path,
                default=None,
                help='Directory containing verified record_*_map.json files.'
            )

            p.add_argument(
                '--calibration-all-records',
                action='store_true',
                help='Prepare a map-only calibration plan for all selected records.'
            )
        elif name=='calibrate':
            p.add_argument('--plan',type=Path,required=True)
            p.add_argument('--port',type=int,required=True)
            p.add_argument('--host',default='127.0.0.1')
            p.add_argument('--timeout',type=float,default=30.)
            p.add_argument('--dedicated-server',action='store_true')
        else:p.add_argument('--dataset',type=Path,required=True)
        for action in p._actions:
            allowed.add(action.dest)
            if action.dest in base:
                value=base[action.dest]
                if value is None and action.required:raise ValueError('Required config field is null: '+action.dest)
                if value is not None and action.type and isinstance(value,str):value=action.type(value)
                action.default=value
                if action.required and value is not None:action.required=False
    if set(base)-allowed:raise ValueError('Unknown config keys: '+str(sorted(set(base)-allowed)))
    args=parser.parse_args()
    if base.get('command',args.command)!=args.command:parser.error('Config command differs from CLI command')
    if args.max_records<0:parser.error('--max-records must be nonnegative')
    global ERROR_OUTPUT_DIRECTORY
    ERROR_OUTPUT_DIRECTORY=args.output_dir if not args.output_dir.exists() else None
    config=base.copy()
    config.update({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    if args.command=='build':
        result=build(args.root,args.experiment,args.dataset_jsonl,args.output_dir,config)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.command=='audit-approaches':audit_dataset(args.dataset,args.output_dir,config)
    elif args.command=='calibrate':
        from v7.calibration import calibrate
        calibrate(args.plan,args.output_dir,args.host,args.port,args.timeout,args.dedicated_server,config)
    else:
        from v7.coordinates import world_to_local
        import numpy as np
        inputs={'conditions.jsonl':args.dataset/'conditions.jsonl'}
        declared=json.loads((args.dataset/'protocol_sha256.json').read_text(encoding='utf-8'))
        from v7.io import sha
        if sha(inputs['conditions.jsonl'])!=declared['conditions.jsonl']:raise ValueError('Condition SHA mismatch')
        initialize_run(args.output_dir,config,inputs,args.expected_input_sha256)
        rows=[]
        records=read_jsonl(inputs['conditions.jsonl'])
        if args.max_records:records=records[:args.max_records]
        for record in records:
            from v7.schema import validate_condition
            validate_condition(record)
            c=record['condition'];errors=[]
            for lane in c['directed_map']['lanes']:
                errors.append(float(np.linalg.norm(np.asarray(world_to_local(c['ego_world_pose'],lane['centerline_world_xyz']))-
                                             np.asarray(lane['centerline_local_xyz']),axis=1).max()))
            rows.append({'record_id':record['record_id'],'coordinate_max_error_m':max(errors,default=None),
                         'topology_complete':c['directed_map']['topology_complete'],'native_replay':'not_run'})
        write_jsonl(args.output_dir/'rows.jsonl',rows)
        write_json(args.output_dir/'summary.json',{'records':len(rows),'native_replay':'not_run','fixed_sedan':'not_run'})
    print('RESULT_DIR='+str(args.output_dir),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        directory=globals().get('ERROR_OUTPUT_DIRECTORY')
        if directory is not None and directory.is_dir():
            with (directory/'errors.log').open('a',encoding='utf-8') as stream:
                stream.write('%s: %s\n'%(type(exc).__name__,exc))
            if not (directory/'summary.json').exists():
                write_json(directory/'summary.json',{'status':'failed','error':str(exc),
                                                    'p0_complete':False,'p1_complete':False,'p2_complete':False})
        raise
