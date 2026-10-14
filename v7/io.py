import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_jsonl(path,rows):
    with Path(path).open('x',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def initialize_run(output,config,inputs,expected=None):
    output=Path(output)
    hashes={name:sha(path) for name,path in inputs.items()}
    if expected and json.loads(Path(expected).read_text(encoding='utf-8'))!=hashes:
        raise ValueError('Input SHA manifest differs')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',config)
    write_json(output/'input_sha256.json',hashes)
    result=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
    (output/'git_commit.txt').write_text(result.stdout.strip() or 'unverified',encoding='utf-8')
    root=Path(__file__).resolve().parents[1]
    code=list((root/'v7').glob('*.py'))+[root/'v7_cli.py',root/'run_hv_native_diagnostics.py',
            root/'run_hv_native_headless.sh',root/'run_hv_v7_calibration_headless.sh',
            root/'scenario-dreamer/high_value_init/native_state_audit.py',
            root/'RIFT/rift/cbv/recognition/base_cbv.py',
            root/'RIFT/rift/scenario/tools/global_route_planner.py']
    write_json(output/'code_sha256.json',{str(p.relative_to(root)):sha(p) for p in code if p.is_file()})
    (output/'errors.log').write_text('',encoding='utf-8')
    return hashes


def resolve_source(source,root,snapshot_root=None):
    source=source.replace('\\','/')
    options=[Path(source)]
    if '/data/' in source:options.append(Path(root)/'data'/source.split('/data/',1)[1])
    for directory in ('RIFT','scratch','metadata'):
        marker='/'+directory+'/'
        if marker in source:options.append(Path(root)/directory/source.split(marker,1)[1])
    if snapshot_root:
        parts=source.replace('\\','/').split('/')
        options.append(Path(snapshot_root)/parts[-2]/parts[-1])
    return next((p for p in options if p.is_file()),None)
