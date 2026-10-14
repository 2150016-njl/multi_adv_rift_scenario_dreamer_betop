#!/usr/bin/env python3
"""Read-only PlanT asset preflight; no CARLA, downloads or model inference."""
import argparse
import json
import os
from pathlib import Path


def check_assets(checkpoint,hf_dir):
    checkpoint=Path(checkpoint);config=Path(hf_dir)/'config.json';errors=[]
    for path in (checkpoint,config):
        try:
            with path.open('rb') as f:
                if not f.read(1):errors.append('Empty asset: %s' % path)
        except OSError as exc:errors.append('%s: %s' % (path,exc))
    if config.is_file():
        try:
            value=json.loads(config.read_text(encoding='utf-8'))
            if not isinstance(value,dict) or not value.get('model_type'):
                errors.append('HF config lacks model_type: %s' % config)
        except (ValueError,OSError) as exc:errors.append('Invalid HF config: %s' % exc)
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parent
    p.add_argument('--checkpoint',type=Path,default=root/'RIFT/rift/ego/model_ckpt/PlanT_state_encoder/checkpoints/state_encoder.ckpt')
    p.add_argument('--hf-dir',type=Path,default=root/'RIFT/rift/ego/model_ckpt/PlanT_state_encoder/checkpoints/bert-medium')
    p.add_argument('--find',action='store_true',help='List possible assets in project, without auto-selecting weights')
    a=p.parse_args()
    if a.find:
        for current,dirs,files in os.walk(root):
            dirs[:]=[d for d in dirs if d not in ('.git','.task_tmp','__pycache__')]
            if 'state_encoder.ckpt' in files:print('checkpoint candidate:',Path(current)/'state_encoder.ckpt')
            if 'config.json' in files and 'bert' in str(current).lower():print('HF candidate:',current)
    errors=check_assets(a.checkpoint,a.hf_dir)
    print(json.dumps({'checkpoint':str(a.checkpoint),'hf_dir':str(a.hf_dir),'passed':not errors,
                      'errors':errors,'architecture_and_checkpoint_load_tested':False},indent=2))
    if errors:raise SystemExit('Missing/unreadable PlanT assets. Restore weights or set ATTN_CKPT / ATTN_HF_DIR before starting CARLA.')


if __name__=='__main__':main()
