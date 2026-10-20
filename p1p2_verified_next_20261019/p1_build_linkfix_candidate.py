#!/usr/bin/env python3
"""Create immutable P1 V2 linkfix capsule with source-SHA binding and real module import."""
from __future__ import annotations
import argparse, hashlib, os, shutil, subprocess, sys, json
from pathlib import Path

ORIGINAL_CONTRACT_LINE = 'linked,link_status=directed_link(route[-1],anchor)'
NEW_CONTRACT_LINE = 'linked,link_status=directed_link(route[-1],anchor,cmap=cmap)'
ORIGINAL_BRANCH = '    # Cross-road or cross-lane requires proof from directed CARLA next().\n'
NEW_BRANCH = '''    # A finite 7-step CARLA waypoint.next search is not a topology proof.
    # Prefer explicit OpenDRIVE lane/junction links; preserve fail-closed if absent.
    if cmap is not None:
        from .p1_od_link_guard import verify_pair
        od_proof = verify_pair(cmap.to_opendrive(), a, b, allowed_distance_m)
        if od_proof['status'] == 'verified':
            return True, 'explicit_xodr_' + od_proof['link_kind']
'''

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--python',required=True);args=p.parse_args()
    parent=Path(args.parent).resolve();dest=Path(args.output).resolve();our=Path(__file__).resolve().parent
    if dest.exists():raise SystemExit('REFUSE_OVERWRITE_EXISTING_CAPSULE')
    sub=parent/'hv_v8';base=sub/'p1_native_audit.py';guard=sub/'p1_science_contract_v2.py'
    if not base.is_file() or not guard.is_file():raise SystemExit('PARENT_MODULES_MISSING')
    if sha(base)!='62b4bbc1bae838cd13185586c546d756a394cd39ae57b35a63242e96aa56609a':
        raise SystemExit('PARENT_SCIENCE_SOURCE_SHA_UNEXPECTED')
    ss=guard.read_text();aa=base.read_text()
    if ss.count(ORIGINAL_BRANCH)!=1 or ss.count(ORIGINAL_CONTRACT_LINE)!=1:
        raise SystemExit('UNEXPECTED_SOURCE_CONTRACT_REFUSE_PATCH')
    if ss.count('def directed_link(a,b,allowed_distance_m=5.0):')!=1:
        raise SystemExit('DIRECTED_LINK_SIGNATURE_CHANGED')
    ss=ss.replace('def directed_link(a,b,allowed_distance_m=5.0):',
                  'def directed_link(a,b,allowed_distance_m=5.0,cmap=None):')
    ss=ss.replace(ORIGINAL_BRANCH,NEW_BRANCH+ORIGINAL_BRANCH)
    ss=ss.replace(ORIGINAL_CONTRACT_LINE,NEW_CONTRACT_LINE)
    # Fail-closed on missing explicit connectivity: the original bounded next()
    # branch still exists for CARLA-direct evidence, not to force a PASS.
    dest.mkdir(parents=True)
    try:
        shutil.copytree(sub,dest/'hv_v8',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (dest/'hv_v8/p1_science_contract_v2.py').write_text(ss)
        (dest/'hv_v8/p1_od_link_guard.py').write_bytes((our/'p1_od_link_guard.py').read_bytes())
        paths=sorted((dest/'hv_v8').glob('*.py'))
        provenance={'schema':'p1_explicit_xodr_link_candidate_v1','parent':str(parent),
                    'source_parent_sha256':sha(base),
                    'candidate_sha256':sha(dest/'hv_v8/p1_native_audit.py'),
                    'files':{x.name:sha(x) for x in paths},
                    'known_status':'not_CARLA_verified'}
        # Keep the exact supervisor's immutable dependency contract, updated only
        # for this verified new helper; original audit source SHA stays unchanged.
        original_prov=parent/'p1_source_provenance.json'
        if not original_prov.is_file():raise RuntimeError('missing parent provenance')
        expected=json.loads(original_prov.read_text())
        if expected.get('dependency_auditor_sha256')!=sha(base):
            raise RuntimeError('parent provenance auditor mismatch')
        expected['dependency_science_helper_sha256']=sha(dest/'hv_v8/p1_science_contract_v2.py')
        expected['dependency_od_link_guard_sha256']=sha(dest/'hv_v8/p1_od_link_guard.py')
        expected['candidate_unverified_in_real_CARLA']=True
        (dest/'p1_source_provenance.json').write_text(json.dumps(expected,indent=2)+'\n')
        (dest/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        runner=Path(args.parent).resolve().parents[3] / 'p1_science_v2_release/hv_v8/p1_isolated95_science_v2.py'
        if not runner.is_file():raise RuntimeError('pinned parent supervisor missing: '+str(runner))
        runner_source=runner.read_text()
        old_cap="CAPSULE='p1_frozen_capsule_science_v2_20261019'"
        new_cap="CAPSULE='"+dest.name+"'"
        if runner_source.count(old_cap)!=1:raise RuntimeError('unexpected parent runner CAPSULE binding')
        anchor="    cfg=root/'RIFT/rift/cbv/recognition/config/attention.yaml'"
        if runner_source.count(anchor)!=1:raise RuntimeError('unexpected supervisor dependency-check location')
        guard_check=("    need(sha(cap/'hv_v8/p1_od_link_guard.py')==prov['dependency_od_link_guard_sha256'],\n"
                     "         'OpenDRIVE link guard SHA changed')\n")
        runner_source=runner_source.replace(anchor,guard_check+anchor)
        (dest/'p1_isolated95_linkfix.py').write_text(runner_source.replace(old_cap,new_cap))
        # Exact python -m entry, in a separate interpreter with import origin checked.
        env=os.environ.copy();env['PYTHONPATH']=str(dest)+os.pathsep+env.get('PYTHONPATH','')
        subprocess.run([args.python,'-m','hv_v8.p1_native_audit','--help'],cwd=str(dest),env=env,
                       stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=35,check=True)
    except Exception:
        shutil.rmtree(dest,ignore_errors=True)
        raise
    print('P1_LINKFIX_CAPSULE='+str(dest))
    print('P1_LINKFIX_SOURCE_SHA256='+provenance['candidate_sha256'])
    print('P1_LINKFIX_CONTRACT_SHA256='+sha(dest/'hv_v8/p1_science_contract_v2.py'))
    print('P1_LINKFIX_FULL_IMPORT_VERIFIED_NOT_CARLA_PASS')
if __name__=='__main__':main()
