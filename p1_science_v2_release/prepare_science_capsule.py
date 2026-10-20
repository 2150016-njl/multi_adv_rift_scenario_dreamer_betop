#!/usr/bin/env python3
"""Install reproducible P1 science candidate as new immutable capsule.
No overwriting of any original source/capsule. Checks actual server frozen parent.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
PARENT='b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'
CHILD='62b4bbc1bae838cd13185586c546d756a394cd39ae57b35a63242e96aa56609a'
COMMON='d89a74a0d4cd41f917cbd85dac37754392240a89252efa721dfc12d230173273'
GUARD='5e7cd2742eff63f714ba2236738d28a70eaef70b3c45ad8c99fc5d7299b0a35b'
BASENAME='p1_frozen_capsule_science_v2_20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(root,source):
    root=Path(root).resolve();src=Path(source).resolve()
    base=root/'data/hv_init/v7';orig=base/'p1_frozen_capsule_world_transition_depcomplete_v2_20261019/hv_v8'
    assert sha(orig/'p1_native_audit.py')==PARENT,'server parent audit changed'
    assert sha(orig/'common.py')==COMMON,'server pinned common changed'
    assert sha(orig/'p1_world_guard.py')==GUARD,'server guard changed'
    assert sha(src/'hv_v8/p1_native_audit.py')==CHILD,'candidate audit changed'
    assert sha(src/'hv_v8/p1_world_guard.py')==GUARD,'candidate guard changed'
    target=base/BASENAME
    if target.exists():
        raise FileExistsError('new capsule exists; refuse overwrite '+str(target))
    staging=base/(BASENAME+'.staging')
    if staging.exists():raise FileExistsError('existing staging dir '+str(staging))
    staging.mkdir(parents=True)
    try:
        hv=staging/'hv_v8';hv.mkdir()
        for fname in ('__init__.py','p1_native_audit.py','p1_science_contract_v2.py','p1_physics_independent.py','p2_prepare_town.py'):
            shutil.copy2(src/'hv_v8'/fname,hv/fname)
        for fname in ('common.py','p1_world_guard.py'):
            shutil.copy2(orig/fname,hv/fname)
        prov={'schema_version':'p1_science_v2_capsule_provenance','dependency_complete_capsule':True,
           'parent_auditor_sha256':PARENT,'dependency_auditor_sha256':sha(hv/'p1_native_audit.py'),
           'dependency_common_sha256':sha(hv/'common.py'),
           'dependency_guard_sha256':sha(hv/'p1_world_guard.py'),
           'dependency_science_helper_sha256':sha(hv/'p1_science_contract_v2.py'),
           'dependency_physics_independent_sha256':sha(hv/'p1_physics_independent.py'),
           'dependency_p2_town_prep_sha256':sha(hv/'p2_prepare_town.py'),
           'changed_science':['route_source_and_directed_continuity','RIFT_3d_ahead_and_half_FOV',
                'finite_junction_angle_in_degrees','raw_spawn_full_3d_pose'],
           'full_P1_proven':False,'Model3_full_scene_training_ready':False}
        (staging/'p1_source_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
        staging.rename(target)
        print('P1_SCIENCE_CANDIDATE_CAPSULE='+str(target))
        print('P1_CANDIDATE_SHA='+CHILD)
        print('P1_PARENT_UNTOUCHED='+PARENT)
        return target
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--source',default=str(Path(__file__).resolve().parent));a=ap.parse_args()
    prepare(a.root,a.source)
if __name__=='__main__':main()
