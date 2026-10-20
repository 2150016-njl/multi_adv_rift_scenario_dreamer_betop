#!/usr/bin/env python3
"""Create a NEW immutable candidate capsule from known server-source SHA.

Does not connect to CARLA, change source data, or touch existing capsules.
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

PARENT_NAME = 'p1_frozen_capsule_occupancy_joint_20261019_034017_031802'
CHILD_NAME = 'p1_frozen_capsule_world_transition_v1_20261019'
PARENT_SHA = 'ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f'
CHILD_SHA = 'b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(root):
    root = Path(root).resolve()
    pkg = Path(__file__).resolve().parent
    base = root / 'data/hv_init/v7'
    parent = base / PARENT_NAME
    child = base / CHILD_NAME
    orig = parent / 'hv_v8/p1_native_audit.py'
    new = pkg / 'p1_native_audit_worldfix.py'
    guard = pkg / 'p1_world_guard.py'
    manifest = base / 'manifests/p1_calibration95.jsonl'
    map_manifest = base / 'manifests/p1_map_identity_manifest_v2_20261017_154737.json'
    assert sha(orig) == PARENT_SHA, 'parent frozen source SHA mismatch'
    assert sha(new) == CHILD_SHA, 'candidate source SHA mismatch'
    assert sha(manifest) == '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
    assert sha(map_manifest) == '9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
    assert (parent / 'p1_source_provenance.json').is_file(), 'missing parent provenance'
    assert not child.exists(), 'refuse overwrite existing capsule: '+str(child)
    (child / 'hv_v8').mkdir(parents=True, exist_ok=False)
    shutil.copyfile(new, child / 'hv_v8/p1_native_audit.py')
    shutil.copyfile(guard, child / 'hv_v8/p1_world_guard.py')
    provenance = json.loads((parent/'p1_source_provenance.json').read_text())
    provenance['frozen_sha256'] = CHILD_SHA
    provenance['parent_frozen_sha256'] = PARENT_SHA
    provenance['upstream_parent_source_provenance_sha256'] = provenance.get('parent_source_provenance_sha256')
    provenance['parent_source_provenance_sha256'] = sha(parent/'p1_source_provenance.json')
    provenance.setdefault('reconciliation_steps', []).append('P1_WORLD_EPISODE_FENCE_V1')
    provenance['fix_scope'] = 'world episode ID/Town/OpenDRIVE SHA readiness; no route/native/physics threshold changes'
    provenance['map_transition_guard_sha256'] = sha(child/'hv_v8/p1_world_guard.py')
    provenance['generator_script_sha256'] = sha(Path(__file__).resolve())
    (child/'p1_source_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    assert sha(child / 'hv_v8/p1_native_audit.py') == CHILD_SHA
    print('P1_NEW_CAPSULE='+str(child))
    print('P1_NEW_SOURCE_SHA256='+CHILD_SHA)
    print('P1_PARENT_SOURCE_SHA256='+PARENT_SHA)
    print('P1_NO_ORIGINAL_CAPSULE_MODIFIED=True')

if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: p1_prepare_worldfix.py ROOT')
    main(sys.argv[1])
