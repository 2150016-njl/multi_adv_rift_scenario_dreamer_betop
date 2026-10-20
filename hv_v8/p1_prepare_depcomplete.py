#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repackage unchanged frozen P1 auditor with all local Python dependencies.

No CARLA access; never modify an existing capsule. Fail closed on source hashes.
"""
from __future__ import annotations
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

CHILD_NAME = 'p1_frozen_capsule_world_transition_depcomplete_v2_20261019'
OLD_NAME = 'p1_frozen_capsule_world_transition_v1_20261019'
PARENT_NAME = 'p1_frozen_capsule_occupancy_joint_20261019_034017_031802'
AUDITOR_SHA = 'b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'
GUARD_SHA = '5e7cd2742eff63f714ba2236738d28a70eaef70b3c45ad8c99fc5d7299b0a35b'
PARENT_SHA = 'ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def main(root):
    root = Path(root).resolve()
    base = root/'data/hv_init/v7'
    old = base/OLD_NAME
    parent = base/PARENT_NAME
    new = base/CHILD_NAME
    orig_aud = old/'hv_v8/p1_native_audit.py'
    orig_guard = old/'hv_v8/p1_world_guard.py'
    parent_aud = parent/'hv_v8/p1_native_audit.py'
    common_sources = [parent/'hv_v8/common.py', root/'hv_v8/common.py']
    common_sources = [p for p in common_sources if p.is_file()]
    require(common_sources, 'COMMON_MODULE_NOT_FOUND in parent capsule or root/hv_v8')
    common = common_sources[0]
    require(sha(orig_aud) == AUDITOR_SHA, 'WORLD_FIX_SOURCE_SHA_MISMATCH')
    require(sha(orig_guard) == GUARD_SHA, 'WORLD_FIX_GUARD_SHA_MISMATCH')
    require(sha(parent_aud) == PARENT_SHA, 'PARENT_AUDITOR_SHA_MISMATCH')
    require(not new.exists(), 'REFUSE_OVERWRITE_EXISTING_CAPSULE: ' + str(new))
    # This is the exact set of local relative modules imported by the auditor.
    mod = ast.parse(orig_aud.read_text(encoding='utf-8'))
    rel = sorted(set(n.module for n in ast.walk(mod)
                     if isinstance(n, ast.ImportFrom) and n.level == 1))
    require(rel == ['common', 'p1_world_guard'], 'UNREVIEWED_RELATIVE_IMPORTS: {}'.format(rel))
    cmod = ast.parse(common.read_text(encoding='utf-8'))
    names = {n.name for n in cmod.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for name in ('ContractError','lane_uid_str','read_json','read_jsonl', 'require',
                 'sha256_bytes','write_json','write_jsonl','yaw_wrap'):
        require(name in names, 'COMMON_MISSING_REQUIRED_SYMBOL: '+name)
    # Do not silently accept a common.py which itself imports another local module.
    extra = [(n.level, n.module) for n in ast.walk(cmod)
             if isinstance(n, ast.ImportFrom) and n.level]
    require(not extra, 'COMMON_HAS_UNBUNDLED_RELATIVE_IMPORTS: '+str(extra))
    prov_file = old/'p1_source_provenance.json'
    require(prov_file.is_file(), 'MISSING_WORLD_FIX_PROVENANCE')
    prov = json.loads(prov_file.read_text(encoding='utf-8'))
    require(prov.get('frozen_sha256') == AUDITOR_SHA, 'WORLD_FIX_PROVENANCE_SHA_MISMATCH')
    # Stage a complete capsule atomically; do not change old or parent.
    temp = Path(tempfile.mkdtemp(prefix='.'+CHILD_NAME+'_stage_', dir=str(base)))
    try:
        pkg = temp/'hv_v8'
        pkg.mkdir()
        (pkg/'__init__.py').write_text('# Isolated audited package.\n', encoding='utf-8')
        shutil.copyfile(orig_aud, pkg/'p1_native_audit.py')
        shutil.copyfile(orig_guard, pkg/'p1_world_guard.py')
        shutil.copyfile(common, pkg/'common.py')
        prov['parent_capsule'] = OLD_NAME
        prov['frozen_sha256'] = AUDITOR_SHA
        prov['dependency_complete_capsule'] = True
        prov['dependency_source_common'] = str(common)
        prov['dependency_common_sha256'] = sha(common)
        prov['dependency_guard_sha256'] = GUARD_SHA
        prov['dependency_auditor_sha256'] = AUDITOR_SHA
        prov['scientific_source_unchanged'] = True
        prov['bundle_files_sha256'] = {p.name: sha(p) for p in pkg.iterdir() if p.is_file()}
        (temp/'p1_source_provenance.json').write_text(
            json.dumps(prov, ensure_ascii=False, sort_keys=True, indent=2)+'\n', encoding='utf-8')
        require(sha(pkg/'p1_native_audit.py') == AUDITOR_SHA, 'STAGING_AUDITOR_CHANGED')
        require(sha(pkg/'p1_world_guard.py') == GUARD_SHA, 'STAGING_GUARD_CHANGED')
        require(sha(pkg/'common.py') == sha(common), 'STAGING_COMMON_CHANGED')
        os.rename(str(temp), str(new))
    except BaseException:
        shutil.rmtree(str(temp), ignore_errors=True)
        raise
    print('P1_COMPLETE_CAPSULE='+str(new))
    print('P1_AUDITOR_UNCHANGED_SHA256='+AUDITOR_SHA)
    print('P1_COMMON_SOURCE='+str(common))
    print('P1_COMMON_SHA256='+sha(common))
    print('P1_DEPENDENCY_BUNDLE_CREATED_NOT_CARLA_PASS')

if __name__ == '__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Usage: p1_prepare_depcomplete.py ROOT')
    main(sys.argv[1])
