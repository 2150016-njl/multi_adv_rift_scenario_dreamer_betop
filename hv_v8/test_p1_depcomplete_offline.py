#!/usr/bin/env python3
"""Integration tests of real, unchanged auditor module packaging and entrypoint.

No CARLA invocation. The server runner separately probes the REAL child environment.
"""
from __future__ import annotations
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p1_prepare_depcomplete as d

# Test helper implements precisely the exposed common API for import only.
# Full science calibration is NOT validated by these offline tests.
COMMON_TEST_FIXTURE = '''from __future__ import annotations
import math, json, hashlib
from pathlib import Path
class ContractError(RuntimeError): pass
def require(x, msg):
    if not x: raise ContractError(msg)
def lane_uid_str(x): return ":".join(map(str,x))
def read_json(p): return json.loads(Path(p).read_text())
def read_jsonl(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def write_json(p,o): Path(p).write_text(json.dumps(o))
def write_jsonl(p,rs): Path(p).write_text("\\n".join(json.dumps(r) for r in rs))
def yaw_wrap(rad): return (rad+math.pi)%(math.pi*2)-math.pi
'''

def require(x, msg):
    if not x: raise AssertionError(msg)

def make_root(root):
    base = root/'data/hv_init/v7'
    old = base/d.OLD_NAME
    parent = base/d.PARENT_NAME
    (old/'hv_v8').mkdir(parents=True)
    (parent/'hv_v8').mkdir(parents=True)
    shutil.copyfile(HERE/'p1_native_audit_worldfix.py', old/'hv_v8/p1_native_audit.py')
    shutil.copyfile(HERE/'p1_world_guard.py', old/'hv_v8/p1_world_guard.py')
    # The test fixture uses the exact current frozen audit source for the
    # parent as well. Override its parent hash in this ephemeral MOCK ROOT;
    # the actual prepare script still pins the real old-parent hash.
    shutil.copyfile(HERE/'p1_native_audit_worldfix.py', parent/'hv_v8/p1_native_audit.py')
    d.PARENT_SHA = d.AUDITOR_SHA
    (parent/'hv_v8/common.py').write_text(COMMON_TEST_FIXTURE, encoding='utf-8')
    (old/'p1_source_provenance.json').write_text(json.dumps({'frozen_sha256': d.AUDITOR_SHA}))
    return base

def python_cmd(path, cp):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(cp)
    r = subprocess.run([sys.executable, '-m','hv_v8.p1_native_audit','--help'],
                       cwd=str(cp), env=env, text=True, capture_output=True)
    return r

def main():
    with tempfile.TemporaryDirectory(prefix='p1depcomplete_') as td:
        root = Path(td)
        base = make_root(root)
        d.main(root)
        cp = base/d.CHILD_NAME
        pkg = cp/'hv_v8'
        prov = json.loads((cp/'p1_source_provenance.json').read_text())
        require(prov['dependency_complete_capsule'], 'dep_complete marker')
        require(prov['dependency_common_sha256'] == d.sha(pkg/'common.py'), 'common sha')
        require(d.sha(pkg/'p1_native_audit.py') == d.AUDITOR_SHA, 'auditor must be unchanged')
        require(d.sha(pkg/'p1_world_guard.py') == d.GUARD_SHA, 'guard must be unchanged')
        require((pkg/'__init__.py').exists(), 'explicit package init')
        print('TEST_UNCHANGED_AUDITOR_AND_HASHED_DEPENDENCIES=PASS')

        # Actual frozen auditor, actual -m entrypoint and relative imports,
        # the very execution form that just failed on the server.
        r = python_cmd(pkg/'p1_native_audit.py', cp)
        require(r.returncode==0 and '--manifest' in r.stdout,
                'real package -m import failed: '+r.stderr[-3000:])
        code = ('import hv_v8.p1_native_audit as a;'
                'import hv_v8.common as c;'
                'import hv_v8.p1_world_guard as g;'
                'print(a.__file__);print(c.__file__);print(g.__file__)')
        env = os.environ.copy(); env['PYTHONPATH'] = str(cp)
        r2 = subprocess.run([sys.executable,'-c',code],cwd=str(cp),env=env,
                            capture_output=True,text=True)
        require(r2.returncode==0, 'package import failed: '+r2.stderr)
        require(all(str((pkg/name).resolve()) in r2.stdout for name in
                    ('p1_native_audit.py','common.py','p1_world_guard.py')), 'wrong package origin')
        print('TEST_REAL_FROZEN_AUDITOR_MINUS_M_CLI_IMPORT_AND_ORIGIN=PASS')

        (pkg/'common.py').unlink()
        r3=python_cmd(pkg/'p1_native_audit.py',cp)
        require(r3.returncode != 0 and 'hv_v8.common' in r3.stderr,
                'missing common must cause visible error')
        print('TEST_MISSING_COMMON_REPRODUCES_SERVER_FAILURE=PASS')
        (pkg/'common.py').write_text(COMMON_TEST_FIXTURE + '\n#tampered\n')
        require(d.sha(pkg/'common.py') != prov['dependency_common_sha256'],
                'tampered common must mismatch provenance')
        print('TEST_TAMPERED_COMMON_DETECTABLE_BY_SHA=PASS')
        try:
            d.main(root)
        except RuntimeError as e:
            require('REFUSE_OVERWRITE' in str(e), 'unexpected duplicate capsule error')
        else:
            raise AssertionError('overwrote capsule')
        print('TEST_NO_OVERWRITE_OF_EXISTING_CAPSULE=PASS')

    print('P1_DEP_COMPLETE_OFFLINE_TESTS_PASS_NOT_CARLA')

if __name__=='__main__':
    main()
