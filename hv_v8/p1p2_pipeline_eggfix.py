#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1 95-record audit launcher / independent ledger / P2 supervision export.

Python 3.8+, stdlib. No changes to snapshots, frozen P1 code or scoring thresholds.
`inputs`, `report`, `p2-export` never contact CARLA.
`run95` explicitly changes the dedicated CARLA world via the EXISTING frozen auditor.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

FROZEN_SHA = 'ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f'
MANIFEST_SHA = '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
MAP_MANIFEST_SHA = '9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'
CAPSULE_NAME = 'p1_frozen_capsule_occupancy_joint_20261019_034017_031802'
MAP_MANIFEST_NAME = 'p1_map_identity_manifest_v2_20261017_154737.json'

class EvidenceError(RuntimeError):
    pass

def ensure(condition, msg):
    if not condition:
        raise EvidenceError(msg)

def shafile(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write_new_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    return str(path.resolve())

def expected_paths(root):
    root = Path(root).resolve()
    base = root / 'data/hv_init/v7'
    return {
        'root': root,
        'base': base,
        'manifest': base / 'manifests/p1_calibration95.jsonl',
        'map_manifest': base / 'manifests' / MAP_MANIFEST_NAME,
        'capsule': base / CAPSULE_NAME,
        'source': base / CAPSULE_NAME / 'hv_v8/p1_native_audit.py',
        'config': root / 'RIFT/rift/cbv/recognition/config/attention.yaml',
        'python_api': root / 'carla/PythonAPI/carla',
    }

def fixed_inputs(root, verify_snapshots=True):
    p = expected_paths(root)
    for k in ('manifest', 'map_manifest', 'source'):
        ensure(p[k].is_file(), 'missing ' + k + ': ' + str(p[k]))
    for k, expected in (('manifest', MANIFEST_SHA), ('map_manifest', MAP_MANIFEST_SHA), ('source', FROZEN_SHA)):
        actual = shafile(p[k])
        ensure(actual == expected, '{} SHA mismatch actual={} expected={}'.format(k, actual, expected))
    provenance_path = p['capsule'] / 'p1_source_provenance.json'
    ensure(provenance_path.is_file(), 'capsule provenance missing')
    prov = read_json(provenance_path)
    ensure(prov.get('frozen_sha256') == FROZEN_SHA, 'capsule provenance frozen SHA mismatch')
    ensure(p['config'].is_file(), 'RIFT config missing')
    ensure(p['python_api'].is_dir(), 'CARLA PythonAPI missing')
    map_doc = read_json(p['map_manifest'])
    ensure(map_doc.get('schema_version') == 'p1_map_identity_manifest_v1', 'map manifest schema mismatch')
    maps = {}
    for entry in map_doc.get('maps', []):
        town = str(entry.get('town', '')).replace('\\', '/').split('/')[-1]
        ensure(town and town not in maps, 'duplicate/missing Town in map manifest')
        maps[town] = entry.get('opendrive_sha256')
    ensure(set(maps) == {'Town04', 'Town11', 'Town12', 'Town13', 'Town15'}, 'unexpected map identity town set')
    raw_rows = [x for x in p['manifest'].read_text(encoding='utf-8').splitlines() if x.strip()]
    ensure(len(raw_rows) == 95, 'manifest row count != 95')
    rows = [json.loads(x) for x in raw_rows]
    ensure(all(isinstance(row, dict) for row in rows), 'manifest rows not dict')
    indices = [int(row['record_index']) for row in rows]
    ensure(len(indices) == len(set(indices)), 'duplicate record_index')
    diagnostics = []
    scene_ids = []
    seen_snapshot_paths = set()
    town_counter = collections.Counter()
    error_count = 0
    for ordinal, row in enumerate(rows):
        snapshot_path = Path(row['snapshot_path'])
        ensure(snapshot_path.is_file(), 'missing snapshot ordinal={} path={}'.format(ordinal, snapshot_path))
        path_key = str(snapshot_path.resolve())
        ensure(path_key not in seen_snapshot_paths, 'duplicate snapshot file ' + path_key)
        seen_snapshot_paths.add(path_key)
        ensure(isinstance(row.get('scene_id'), str) and row['scene_id'], 'empty manifest scene ID')
        scene_ids.append(row['scene_id'])
        snapshot_sha = shafile(snapshot_path)
        declared_sha = row.get('snapshot_sha256')
        if declared_sha is not None:
            ensure(snapshot_sha == declared_sha, 'snapshot hash mismatch ordinal={}'.format(ordinal))
        scene = read_json(snapshot_path)
        scene_id = scene.get('scene_id')
        town = str(scene.get('town') or scene.get('map_name') or '').replace('\\', '/').split('/')[-1]
        ensure(town in maps, 'missing / unknown Town ordinal={} town={}'.format(ordinal, town))
        town_counter[town] += 1
        identity = 'exact' if row['scene_id'] == scene_id else 'unresolved_difference'
        route = scene.get('ego_route')
        agents, labels = scene.get('agents'), scene.get('rift_labels')
        problems = []
        if identity != 'exact':
            problems.append('manifest_scene_id_differs_from_snapshot_scene_id')
        if not isinstance(route, list) or len(route) < 2:
            problems.append('ego_route_missing_or_short')
        if not isinstance(agents, list) or not isinstance(labels, list):
            problems.append('agents_or_rift_labels_missing')
            gt_ids = []
        else:
            try:
                ids = [str(a['actor_id']) for a in agents]
                lids = [str(l['actor_id']) for l in labels]
                ensure(len(set(ids)) == len(ids) and len(set(lids)) == len(lids) and set(ids) == set(lids), 'id_drift')
                ensure(all(isinstance(l.get('is_interactive'), bool) for l in labels), 'label_nonbool')
                gt_ids = [str(l['actor_id']) for l in labels if l['is_interactive']]
                if not gt_ids:
                    problems.append('no_positive_rift_label')
            except (KeyError, TypeError, EvidenceError):
                gt_ids = []
                problems.append('agents_labels_not_identity_verifiable')
        error_count += len(problems)
        diagnostics.append({
            'ordinal': ordinal, 'record_index': int(row['record_index']),
            'manifest_scene_id': row['scene_id'], 'snapshot_scene_id': scene_id,
            'scene_identity': identity, 'town': town, 'snapshot_sha256': snapshot_sha,
            'snapshot_path': str(snapshot_path), 'route_length': len(route) if isinstance(route, list) else None,
            'native_label_positive_count': len(gt_ids), 'native_positive_actor_ids': gt_ids,
            'problems': problems,
        })
    # A manifest scene_id is a source label, not the unique audit storage key.
    # A repeated label must NOT cause 95 fixed records to be silently dropped,
    # nor prevent diagnostics. Keep it unresolved until source provenance proves
    # what the two underlying observations represent.
    by_scene_id = collections.defaultdict(list)
    for d in diagnostics:
        by_scene_id[d['manifest_scene_id']].append(d)
    for sid, grouped in by_scene_id.items():
        if len(grouped) > 1:
            for d in grouped:
                d['scene_id_globally_unique'] = False
                d['problems'].append('duplicate_manifest_scene_id_unresolved')
                error_count += 1
        else:
            grouped[0]['scene_id_globally_unique'] = True
    return p, rows, diagnostics, dict(sorted(town_counter.items())), error_count

def scene_id_duplicate_groups(diagnostics):
    groups = collections.defaultdict(list)
    for d in diagnostics:
        groups[d['manifest_scene_id']].append(d)
    return [
        {'manifest_scene_id': sid, 'rows': [
            {'ordinal': x['ordinal'], 'record_index': x['record_index'],
             'snapshot_path': x['snapshot_path'],
             'snapshot_sha256': x['snapshot_sha256'],
             'snapshot_scene_id': x['snapshot_scene_id']}
            for x in members]}
        for sid, members in sorted(groups.items()) if len(members) > 1
    ]

def cmd_inputs(args):
    _, _, diagnostics, towns, error_count = fixed_inputs(args.root)
    mismatches = [x for x in diagnostics if x['scene_identity'] != 'exact']
    duplicates = scene_id_duplicate_groups(diagnostics)
    result = {
        'schema_version': 'p1_inputs_95_v1', 'records': len(diagnostics),
        'town_counts_snapshot': towns, 'scene_id_discrepancy_count': len(mismatches),
        'scene_id_discrepancies': mismatches, 'problems_total': error_count,
        'scene_id_duplicate_groups': duplicates,
        'scientific_identity_all_exact': not bool(mismatches) and not bool(duplicates),
        'source_frozen_sha256': FROZEN_SHA,
        'note': 'composite :sparse0 ID is NOT silently normalized into PASS; inspect generator provenance separately',
    }
    print('P1_INPUT_RECORDS=95')
    print('P1_TOWNS=' + json.dumps(towns, sort_keys=True))
    print('P1_SCENE_ID_UNRESOLVED=' + json.dumps([{'ordinal':x['ordinal'],'manifest':x['manifest_scene_id'],'snapshot':x['snapshot_scene_id']} for x in mismatches], ensure_ascii=False))
    print('P1_DUPLICATE_SCENE_ID_GROUPS=' + json.dumps(duplicates, ensure_ascii=False))
    print('P1_INPUT_TOTAL_FLAGS={}'.format(error_count))
    if args.output:
        print('P1_INPUT_REPORT=' + write_new_json(args.output, result))
    print('P1_INPUT_PROVENANCE_CHECK_COMPLETED_NOT_ALL_GATES_PASS')
    return 0

def pinned_carla_egg(p):
    # The CARLA 0.9.15 distribution in this recorded Python 3.8 environment
    # has a working py3.7 Linux egg AND an incompatible py2.7 Linux egg.
    # Do not glob both: sorting puts py2.7 first in a child PYTHONPATH.
    egg = p['python_api'] / 'dist/carla-0.9.15-py3.7-linux-x86_64.egg'
    ensure(egg.is_file(), 'required CARLA py3.7 egg absent: ' + str(egg))
    return egg


def build_carla_pythonpath(p, cp, inherited):
    egg = pinned_carla_egg(p)
    roots = [str(cp), str(egg), str(p['root'] / 'RIFT'),
             str(p['root'] / 'scenario-dreamer'),
             str(p['root'] / 'carla/PythonAPI'), str(p['python_api'])]
    # Only drop foreign CARLA eggs; preserve all unrelated site paths.
    prior = [s for s in (inherited or '').split(os.pathsep)
             if s and not Path(s).name.startswith('carla-0.9.15-py')]
    return os.pathsep.join(roots + prior)


def child_carla_import_probe(python, cwd, env, egg):
    # This checks the EXACT interpreter and PYTHONPATH used by the real auditor.
    code = (
        'import carla,sys;'
        'print("PY="+sys.version.split()[0]);'
        'print("CARLA_ORIGIN="+str(carla.__file__));'
        'print("CLIENT="+str(carla.Client("127.0.0.1",31120).get_client_version()))'
    )
    info = subprocess.check_output([str(python), '-c', code], cwd=str(cwd),
                                   env=env, text=True, stderr=subprocess.STDOUT)
    for line in info.splitlines():
        print('P1_CHILD_IMPORT_' + line, flush=True)
    origin = next((line.split('=',1)[1] for line in info.splitlines()
                   if line.startswith('CARLA_ORIGIN=')), '')
    version = next((line.split('=',1)[1] for line in info.splitlines()
                    if line.startswith('CLIENT=')), '')
    ensure(origin.startswith(str(egg) + os.sep),
           'child did not import pinned CARLA egg: origin={} egg={}'.format(origin, egg))
    ensure(version == '0.9.15', 'child CARLA version mismatch: ' + version)


def carla_preflight(p, host, port):
    egg = pinned_carla_egg(p)
    sys.path.insert(0, str(p['python_api']))
    sys.path.insert(0, str(egg))
    import carla
    ensure(str(getattr(carla, '__file__', '')).startswith(str(egg) + os.sep),
           'preflight did not import pinned CARLA py3.7 egg: ' + str(getattr(carla,'__file__','')))
    c = carla.Client(host, port)
    c.set_timeout(20.0)
    ensure(c.get_server_version() == '0.9.15' and c.get_client_version() == '0.9.15', 'CARLA 0.9.15 mismatch')
    w = c.get_world()
    actors = w.get_actors()
    nvehicles = len(actors.filter('vehicle.*'))
    nwalkers = len(actors.filter('walker.pedestrian.*'))
    ensure(nvehicles == 0 and nwalkers == 0, 'nonempty world vehicles={} walkers={}'.format(nvehicles, nwalkers))
    print('P1_CARLA_PREFLIGHT_MAP=' + w.get_map().name)
    print('P1_CARLA_PREFLIGHT_NO_VEHICLES_OR_WALKERS')
    print('P1_EXCLUSIVITY_NOT_PROVEN_BY_ACTOR_COUNTS')

def cmd_run95(args):
    ensure(args.ack_dedicated, '--ack-dedicated is required; world may be loaded for every record')
    p, _, diagnostics, towns, flags = fixed_inputs(args.root)
    ensure(all('ego_route_missing_or_short' not in x['problems'] and
               'agents_or_rift_labels_missing' not in x['problems'] and
               'agents_labels_not_identity_verifiable' not in x['problems'] and
               'no_positive_rift_label' not in x['problems'] for x in diagnostics),
           'essential snapshot schema/labels invalid; audit not started')
    # All scene-label identity discrepancies and repeated source labels remain
    # independently unresolved; running an audit collects data, not identity PASS.
    groups = scene_id_duplicate_groups(diagnostics)
    if groups:
        print('P1_RUN95_UNRESOLVED_DUPLICATE_SCENE_IDS=' + json.dumps(groups, ensure_ascii=False), flush=True)
    carla_preflight(p, args.host, args.port)
    source_path = p['source'].resolve()
    cp = p['capsule'].resolve()
    command = [str(args.python), '-m', 'hv_v8.p1_native_audit',
               '--manifest', str(p['manifest']),
               '--rift-root', str(p['root'] / 'RIFT'),
               '--rift-recognition-config', str(p['config']),
               '--rift-mode', 'eval',
               '--carla-pythonapi', str(p['python_api']),
               '--map-identity-manifest', str(p['map_manifest']),
               '--host', args.host, '--port', str(args.port),
               '--startup-timeout', '600', '--startup-rpc-timeout', '5',
               '--startup-poll-interval', '2', '--rpc-timeout', '120',
               '--expected-carla-version', '0.9.15']
    env = os.environ.copy()
    egg = pinned_carla_egg(p)
    env['PYTHONPATH'] = build_carla_pythonpath(p, cp, env.get('PYTHONPATH', ''))
    env['PYTHONUNBUFFERED'] = '1'
    child_carla_import_probe(args.python, cp, env, egg)
    origin_code = 'import importlib.util;print(importlib.util.find_spec("hv_v8.p1_native_audit").origin)'
    origin = subprocess.check_output([str(args.python), '-c', origin_code], cwd=str(cp), env=env, text=True).strip()
    ensure(Path(origin).resolve() == source_path, 'wrong frozen module origin: '+ origin)
    stamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '_{}'.format(os.getpid())
    run_dir = p['base'] / ('p1_all95_locked_' + stamp)
    run_dir.mkdir(parents=False, exist_ok=False)
    out = run_dir / 'audit'
    log = run_dir / 'audit.log'
    cmd = command + ['--output', str(out)]
    launch = {
        'schema_version':'p1_full95_launch_v1', 'command':cmd,
        'capsule':str(cp), 'frozen_sha256':FROZEN_SHA,
        'manifest_sha256':MANIFEST_SHA,'map_manifest_sha256':MAP_MANIFEST_SHA,
        'input_town_distribution':towns, 'input_flags_count':flags,
        'unresolved_scene_identity_ordinals':[x['ordinal'] for x in diagnostics if x['scene_identity'] != 'exact'],
        'unresolved_scene_id_duplicate_groups':groups,
        'scientific_identity_verified':not groups and all(x['scene_identity'] == 'exact' for x in diagnostics),
        'dedicated_instance_acknowledged_by_operator':True,
        'note':'audit loads CARLA worlds/spawns/destroys actors. Does NOT imply physical model3 feasibility.'}
    write_new_json(run_dir / 'launch.json', launch)
    print('P1_FULL95_RUN_DIR=' + str(run_dir), flush=True)
    print('P1_FULL95_AUDIT_OUT=' + str(out), flush=True)
    print('P1_FULL95_LOG=' + str(log), flush=True)
    print('P1_FULL95_NEW_RUN_ONE_ATTEMPT_NO_RESUME', flush=True)
    with log.open('x', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, cwd=str(cp), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                universal_newlines=True, bufsize=1)
        for line in proc.stdout:
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        rc = proc.wait()
    write_new_json(run_dir / 'process_exit.json', {'returncode':rc, 'source_sha256':FROZEN_SHA,
                                                   'log_sha256':shafile(log), 'audit_path':str(out)})
    print('P1_FULL95_PROCESS_EXIT=' + str(rc))
    print('P1_FULL95_RESULT_DIR=' + str(run_dir))
    print('P1_FULL95_NOT_PHYSICAL_PASS_EVEN_IF_SUMMARY_TRUE')
    return 0 if rc == 0 else 3

def _validated_scene(doc):
    agents, labels = doc.get('agents'), doc.get('rift_labels')
    if not isinstance(agents, list) or not isinstance(labels, list):
        return None
    try:
        all_agent_ids = [str(a['actor_id']) for a in agents]
        all_label_ids = [str(l['actor_id']) for l in labels]
        if len(set(all_agent_ids)) != len(all_agent_ids) or len(set(all_label_ids)) != len(all_label_ids) or set(all_agent_ids) != set(all_label_ids):
            return None
        if not all(isinstance(l.get('is_interactive'), bool) for l in labels):
            return None
        return {str(l['actor_id']) for l in labels if l['is_interactive']}
    except (KeyError, TypeError):
        return None

def pass_nonnegative_finite_metric(value, tolerance):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0.0 <= value <= tolerance)

def analyze_one(row, ordinal, run_out, map_hashes):
    idx = int(row['record_index'])
    record_path = run_out / ('record_{:04d}.json'.format(idx))
    scene = read_json(row['snapshot_path'])
    positives = _validated_scene(scene)
    audit = read_json(record_path) if record_path.is_file() else None
    status = {
        'ordinal':ordinal,'record_index':idx, 'town':scene.get('town') or scene.get('map_name'),
        'snapshot_scene_id':scene.get('scene_id'), 'manifest_scene_id':row.get('scene_id'),
        'scene_id_exact':row.get('scene_id') == scene.get('scene_id'),
        'record_written':audit is not None,'native_subgates_verified':False,
        'target_count_gt':len(positives) if positives is not None else None,
        'target_count_audited':None,'raw_spawn_accepted_pose_verified':0,
        'raw_spawn_rejected_observed':0,'raw_spawn_unknown':0,
        'occupancy_status':'not_run', 'occupancy_reason':None,
        'occupancy_population_verified':False, 'occupancy_cleanup_verified':None,
        'joint_spawn_accepted':None,'full_scene_physical_feasibility':'unknown',
        'proven_legal_model3_positive_count':0,
        'exceptions':[],
    }
    if audit is None:
        status['exceptions'].append('record_not_written')
        return status
    if audit.get('record_index') != idx or audit.get('scene_id') != row.get('scene_id'):
        status['exceptions'].append('audit_manifest_record_identity_mismatch')
    native = audit.get('native_rows') or []
    raw = audit.get('fixed_model3_raw_pose_spawn') or []
    ids = (audit.get('target_selection') or {}).get('selected_target_actor_ids') or []
    audited_ids = [str(x) for x in ids]
    status['target_count_audited'] = len(audited_ids)
    if positives is None or not positives or set(audited_ids) != positives or len(audited_ids) != len(positives):
        status['exceptions'].append('target_selection_not_proven_equal_gt')
    native_ids = [str(n.get('actor_id')) for n in native]
    raw_ids = [str(n.get('actor_id')) for n in raw]
    if len(set(native_ids)) != len(native_ids) or set(native_ids) != set(audited_ids) or len(native_ids) != len(audited_ids):
        status['exceptions'].append('native_id_count_mismatch')
    if len(set(raw_ids)) != len(raw_ids) or set(raw_ids) != set(audited_ids) or len(raw_ids) != len(audited_ids):
        status['exceptions'].append('raw_id_count_mismatch')
    mapid = audit.get('map_identity') or {}
    town = str(status['town']).replace('\\','/').split('/')[-1]
    map_ok = (mapid.get('town_pass') is True and mapid.get('opendrive_status') == 'pass'
              and mapid.get('opendrive_expected_source') == 'map_identity_manifest'
              and mapid.get('opendrive_sha256_actual') == map_hashes.get(town)
              and mapid.get('opendrive_sha256_expected') == map_hashes.get(town))
    route = audit.get('route_checks') or []
    route_ok = bool(route) and all(c.get('pass') is True for c in route)
    parity_ok = bool(native) and all(n.get('rift_reference_parity_status') == 'pass' for n in native)
    roundtrip_ok = bool(native) and all(
        all(pass_nonnegative_finite_metric((n.get('coordinate_roundtrip') or {}).get(name), tol)
            for name, tol in (('roundtrip_position_error_m', 1e-6),
                              ('roundtrip_z_error_m', 1e-6),
                              ('roundtrip_yaw_error_rad', 1e-9)))
        for n in native)
    # RIFT-positive labels are supervision only if the independently re-run
    # native oracle returns a positive result, not merely 'parity PASS' for two negatives.
    native_positive_ok = (bool(native) and all(
        n.get('native_positive_status') == 'pass' and n.get('native_positive') is True
        for n in native))
    bp_ok = (audit.get('fixed_model3_blueprint') or {}).get('status') == 'pass'
    selection_ok = (audit.get('target_selection') or {}).get('status') == 'pass'
    no_mutation = audit.get('forbidden_fallbacks_used') == [] and audit.get('post_spawn_pose_correction') is False
    status['native_subgates_verified'] = all((map_ok,route_ok,parity_ok,roundtrip_ok,bp_ok,selection_ok,no_mutation,
                                             native_positive_ok,
                                             positives is not None and len(audited_ids) == len(positives)
                                             and set(audited_ids) == positives))
    for key,val in [('map_not_verified',map_ok),('route_not_verified',route_ok),('native_parity_not_verified',parity_ok),
                    ('coordinate_roundtrip_not_verified',roundtrip_ok),('native_positive_not_verified',native_positive_ok),('blueprint_not_verified',bp_ok),
                    ('selection_not_verified',selection_ok),('mutation_or_fallback_detected',no_mutation)]:
        if not val:
            status['exceptions'].append(key)
    for sp in raw:
        if sp.get('status') == 'accepted' and sp.get('spawn_rpc_accepted') is True and sp.get('pose_validation_status') == 'pass' and sp.get('raw_pose_correction') is False:
            status['raw_spawn_accepted_pose_verified'] += 1
        elif sp.get('status') == 'rejected' and sp.get('spawn_rpc_accepted') is False and sp.get('pose_validation_status') == 'not_spawned' and sp.get('reason'):
            status['raw_spawn_rejected_observed'] += 1
        else:
            status['raw_spawn_unknown'] += 1
    occ = audit.get('fixed_model3_scene_occupancy') or {}
    status['occupancy_status'] = occ.get('status', 'unknown')
    status['occupancy_reason'] = occ.get('reason')
    status['occupancy_cleanup_verified'] = occ.get('cleanup_verified')
    status['joint_spawn_accepted'] = occ.get('joint_model3_spawn_accepted')
    status['full_scene_physical_feasibility'] = occ.get('full_scene_physical_feasibility','unknown')
    if positives is not None and isinstance(scene.get('agents'),list):
        expected_bg = ['__ego__'] + [str(a['actor_id']) for a in scene['agents'] if str(a['actor_id']) not in positives]
        expected_t = [str(a['actor_id']) for a in scene['agents'] if str(a['actor_id']) in positives]
        obs_bg = [str(a.get('actor_id')) for a in occ.get('background', [])]
        obs_ids = [str(x) for x in occ.get('expected_background_actor_ids',[])]
        obs_t = [str(x) for x in occ.get('expected_target_actor_ids',[])]
        status['occupancy_population_verified'] = (
            occ.get('schema_version') == 'p1_occupancy_agents_joint_v1'
            and obs_ids == expected_bg
            and obs_t == expected_t
            and len(obs_bg) == len(expected_bg)
            and obs_bg == expected_bg
            and occ.get('expected_background_count') == len(expected_bg)
            and occ.get('expected_target_count') == len(expected_t)
            and occ.get('source_pose_modified') is False
        )
    if not status['occupancy_population_verified']:
        status['exceptions'].append('occupancy_population_not_verified')
    if status['occupancy_cleanup_verified'] is not True:
        status['exceptions'].append('cleanup_not_verified')
    if status['raw_spawn_unknown']:
        status['exceptions'].append('raw_spawn_unknown_or_inconsistent')
    if status['full_scene_physical_feasibility'] == 'pass':
        # The current frozen implementation deliberately never asserts this.
        status['exceptions'].append('unexpected_untrusted_physical_pass_claim')
    # The current occupancy auditor does NOT verify post-spawn physics, so
    # no actor can be called a proven *full scene* legal Model3 positive.
    return status

def cmd_report(args):
    p, rows, diagnostics, towns, flags = fixed_inputs(args.root)
    out = Path(args.audit).resolve()
    ensure(out.is_dir(), 'audit directory absent: '+str(out))
    mm = read_json(p['map_manifest'])
    hashes = {str(m['town']).replace('\\','/').split('/')[-1]:m['opendrive_sha256'] for m in mm['maps']}
    result_rows = [analyze_one(row, i, out, hashes) for i, row in enumerate(rows)]
    written = sum(int(x['record_written']) for x in result_rows)
    native = sum(int(x['native_subgates_verified']) for x in result_rows)
    pop = sum(int(x['occupancy_population_verified']) for x in result_rows)
    cleanup = sum(int(x['occupancy_cleanup_verified'] is True) for x in result_rows)
    counters = collections.Counter(x['occupancy_status'] for x in result_rows)
    rawaccepted = sum(x['raw_spawn_accepted_pose_verified'] for x in result_rows)
    rawrejected = sum(x['raw_spawn_rejected_observed'] for x in result_rows)
    rawunknown = sum(x['raw_spawn_unknown'] for x in result_rows)
    summary = read_json(out / 'summary.json') if (out / 'summary.json').is_file() else None
    summary_records_ok = summary is not None and summary.get('records') == 95
    summary_native = summary is not None and summary.get('p1_pass') is True
    # A mismatched manifest/snapshot ID remains UNKNOWN until its provenance
    # is independently established. Never promote all-95 to PASS on that basis.
    duplicates = scene_id_duplicate_groups(diagnostics)
    identity_all_exact = all(x['scene_id_exact'] for x in result_rows) and not duplicates
    all_native = summary_records_ok and summary_native and native == 95 and identity_all_exact
    extra_files = sorted(x.name for x in out.glob('record_*.json') if x.name not in
                         {'record_{:04d}.json'.format(int(r['record_index'])) for r in rows})
    calibration_complete = (all_native and not extra_files and written == 95
                            and pop == 95 and cleanup == 95 and rawunknown == 0
                            and all(x['occupancy_status'] == 'complete' for x in result_rows))
    accepted = {
        'schema_version':'p1_95_independent_ledger_v1',
        'source_frozen_sha256':FROZEN_SHA,
        'input_manifest_sha256':MANIFEST_SHA,
        'record_files_found':written,'expected_records':95,
        'summary_records':summary.get('records') if summary else None,
        'summary_present':summary is not None,
        'record_provenance_identity_all_exact':identity_all_exact,
        'unresolved_manifest_scene_id_duplicate_groups':duplicates,
        'summary_p1_pass_subgate_only':summary.get('p1_pass') if summary else None,
        'native_subgates_verified_records':native,
        'native_95_verified':bool(all_native and not extra_files),
        'p1_95_calibration_complete':bool(calibration_complete),
        'p1_model3_legal_training_dataset_ready':False,
        'occupancy_population_verified_records':pop,
        'occupancy_cleanup_verified_records':cleanup,
        'occupancy_status_counts':dict(sorted(counters.items())),
        'raw_model3_spawn_accepted_pose_verified':rawaccepted,
        'raw_model3_spawn_rejected_observed':rawrejected,
        'raw_model3_spawn_unknown_or_inconsistent':rawunknown,
        'full_scene_model3_legal_positive_proven_count':0,
        'all_95_full_scene_physics_proven':False,
        'scene_id_exact_records':sum(int(x['scene_id_exact']) for x in result_rows),
        'scene_identity_unresolved_record_indices':[x['record_index'] for x in result_rows if not x['scene_id_exact']],
        'unexpected_extra_record_files':extra_files,
        'first_not_written':next(({'ordinal':x['ordinal'],'record_index':x['record_index']} for x in result_rows if not x['record_written']),None),
        'important':'Native subgates and raw Model3 physical feasibility are not equivalent. Occupancy complete means all spawn outcomes recorded, NOT full-scene physics PASS.',
        'records':result_rows,
    }
    target_dir = Path(args.report_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=False)
    jpath = write_new_json(target_dir / 'p1_95_ledger.json', accepted)
    columns = ['ordinal','record_index','town','record_written','scene_id_exact','native_subgates_verified',
               'target_count_gt','target_count_audited','raw_spawn_accepted_pose_verified','raw_spawn_rejected_observed',
               'raw_spawn_unknown','occupancy_status','occupancy_reason','occupancy_population_verified',
               'occupancy_cleanup_verified','joint_spawn_accepted','full_scene_physical_feasibility','exceptions']
    with (target_dir / 'p1_95_ledger.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in result_rows:
            writer.writerow({k: (';'.join(r[k]) if k == 'exceptions' else r.get(k)) for k in columns})
    # Per-target ledger keeps real negative spawn results separate from native GT.
    # It never promotes accepted raw RPC to full-scene physically legal status.
    target_fields = ('ordinal','record_index','town','actor_id','historical_type_id',
                     'rift_interactive_label','native_parity_status','raw_spawn_status',
                     'raw_spawn_reason','raw_spawn_rpc_accepted','pose_validation_status',
                     'occupancy_status','full_scene_model3_legality','model3_legal_supervision_positive')
    actor_total = 0
    with (target_dir / 'p1_95_targets.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=target_fields)
        writer.writeheader()
        for ordinal, row in enumerate(rows):
            scene = read_json(row['snapshot_path'])
            positives = _validated_scene(scene)
            if positives is None:
                continue
            agent_map = {str(a['actor_id']):a for a in scene['agents']}
            rec_path = out / ('record_{:04d}.json'.format(int(row['record_index'])))
            audit = read_json(rec_path) if rec_path.is_file() else {}
            native_by_id = {str(a['actor_id']):a for a in audit.get('native_rows',[]) if a.get('actor_id') is not None}
            sedan = {str(a['actor_id']):a for a in audit.get('fixed_model3_raw_pose_spawn',[]) if a.get('actor_id') is not None}
            occ = audit.get('fixed_model3_scene_occupancy') or {}
            for aid in sorted(positives):
                nr,sp = native_by_id.get(aid,{}),sedan.get(aid,{})
                writer.writerow({
                    'ordinal':ordinal,'record_index':row['record_index'],
                    'town':scene.get('town') or scene.get('map_name'),'actor_id':aid,
                    'historical_type_id':agent_map[aid].get('type_id'),
                    'rift_interactive_label':True,
                    'native_parity_status':nr.get('rift_reference_parity_status','not_run'),
                    'raw_spawn_status':sp.get('status','not_run'),
                    'raw_spawn_reason':sp.get('reason'),
                    'raw_spawn_rpc_accepted':sp.get('spawn_rpc_accepted'),
                    'pose_validation_status':sp.get('pose_validation_status','not_run'),
                    'occupancy_status':occ.get('status','not_run'),
                    'full_scene_model3_legality':'unproven',
                    'model3_legal_supervision_positive':False,
                })
                actor_total += 1
    print('P1_LEDGER_TARGETS_CSV=' + str(target_dir / 'p1_95_targets.csv'))
    print('P1_LEDGER_GT_INTERACTIVE_TARGETS=' + str(actor_total))
    print('P1_LEDGER_JSON=' + jpath)
    print('P1_LEDGER_CSV=' + str(target_dir / 'p1_95_ledger.csv'))
    print('P1_NATIVE_95_VERIFIED=' + str(accepted['native_95_verified']))
    print('P1_CALIBRATION_95_COMPLETE=' + str(accepted['p1_95_calibration_complete']))
    print('P1_MODEL3_LEGAL_TRAINING_READY=False')
    print('P1_WRITTEN={}/95'.format(written))
    print('P1_NATIVE_RECORDS_VERIFIED={}/95'.format(native))
    print('P1_OCC_POPULATION_VERIFIED={}/95'.format(pop))
    print('P1_RAW_SPAWN_ACCEPTED={} REJECTED={} UNKNOWN={}'.format(rawaccepted,rawrejected,rawunknown))
    print('P1_OCC_STATUS_COUNTS=' + json.dumps(dict(counters),ensure_ascii=False))
    print('P1_FULL_SCENE_PHYSICS_PROVEN=False (by current audit contract)')
    if written < 95 or not accepted['native_95_verified']:
        print('P1_FULL95_NATIVE_GATE_INCOMPLETE_OR_FAIL')
    if not identity_all_exact:
        print('P1_MANIFEST_SCENE_ID_PROVENANCE_UNRESOLVED_NOT_PASS')
    if duplicates:
        print('P1_MANIFEST_SCENE_ID_DUPLICATES_UNRESOLVED=' + json.dumps(duplicates, ensure_ascii=False))
    if flags:
        print('P1_INPUT_IDENTITY_FLAGS_UNRESOLVED=' + str(flags))
    return 0

def cmd_p2_export(args):
    _, rows, diagnostics, towns, flags = fixed_inputs(args.root)
    duplicates = scene_id_duplicate_groups(diagnostics)
    conditions = []
    labels = []
    for i, row in enumerate(rows):
        scene = read_json(row['snapshot_path'])
        positives = _validated_scene(scene)
        ensure(positives is not None and positives, 'GT labels incomplete ordinal={}'.format(i))
        common = {'ordinal':i,'record_index':int(row['record_index']),
                  'manifest_scene_id':row['scene_id'], 'snapshot_scene_id':scene.get('scene_id'),
                  'town':scene.get('town') or scene.get('map_name')}
        # Online condition: **no target agents/labels/interactions**.
        condition = dict(common)
        condition.update({'ego':scene.get('ego'),'ego_route':scene.get('ego_route'),
                          'map_lanes':None,
                          'map_lanes_warning':'carla map graph required; snapshot map_lanes may be absent/truncated',
                          'gt_target_positions_in_condition':False})
        label = dict(common)
        byid = {str(x['actor_id']):x for x in scene['agents']}
        label.update({'gt_source':'snapshot.rift_labels.is_interactive',
                      'targets':[{'actor_id':l['actor_id'], 'pose':{
                          k:byid[str(l['actor_id'])].get(k) for k in ('x','y','z','yaw_deg','vx','vy','vz','speed_mps','type_id')},
                          'interaction_waypoint':l.get('interaction_waypoint'),
                          'ego_route_index':l.get('ego_route_index'),
                          'ego_distance':l.get('ego_distance'),
                          'cbv_distance':l.get('cbv_distance'),
                          'distance_diff':l.get('distance_diff')}
                          for l in scene['rift_labels'] if l['is_interactive']],
                      'native_parity_from_real_audit':'NOT_JOINED_MUST_USE_P1_LEDGER',
                      'raw_pose_model3_physical_legal':'NOT_PROVEN_FROM_GT'})
        conditions.append(condition)
        labels.append(label)
    dest = Path(args.output_dir).resolve()
    dest.mkdir(parents=True, exist_ok=False)
    for fn, lines in (('p2_conditions_95.jsonl', conditions),('p2_gt_labels_95.jsonl', labels)):
        with (dest / fn).open('x', encoding='utf-8') as f:
            for line in lines:
                f.write(json.dumps(line,ensure_ascii=False,sort_keys=True) + '\n')
    manifest = {'schema_version':'p2_export_95_v1', 'records':95,'town_counts':towns,
                'input_manifest_sha256':MANIFEST_SHA,
                'conditions_file_sha256':shafile(dest / 'p2_conditions_95.jsonl'),
                'gt_file_sha256':shafile(dest / 'p2_gt_labels_95.jsonl'),
                'unresolved_manifest_scene_id_ordinals':[x['ordinal'] for x in diagnostics if x['scene_identity'] != 'exact'],
                'unresolved_manifest_scene_id_duplicate_groups':duplicates,
                'scientific_identity_verified':not duplicates and all(x['scene_identity'] == 'exact' for x in diagnostics),
                'conditions_are_complete_model_inputs':False,
                'do_not_claim':'This is 95 P1 snapshot supervision, NOT the fixed 110 front-GT P2 denominator, NOT P2 recall, NOT Model3 spawn feasibility.',
                'leakage_guard':'online conditions exclude agents, rift_labels and GT interaction_waypoint'}
    print('P2_EXPORT_MANIFEST=' + write_new_json(dest / 'provenance.json',manifest))
    print('P2_CONDITIONS=' + str(dest / 'p2_conditions_95.jsonl'))
    print('P2_GT_LABELS=' + str(dest / 'p2_gt_labels_95.jsonl'))
    print('P2_SCENE_ID_DUPLICATES_UNRESOLVED=' + json.dumps(duplicates, ensure_ascii=False))
    print('P2_NOT_110_GT_APPROACH_RECALL_OR_TRAINING_GATE')
    return 0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd')
    for name in ('inputs','run95','report','p2-export'):
        x = sub.add_parser(name)
        x.add_argument('--root',required=True)
        if name == 'inputs':
            x.add_argument('--output')
        if name == 'run95':
            x.add_argument('--python',required=True)
            x.add_argument('--host',default='127.0.0.1')
            x.add_argument('--port',type=int,default=31120)
            x.add_argument('--ack-dedicated',action='store_true')
        if name == 'report':
            x.add_argument('--audit',required=True)
            x.add_argument('--report-dir',required=True)
        if name == 'p2-export':
            x.add_argument('--output-dir',required=True)
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return 2
    try:
        return {'inputs':cmd_inputs,'run95':cmd_run95,'report':cmd_report,'p2-export':cmd_p2_export}[args.cmd](args)
    except (EvidenceError,ValueError,KeyError,TypeError,FileNotFoundError,subprocess.CalledProcessError) as exc:
        print('P1P2_FAIL_CLOSED={} {}'.format(type(exc).__name__,exc),file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())
