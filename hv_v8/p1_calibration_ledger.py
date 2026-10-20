#!/usr/bin/env python3
"""Independent, offline P1 calibration ledger; never equate p1_pass with feasibility.

Inputs are original snapshot + already-generated real CARLA record/summary.
Creates new JSON evidence (exclusive filename) without connecting to CARLA.
A successful ledger process exit is NOT a CARLA PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fail(reason):
    raise ValueError(reason)


def normalized_id(row):
    return str(row['actor_id'])


def ledger(snapshot, record, summary, source_path):
    if not isinstance(snapshot.get('agents'), list) or not isinstance(snapshot.get('rift_labels'), list):
        fail('agents/rift_labels missing')
    if not isinstance(snapshot.get('ego'), dict):
        fail('ego missing')
    agents = snapshot['agents']; labels = snapshot['rift_labels']
    if not agents or not labels or any(not isinstance(x, dict) for x in agents+labels):
        fail('invalid or empty agents or labels')
    if any('actor_id' not in x for x in agents+labels):
        fail('actor ID missing')
    agent_ids = [normalized_id(x) for x in agents]
    label_ids = [normalized_id(x) for x in labels]
    if len(set(agent_ids)) != len(agent_ids) or len(set(label_ids)) != len(label_ids) or set(agent_ids) != set(label_ids):
        fail('agent / label ID mismatch or duplicate')
    if any(not isinstance(l.get('is_interactive'),bool) for l in labels):
        fail('rift label truth non-boolean or missing')
    positives = sorted(str(l['actor_id']) for l in labels if l['is_interactive'])
    if not positives:
        fail('no interactive target; no fallback')
    if int(record.get('record_index', -1)) != 100 or snapshot.get('scene_id') != record.get('scene_id'):
        fail('record/snapshot identity mismatch; expected only record_index 100')
    if int(summary.get('records', -1)) != 1:
        fail('summary not one record')
    selected = (record.get('target_selection') or {}).get('selected_target_actor_ids', [])
    if set(str(x) for x in selected) != set(positives) or len(selected) != len(positives):
        fail('selected targets do not match authoritative rift labels')
    native = record.get('native_rows') or []
    if {str(x.get('actor_id')) for x in native} != set(positives) or len(native) != len(positives):
        fail('native row identities incomplete')
    raw = record.get('fixed_model3_raw_pose_spawn') or []
    if {str(x.get('actor_id')) for x in raw} != set(positives) or len(raw) != len(positives):
        fail('raw pose outcomes incomplete')
    obs = record.get('fixed_model3_scene_occupancy') or {}
    route = record.get('route_checks') or []
    mapid = record.get('map_identity') or {}
    native_result = {
        'map_verified': bool(mapid.get('town_pass') is True and mapid.get('opendrive_status')=='pass' and mapid.get('opendrive_expected_source') == 'map_identity_manifest'),
        'route_verified': bool(route and all(c.get('pass') is True for c in route)),
        'target_selection_verified': record.get('target_selection',{}).get('status')=='pass',
        'native_parity_verified': bool(native and all(n.get('rift_reference_parity_status') == 'pass' for n in native)),
        'coordinate_roundtrip_summary_pass': summary.get('coordinate_roundtrip_all_pass') is True,
    }
    native_result['all_verified'] = all(native_result.values())
    byid = {str(x['actor_id']): x for x in agents}
    rows = []
    for x in raw:
        aid = str(x['actor_id'])
        state = x.get('status')
        if state == 'accepted' and x.get('spawn_rpc_accepted') is True and x.get('pose_validation_status') == 'pass' and x.get('raw_pose_correction') is False:
            raw_feasible = 'accepted_pose_verified'
        elif state == 'rejected' and x.get('spawn_rpc_accepted') is False and x.get('pose_validation_status') == 'not_spawned' and x.get('reason'):
            raw_feasible = 'rejected_observed'
        else:
            raw_feasible = 'unknown_or_inconsistent'
        rows.append({
            'actor_id': aid,
            'historical_type_id': byid[aid].get('type_id'),
            'historical_rift_parity_status': next(n['rift_reference_parity_status'] for n in native if str(n.get('actor_id'))==aid),
            'raw_model3_spawn_calibration_status': raw_feasible,
            'raw_model3_spawn_observed_status': state,
            'qualified_as_proven_full_scene_model3_positive': False,
            'full_scene_feasibility_status': 'unknown' if obs.get('status')=='unknown' else 'not_independently_established',
        })
    expected_background = ['__ego__'] + [str(x['actor_id']) for x in agents if str(x['actor_id']) not in set(positives)]
    observed_background = [str(x.get('actor_id')) for x in obs.get('background', [])]
    incomplete = sorted(set(expected_background) - set(observed_background))
    if 'schema_version' in obs and obs.get('schema_version') == 'p1_occupancy_agents_joint_v1':
        occupancy_population_audited = (obs.get('expected_background_actor_ids') == expected_background and
                                        set(obs.get('expected_target_actor_ids', [])) == set(positives) and
                                        obs.get('expected_background_count') == len(expected_background) and
                                        obs.get('expected_target_count') == len(positives))
    else:
        occupancy_population_audited = False
    return {
        'schema_version': 'p1_record100_calibration_ledger_v1',
        'interpretation': 'measured raw spawn rejections are negative evidence; unknown occupancy is never treated as legal',
        'record_index': 100, 'scene_id': snapshot['scene_id'],
        'original_manifest_95_pass': False,
        'this_record_native_calibrated': native_result,
        'source_reported_p1_pass': summary.get('p1_pass'),
        'source_summary_is_not_physical_feasibility_pass': True,
        'agents_count': len(agents), 'selected_target_count': len(positives),
        'expected_background_actor_ids': expected_background,
        'observed_background_actor_ids': observed_background,
        'missing_from_occupancy_replay': incomplete,
        'occupancy_population_contract_verified': occupancy_population_audited,
        'occupancy_status': obs.get('status','unknown'),
        'occupancy_reason': obs.get('reason'),
        'occupancy_full_scene_feasible': None,
        'model3_generation_positive_examples_from_this_record': [],
        'actor_calibrations': rows,
        'independent_physical_controls': 'not_in_this_uploaded_evidence_package; no unverified controls imported',
        'diagnostic_modified_pose_may_not_become_GT': True,
        'source_audit_sha256': sha(source_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ('snapshot','record','summary','frozen-source','output'):
        ap.add_argument('--'+k,required=True)
    args = ap.parse_args()
    src = Path(args.frozen_source)
    if sha(src) not in (
        '34178e62ec0327768135d1882a0d8fd5d78ed12260cc2db1d176e4762f67d7e5',
        'ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f',
    ):
        fail('wrong frozen source SHA; this version is scoped to actual evidence record 100')
    p = Path(args.snapshot); r = Path(args.record); s = Path(args.summary)
    if sha(p) != '37bd5c64865372a4627b9a7c9d53f03bbe4b8cb2b3dc7ff8979a3f0d203447c7':
        fail('snapshot source SHA mismatch')
    result = ledger(json.loads(p.read_text()),json.loads(r.read_text()),json.loads(s.read_text()),src)
    result['sources'] = {key:{'path':str(path.resolve()), 'sha256':sha(path)} for key,path in (('snapshot',p),('record',r),('summary',s),('frozen_source',src))}
    out = Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf8') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print('P1_LEDGER_WRITTEN='+str(out.resolve()))
    print('P1_NATIVE_SUBGATES='+json.dumps(result['this_record_native_calibrated'],ensure_ascii=False))
    print('P1_OCCUPANCY_MISSING_BACKGROUND_IDS='+json.dumps(result['missing_from_occupancy_replay']))
    print('P1_MODEL3_FEASIBLE_POSITIVE_COUNT=0')
    print('P1_LEDGER_EVIDENCE_ONLY_NOT_95_PASS')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        print('P1_LEDGER_FAIL_CLOSED=%s:%s'%(type(e).__name__,e),file=sys.stderr)
        sys.exit(2)
