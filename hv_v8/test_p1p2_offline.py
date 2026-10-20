#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline integrated real-record fixture tests; never contacts CARLA."""
from __future__ import annotations
import importlib.util
import hashlib
import json
import pathlib
import shutil
import tempfile
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = HERE / 'p1p2_fixtures'
# Production deploy: tests are optional, but the source artifacts need to be
# bundled for this developer regression to execute.

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def real_file(pattern):
    fs=list(EVIDENCE.rglob(pattern))
    assert len(fs)==1,(pattern,fs)
    return fs[0]

def run():
    mod=module(HERE/'p1p2_pipeline.py','p1p2_pipeline')
    p2=module(HERE/'p2_topology_probe.py','p2_topology_probe')
    source=real_file('p1_native_audit.py')
    snapshot=json.loads(real_file('route27_env0_RouteScenario_24333_idx27_rep0_frame145.json').read_text())
    record=json.loads(real_file('record_0100.json').read_text())
    assert len(snapshot['agents'])==8
    assert sum(x['is_interactive'] for x in snapshot['rift_labels'])==2
    with tempfile.TemporaryDirectory(prefix='p1p2_95_regression_') as root_s:
        root=pathlib.Path(root_s)
        base=root/'data/hv_init/v7'
        (base/'manifests').mkdir(parents=True)
        (root/'RIFT/rift/cbv/recognition/config').mkdir(parents=True)
        (root/'RIFT/rift/cbv/recognition/config/attention.yaml').write_text('{}')
        (root/'carla/PythonAPI/carla').mkdir(parents=True)
        cp=base/mod.CAPSULE_NAME
        (cp/'hv_v8').mkdir(parents=True)
        shutil.copy2(source,cp/'hv_v8/p1_native_audit.py')
        mod.FROZEN_SHA=mod.shafile(cp/'hv_v8/p1_native_audit.py')
        (cp/'p1_source_provenance.json').write_text(json.dumps({'frozen_sha256':mod.FROZEN_SHA}))
        maps={'schema_version':'p1_map_identity_manifest_v1',
              'maps':[{'town':t,'opendrive_sha256':('8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17' if t=='Town15' else 'a'*64)} for t in ('Town04','Town11','Town12','Town13','Town15')]}
        mpath=base/'manifests'/mod.MAP_MANIFEST_NAME
        mpath.write_text(json.dumps(maps))
        mod.MAP_MANIFEST_SHA=mod.shafile(mpath)
        rows=[]
        snapdir=root/'snapshots'
        snapdir.mkdir()
        for i in range(95):
            scene=dict(snapshot)
            scene['scene_id']=snapshot['scene_id'] if i==8 else 'synthetic_case_{:03d}'.format(i)
            path=snapdir/'scene_{:03d}.json'.format(i)
            path.write_text(json.dumps(scene))
            idx=100 if i==8 else 1000+i
            sid=scene['scene_id']+(':sparse0' if i==50 else '')
            rows.append({'record_index':idx,'scene_id':sid,'snapshot_path':str(path),
                         'snapshot_sha256':mod.shafile(path)})
        mf=base/'manifests/p1_calibration95.jsonl'
        mf.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        mod.MANIFEST_SHA=mod.shafile(mf)
        args=types.SimpleNamespace(root=str(root),output=str(root/'inputs.json'))
        assert mod.cmd_inputs(args)==0
        inputs=json.loads((root/'inputs.json').read_text())
        assert inputs['records']==95 and inputs['scene_id_discrepancy_count']==1
        assert inputs['scientific_identity_all_exact'] is False
        print('TEST_INPUTS_95_WITH_ONE_UNRESOLVED_SCENE_ID=PASS')
        export_dir=root/'p2_export'
        assert mod.cmd_p2_export(types.SimpleNamespace(root=str(root),output_dir=str(export_dir)))==0
        conditions=[json.loads(x) for x in (export_dir/'p2_conditions_95.jsonl').read_text().splitlines()]
        targets=[json.loads(x) for x in (export_dir/'p2_gt_labels_95.jsonl').read_text().splitlines()]
        assert len(conditions)==len(targets)==95
        assert len(targets[8]['targets'])==2
        assert conditions[8]['map_lanes'] is None
        assert 'agents' not in conditions[8] and 'rift_labels' not in conditions[8]
        assert not any(x in str(conditions[8]) for x in ('11141','11143'))
        assert json.loads((export_dir/'provenance.json').read_text())['unresolved_manifest_scene_id_ordinals']==[50]
        print('TEST_P2_TARGET_LABEL_LEAKAGE_GUARD_AND_PROVENANCE=PASS')
        auditdir=root/'audit'
        auditdir.mkdir()
        shutil.copy2(real_file('record_0100.json'),auditdir/'record_0100.json')
        # This old audit had only ego occupancy, so it cannot be counted as complete.
        assert mod.cmd_report(types.SimpleNamespace(root=str(root),audit=str(auditdir),report_dir=str(root/'report_partial')))==0
        ledger=json.loads((root/'report_partial/p1_95_ledger.json').read_text())
        assert ledger['record_files_found']==1 and ledger['native_95_verified'] is False
        assert ledger['full_scene_model3_legal_positive_proven_count']==0
        print('TEST_PARTIAL_AUDIT_CANNOT_BECOME_95_PASS=PASS')
        # Deliberately set summary PASS and populate all 95 records to ensure
        # :sparse0 still blocks independently claimed 95 record identity PASS.
        for row in rows:
            x=dict(record)
            x['record_index']=row['record_index']
            x['scene_id']=row['scene_id']
            (auditdir/'record_{:04d}.json'.format(row['record_index'])).write_text(json.dumps(x))
        (auditdir/'summary.json').write_text(json.dumps({'records':95,'p1_pass':True}))
        assert mod.cmd_report(types.SimpleNamespace(root=str(root),audit=str(auditdir),report_dir=str(root/'report_full_fixture')))==0
        ledger2=json.loads((root/'report_full_fixture/p1_95_ledger.json').read_text())
        assert ledger2['native_95_verified'] is False
        assert ledger2['scene_identity_unresolved_record_indices']==[1050]
        print('TEST_UNRESOLVED_PROVENANCE_CANNOT_BECOME_ALL95_PASS=PASS')
    class WP:
        def __init__(self,k,nxt=None):
            self.road_id,self.section_id,self.lane_id=k
            self._next=nxt or []
        def next(self,step):return self._next
    ka,kb,kc=(1,0,-1),(2,0,-1),(3,0,-1)
    A,B,C=WP(ka),WP(kb),WP(kc)
    B._next=[C]
    class Map:
        def get_topology(self): return [(A,B)]
    nodes,edges,stats=p2.construct_graph(Map())
    assert edges[ka]=={kb} and edges[kb]=={kc}
    assert p2.forward_reachable(edges,ka,kc) is True
    assert p2.forward_reachable(edges,kc,ka) is False
    assert p2.forward_reachable(edges,ka,kc,cap=1) is False
    print('TEST_P2_DIRECTIONAL_TOPOLOGY_AND_BOUND=PASS')
    print('ALL_P1P2_OFFLINE_INTEGRATION_TESTS_PASS_NOT_CARLA')

if __name__=='__main__':
    run()
