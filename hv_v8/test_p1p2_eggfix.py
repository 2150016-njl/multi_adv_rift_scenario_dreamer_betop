#!/usr/bin/env python3
"""Only launcher/import-selection and offline probe tests. NOT real CARLA pass."""
from pathlib import Path
import tempfile, os, sys, zipfile, subprocess, importlib.util, json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('launcher',HERE/'p1p2_pipeline_eggfix.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)/'repo'; api=root/'carla/PythonAPI/carla'; dist=api/'dist'; dist.mkdir(parents=True)
    good=dist/'carla-0.9.15-py3.7-linux-x86_64.egg'
    bad=dist/'carla-0.9.15-py2.7-linux-x86_64.egg'
    for egg,content in [(good,'MAGIC="GOOD_PY3"'),(bad,'raise ImportError("BAD_PY2_EGG_SELECTED")')]:
        with zipfile.ZipFile(egg,'w') as z:z.writestr('carla/__init__.py',content+'\n')
    p={'root':root,'python_api':api}; cp=root/'capsule';cp.mkdir()
    selected=module.pinned_carla_egg(p)
    assert selected==good
    inherited=os.pathsep.join([str(bad),'/some/non_carla_dir',str(good)])
    path=module.build_carla_pythonpath(p,cp,inherited)
    items=path.split(os.pathsep)
    assert items[1]==str(good) and str(bad) not in items and '/some/non_carla_dir' in items
    env=os.environ.copy();env['PYTHONPATH']=path
    child=subprocess.check_output([sys.executable,'-c','import carla;print(carla.MAGIC)'],env=env,text=True)
    assert child.strip()=='GOOD_PY3',child
    print('TEST_CHILD_IMPORT_SELECTS_PINNED_PY3_NOT_PY2=PASS')

    # Fail closed when only py2 exists; no automatic fallback.
    good.unlink()
    try:
        module.pinned_carla_egg(p)
    except module.EvidenceError:
        pass
    else:
        raise AssertionError('py2 fallback wrongly allowed')
    print('TEST_MISSING_PY3_EGG_FAIL_CLOSED=PASS')

# Existing report from real CARLA probe (if supplied) for offline classification.
if len(sys.argv)>1:
    probe=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('analyzer',HERE/'p2_town15_lane_miss_analysis.py')
    ana=importlib.util.module_from_spec(spec);spec.loader.exec_module(ana)
    obj=ana.analyze(probe)
    assert obj['gt_target_count']==26 and obj['counts']=={
        'adjacent_lane_lateral_rule_not_evaluated':12,
        'longitudinal_reachability_true':14},obj['counts']
    assert all(r['lane_change_legality']=='NOT_EVALUATED' for r in obj['results'])
    print('TEST_REAL_TOWN15_REPORT_12_ADJACENT_LANE_MISSES=PASS')
print('P1P2_EGGFIX_OFFLINE_TEST_PASS_NOT_REAL_CARLA')
