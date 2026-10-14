import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

import audit_hv_snapshot_experiment as audit
from high_value_init.audit_physics_geometry import StaticBoxCollection,boxes_intersect
from run_hv_mainline_batch import stage_signature


def box(center,angle=0.,extent=(2.,1.,1.)):
    c,s=np.cos(angle),np.sin(angle)
    return {'center':list(center),'axes':[[c,-s,0.],[s,c,0.],[0.,0.,1.]],'extent':list(extent)}


class GeometryResume(unittest.TestCase):
    def test_vector_prefilter_preserves_exact_sat_results_for_rotated_boxes(self):
        rng=np.random.default_rng(3)
        boxes=[box(rng.uniform(-20,20,3),rng.uniform(-np.pi,np.pi),rng.uniform(.1,5,3)) for _ in range(120)]
        for b in boxes:b['axes']=np.linalg.qr(rng.normal(size=(3,3)))[0].tolist()
        index=StaticBoxCollection([{'box':b} for b in boxes])
        for _ in range(8):
            actor=box(rng.uniform(-10,10,3),rng.uniform(-np.pi,np.pi),rng.uniform(.1,4,3))
            actor['axes']=np.linalg.qr(rng.normal(size=(3,3)))[0].tolist()
            exhaustive=[i for i,b in enumerate(boxes) if boxes_intersect(actor,b)]
            filtered=[i for i in index.candidates(actor) if boxes_intersect(actor,boxes[i])]
            self.assertEqual(filtered,exhaustive)

    def test_touching_tolerance_and_empty_collection_are_preserved(self):
        actor=box([0.,0.,0.]);neighbor=box([4.+5.e-7,0.,0.])
        self.assertTrue(boxes_intersect(actor,neighbor))
        self.assertEqual(StaticBoxCollection([{'box':neighbor}]).candidates(actor),[0])
        self.assertEqual(StaticBoxCollection([]).candidates(actor),[])

    def test_resume_prefix_preserves_completed_rows_and_refuses_order_change(self):
        jobs=[{'snapshot_path':'first','snapshot_sha256':'a'},{'snapshot_path':'second','snapshot_sha256':'b'}]
        rows=[{'source_path':'first','source_sha256':'a','errors':[]}]
        command=['python','audit.py','--mode','spawn','--physical-context','full_snapshot']
        with patch.object(Path,'read_text',side_effect=[json.dumps(jobs[0])+'\n'+json.dumps(jobs[1]),json.dumps(rows[0])]), \
                patch.object(Path,'is_file',return_value=True), \
                patch.object(Path,'open',return_value=io.StringIO(json.dumps(command))):
            self.assertEqual(audit.resume_prefix(Path('old'),jobs,'spawn','full_snapshot'),rows)
        with patch.object(Path,'read_text',return_value=json.dumps(jobs[1])+'\n'+json.dumps(jobs[0])):
            with self.assertRaises(ValueError):audit.resume_prefix(Path('old'),jobs,'spawn','full_snapshot')

    def test_resume_refuses_context_change(self):
        jobs=[{'snapshot_path':'first','snapshot_sha256':'a'}]
        command=['python','audit.py','--mode','spawn','--physical-context','manifest']
        with patch.object(Path,'read_text',return_value=json.dumps(jobs[0])), \
                patch.object(Path,'is_file',return_value=True), \
                patch.object(Path,'open',return_value=io.StringIO(json.dumps(command))):
            with self.assertRaisesRegex(ValueError,'context'):
                audit.resume_prefix(Path('old'),jobs,'spawn','full_snapshot')

    def test_stage_signature_compares_input_content_not_new_output_paths(self):
        with patch('run_hv_mainline_batch.sha',return_value='same-input'):
            a=stage_signature(['py','audit','--manifest','old/input','--mode','spawn','--output-dir','old/output'])
            b=stage_signature(['py','audit','--manifest','new/input','--mode','spawn','--output-dir','new/output'])
            self.assertEqual(a,b)

    def test_resume_allows_transport_change_but_requires_same_experiment(self):
        base=['py','audit','--mode','spawn','--physical-context','full_snapshot']
        a=stage_signature(base+['--port','31100','--host','127.0.0.1','--timeout','300'])
        b=stage_signature(base+['--port','31200','--host','localhost','--timeout','600'])
        self.assertEqual(a,b)
        self.assertNotEqual(a,stage_signature(['py','audit','--mode','spawn','--physical-context','manifest']))
        with patch('run_hv_mainline_batch.sha',side_effect=['original','changed']):
            self.assertNotEqual(stage_signature(base+['--manifest','old']),stage_signature(base+['--manifest','new']))


if __name__=='__main__':unittest.main()
