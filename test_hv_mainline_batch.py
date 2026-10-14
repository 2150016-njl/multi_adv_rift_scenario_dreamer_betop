import io
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock,patch

import run_hv_mainline_batch as batch
import test_hv_snapshot_experiment as fixtures
from build_hv_snapshot_audit_manifest import pair_job


class MainlineBatch(unittest.TestCase):
    def test_unique_s1_is_required_even_when_two_samples_have_same_scene(self):
        sample={'record_idx':7,'sample_idx':0}
        with self.assertRaises(ValueError):batch.selected_samples([sample,sample],[7])
        with self.assertRaises(ValueError):batch.selected_samples([dict(sample,sample_idx=1)],[7])
        with self.assertRaises(ValueError):batch.selected_samples([],[7])

    def test_v3_reuses_source_and_context_without_scanning_collection(self):
        scene,record,sample=fixtures.TestManifestAndPrecision().fixture()
        data=json.dumps(scene).encode();base=pair_job('memory.json',data,record,sample);base['record_index']=0
        sample=dict(sample,record_index=0,sample_index=0,town=scene['town'])
        sample['generated']={'generated_agent_states':[[14.,1.,0.,3.,4.6,1.9,1.],[24.,1.,0.,3.,4.6,1.9,1.]]}
        written={}
        def write(path,text,**kwargs):written[str(path)]=text
        with patch('hv_audit_inputs.manifest_jobs',return_value=[base]), \
                patch.object(Path,'read_text',return_value=json.dumps({'dataset_sha256':'dataset','passed':True,
                    'full_evaluation_eligible':False,'original_requested_records':100})), \
                patch.object(Path,'open',return_value=io.StringIO(json.dumps(record)+'\n')), \
                patch.object(Path,'read_bytes',return_value=data), \
                patch.object(Path,'exists',return_value=False), \
                patch.object(Path,'write_text',autospec=True,side_effect=write), \
                patch.object(batch,'sha',side_effect=lambda p:'dataset' if str(p)=='dataset' else 'hash'), \
                patch.object(batch,'read_jsonl',return_value=[sample]), \
                patch('hv_audit_inputs.snapshot_paths',side_effect=AssertionError('No collection rescan')):
            result=batch.rebind_generated(Path('base'),Path('dataset'),Path('generated'),Path('output'))
        rebound=json.loads(written['output'])
        self.assertEqual(result['jobs'],1)
        self.assertEqual(rebound['context_actor_ids'],base['context_actor_ids'])
        self.assertEqual(rebound['snapshot_sha256'],base['snapshot_sha256'])
        self.assertEqual(rebound['generated_agent_states'],sample['generated']['generated_agent_states'])
        self.assertFalse(json.loads(written['output.audit.json'])['full_evaluation_eligible'])

    def test_raw_pool_deduplicates_source_and_restores_full_fleet_semantics(self):
        source={'snapshot_path':'source','snapshot_sha256':'sha','snapshot_replay_sha256':'physical',
                'record_index':1,'target_actor_ids':[1,2],'context_actor_ids':[3],'generated_agent_states':[[1],[2]],
                'town':'Town12','episode_key':'episode'}
        written={}
        def write(path,text,**kwargs):written[str(path)]=text
        with patch('hv_audit_inputs.manifest_jobs',return_value=[source,dict(source,record_index=2)]), \
                patch.object(Path,'exists',return_value=False), \
                patch.object(Path,'write_text',autospec=True,side_effect=write),patch.object(batch,'sha',return_value='hash'):
            count=batch.raw_source_manifest(Path('base'),Path('output'))
        self.assertEqual(count,1)
        row=json.loads(written['output'])
        self.assertNotIn('target_actor_ids',row)
        self.assertNotIn('context_actor_ids',row)
        self.assertNotIn('generated_agent_states',row)

    def test_joint_truth_keeps_unknown_and_known_failure_distinct(self):
        self.assertIsNone(batch.combine([True,None]))
        self.assertFalse(batch.combine([False,None]))
        self.assertTrue(batch.combine([True,None],True))
        self.assertIsNone(batch.combine([False,None],True))

    def test_geometry_string_ids_from_json_are_matched_to_integer_actor_ids(self):
        actor={'actor_id':5,'forward_fov':True,'native':{'evaluated':True,'native_candidate':True}}
        row={'spawn':{'selected_gt':{'context_complete':True,
            'attempts':[{'role':'target','source_actor_id':5,'accepted':True}],
            'geometry_3d':{'5':{'whole_road':{'whole_box_in_sampled_driving_polygon':True},'peer_3d_overlap_ids':[]}}}}}
        gates=batch.actor_gates(actor,row,'selected_gt')
        self.assertTrue(gates['raw_spawn']);self.assertTrue(gates['whole_polygon']);self.assertTrue(gates['peer_3d_clear'])

    def test_unwritten_or_failed_record_stays_in_scene_denominator(self):
        result=batch.cohort_diagnosis([{'errors':['load failed']}],'generated','native')
        metric=result['scene_metrics']['at_least_one_front_native']
        self.assertEqual(metric['planned_scenes'],1);self.assertEqual(metric['unknown'],1)
        self.assertIsNone(metric['complete_rate'])

    def test_cluster_comparison_counts_episode_blocks_not_sparse_copies(self):
        left=[];right=[]
        for i in range(6):
            row={'record_index':i,'block':'episodeA' if i<5 else 'episodeB','snapshot_sha256':str(i),
                 'context_sha256':'condition','metric':False}
            left.append(row);right.append(dict(row,metric=True))
        result=batch.paired_comparison(left,right,'metric',replicates=100)
        self.assertEqual(result['episode_blocks'],2)
        self.assertEqual(result['known_both_records'],6)
        self.assertEqual(result['v3_minus_v2_on_known_pairs'],1.)
        self.assertEqual(result['episode_block_bootstrap_95_interval'],[1.,1.])

    def test_paired_comparison_rejects_context_mismatch(self):
        row={'record_index':1,'block':'A','snapshot_sha256':'source','context_sha256':'one','metric':True}
        with self.assertRaises(ValueError):batch.paired_comparison([row],[dict(row,context_sha256='two')],'metric')

    def test_independent_stages_continue_after_failure_and_batch_remains_failed(self):
        written=[]
        def write(path,text,**kwargs):written.append((path.name,json.loads(text)))
        def spawn(command,**kwargs):
            process=MagicMock();process.stdout=io.StringIO('stage output\n')
            is_failed='--output-dir' in command and Path(command[command.index('--output-dir')+1]).name=='pool_native'
            process.wait.return_value=1 if is_failed else 0
            return process
        with patch('sys.argv',['batch','--pair-report','report','--output-dir','memory','--with-physics']), \
                patch.object(Path,'mkdir'),patch.object(Path,'open',side_effect=lambda *args,**kwargs:io.StringIO()), \
                patch.object(Path,'write_text',autospec=True,side_effect=write), \
                patch.object(batch.subprocess,'Popen',side_effect=spawn), \
                patch.object(batch,'raw_source_manifest',return_value=1), \
                patch.object(batch,'rebind_generated',return_value={'jobs':1}), \
                patch.object(batch,'analyze',return_value={'stages':{}}),patch('builtins.print'):
            with self.assertRaises(SystemExit):batch.main()
        status=[v for name,v in written if name=='batch_status.json'][-1]
        self.assertEqual(status['status'],'incomplete')
        self.assertEqual(status['stages']['pool_native']['status'],'failed')
        self.assertEqual(status['stages']['v3_physics']['status'],'completed')
        self.assertTrue(status['physics_requested'])
        self.assertTrue(any(name=='analysis.json' for name,value in written))

    def test_resume_copies_completed_stage_and_forwards_partial_stage_prefix(self):
        previous={'stages':{'pool_snapshot':{'status':'completed','command':['py','audit']},
                            'pool_spawn':{'status':'running','command':['py','audit']}}}
        written=[];commands=[]
        def write(path,text,**kwargs):written.append((path.name,json.loads(text)))
        def is_file(path):
            return str(path).startswith(str(Path('old'))) and path.parent.name in ('pool_snapshot','pool_spawn') and \
                path.name in ('jobs.jsonl','rows.jsonl','summary.json')
        def read_text(path,**kwargs):
            return json.dumps(previous if path.name=='batch_status.json' else {'planned_jobs':1})
        def spawn(command,**kwargs):
            commands.append(command);process=MagicMock();process.stdout=io.StringIO('output\n');process.wait.return_value=0
            return process
        with patch('sys.argv',['batch','--pair-report','report','--output-dir','memory','--resume-from','old']), \
                patch.object(Path,'mkdir'),patch.object(Path,'is_file',autospec=True,side_effect=is_file), \
                patch.object(Path,'read_text',autospec=True,side_effect=read_text), \
                patch.object(Path,'open',side_effect=lambda *args,**kwargs:io.StringIO()), \
                patch.object(Path,'write_text',autospec=True,side_effect=write), \
                patch.object(batch.subprocess,'Popen',side_effect=spawn), \
                patch.object(batch,'raw_source_manifest',return_value=1), \
                patch.object(batch,'rebind_generated',return_value={'jobs':1}), \
                patch.object(batch,'read_jsonl',return_value=[{}]), \
                patch.object(batch,'stage_signature',return_value=['same']), \
                patch.object(batch.shutil,'copy2') as copy, \
                patch.object(batch,'analyze',return_value={'stages':{}}),patch('builtins.print'):
            batch.main()
        state=[v for name,v in written if name=='batch_status.json'][-1]
        self.assertIn('reused_from',state['stages']['pool_snapshot'])
        resumed=[c for c in commands if '--resume-from' in c]
        self.assertEqual(len(resumed),1)
        self.assertEqual(resumed[0][-1],str(Path('old/pool_spawn')))
        self.assertEqual(copy.call_count,3)
        self.assertFalse(state['physics_requested'])
        self.assertEqual(state['excluded_stages'],['pool_physics','v2_physics','v3_physics'])
        self.assertIn('v3_spawn',state['stages'])
        self.assertIn('v3_native',state['stages'])
        self.assertIn('v3_spawn_sparse',state['stages'])
        self.assertFalse(any(name.endswith('_physics') for name in state['stages']))
        self.assertFalse(any('--mode' in c and c[c.index('--mode')+1]=='physics' for c in commands))


if __name__=='__main__':unittest.main()
