"""No-disk regression tests for explicit subset scope and dependency preflight."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

from prepare_hv_diagnostic_subset import validate_subset
from check_hv_attention_assets import check_assets
from diagnose_hv_pair_ambiguity import differences
import audit_hv_one_shot_results as online
from test_hv_snapshot_followup import method


class ResumeEvidence(unittest.TestCase):
    def fixture(self):
        jobs=[dict(record_index=i,snapshot_path='source.json',snapshot_sha256='a'*64) for i in (0,5)]
        data=''.join(json.dumps(j)+'\n' for j in jobs).encode()
        report=dict(passed=False,partial_manifest_sha256=hashlib.sha256(data).hexdigest(),written_jobs=2,
                    selected_dataset_indices=list(range(10)),requested_records=10,
                    missing_record_indices=[],ambiguous_record_indices=[1,2,3,4,6,7,8,9],
                    selection='prefix',dataset_sha256='dataset',generated_sha256='generated')
        return report,data

    def test_subset_retains_original_denominator_and_incomplete_scope(self):
        report,data=self.fixture();jobs,scope=validate_subset(report,data)
        self.assertEqual([j['record_index'] for j in jobs],[0,5])
        self.assertEqual(scope['original_requested_records'],10)
        self.assertFalse(scope['full_evaluation_eligible'])
        self.assertEqual(len(scope['excluded_original_indices']),8)

    def test_modified_partial_is_rejected(self):
        report,data=self.fixture()
        with self.assertRaisesRegex(ValueError,'SHA'):
            validate_subset(report,data+b'\n')

    def test_ambiguous_index_cannot_be_added_to_subset(self):
        report,data=self.fixture();report['ambiguous_record_indices'].append(0)
        with self.assertRaisesRegex(ValueError,'partition'):
            validate_subset(report,data)

    def test_changed_requested_denominator_is_rejected(self):
        report,data=self.fixture();report['requested_records']=11
        with self.assertRaisesRegex(ValueError,'partition'):
            validate_subset(report,data)

    def test_source_differences_retain_small_numeric_deltas(self):
        rows=differences({'ego':{'z':1.}},{'ego':{'z':1.+1.e-9}})
        self.assertEqual(rows[0]['path'],'.ego.z')
        self.assertGreater(rows[0]['absolute_delta'],0.)


class AttentionPreflight(unittest.TestCase):
    def test_missing_assets_fail_without_starting_carla(self):
        with patch.object(Path,'open',side_effect=FileNotFoundError('missing')),patch.object(Path,'is_file',return_value=False):
            self.assertEqual(len(check_assets('weights.ckpt','bert-medium')),2)

    def test_local_config_and_nonempty_weight_pass_basic_preflight(self):
        with patch.object(Path,'open',side_effect=lambda *a,**k:io.BytesIO(b'x')), \
                patch.object(Path,'is_file',return_value=True), \
                patch.object(Path,'read_text',return_value='{"model_type":"bert"}'):
            self.assertEqual(check_assets('weights.ckpt','bert-medium'),[])

    def test_explicit_model_paths_do_not_mutate_original_config(self):
        fn=method('RIFT/rift/cbv/recognition/attention_based/attn_model.py','AttnModel','__init__',{})
        config=dict(network={'hf_checkpoint':'old_hf'},pretrained_model_path='old.ckpt',obs_type='plant',
                    viz_attn_map=False,frame_rate=10,cbv_attention_checkpoint='new.ckpt',cbv_attention_hf_dir='new_hf')
        original=copy.deepcopy(config);obj=SimpleNamespace();fn(obj,config,None)
        self.assertEqual(config,original);self.assertEqual(obj.model_path,'new.ckpt')
        self.assertEqual(obj.config['network']['hf_checkpoint'],'new_hf')

    def test_failure_before_first_request_is_not_model_failure_rate(self):
        directory=MagicMock();directory.glob.return_value=[]
        directory.__truediv__.return_value.exists.return_value=False
        result=online.audit(directory,simulator_exit_code=1)
        self.assertEqual(result['evaluation_status'],'not_started_runner_failed')
        self.assertIsNone(result['native_green_rate_per_eligible_request'])
        self.assertEqual(result['num_generator_called'],0)


if __name__=='__main__':unittest.main()
