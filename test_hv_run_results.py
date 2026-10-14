import json
from pathlib import Path
import unittest
from unittest.mock import patch

from report_hv_run_results import collect


class RunResults(unittest.TestCase):
    def test_abort_skips_are_separate_from_first_actual_error(self):
        root = Path('memory')
        folder = root/'experiment'/'v3_spawn_sparse'
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'iterdir',return_value=iter([folder])), \
                patch.object(Path,'is_file',autospec=True,side_effect=lambda p:p.name in ('jobs.jsonl','rows.jsonl')), \
                patch('report_hv_run_results.read_jsonl') as read:
            jobs = [{'record_index': i} for i in range(5)]
            rows = [{'errors': []}, {'errors': ['Owned actor cleanup failed; stop replay'],
                                     'abort_remaining_jobs': True},
                    {'errors': ['Skipped after replay isolation/cleanup failure']},
                    {'errors': ['Skipped after replay isolation/cleanup failure']}]
            read.side_effect=lambda p: rows if p.name=='rows.jsonl' else jobs
            with patch('report_hv_run_results.analyze', return_value={'valid': True}):
                report = collect(root)
            stage = report['execution']['v3_spawn_sparse']
            self.assertEqual(stage['executed_with_errors'], 1)
            self.assertEqual(stage['skipped_after_abort'], 2)
            self.assertEqual(stage['unwritten'], 1)
            self.assertEqual(stage['rows_without_execution_errors'], 1)
            self.assertEqual(stage['first_executed_error']['job']['record_index'], 1)
            self.assertEqual(report['metrics'], {'valid': True})
            with patch.object(Path,'iterdir',return_value=iter([folder])), \
                    patch('report_hv_run_results.analyze', side_effect=ValueError('pair mismatch')):
                report = collect(root)
            self.assertIn('pair mismatch', report['metrics_error'])
            self.assertIn('v3_spawn_sparse', report['execution'])


if __name__ == '__main__':
    unittest.main()
