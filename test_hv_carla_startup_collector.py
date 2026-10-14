import io
from pathlib import Path
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from diagnose_hv_carla_startup import command_result, existing_logs, tail_file


class StartupCollector(unittest.TestCase):
    def test_missing_optional_tool_is_evidence_instead_of_failure(self):
        with patch('diagnose_hv_carla_startup.subprocess.run',side_effect=FileNotFoundError('missing')):
            report=command_result(['vulkaninfo','--summary'])
        self.assertIn('missing',report['error'])

    def test_slow_tool_is_bounded_and_reported(self):
        with patch('diagnose_hv_carla_startup.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(['nvidia-smi'],12)) as run:
            report=command_result(['nvidia-smi'])
        self.assertEqual(run.call_args.kwargs['timeout'],12)
        self.assertIn('error',report)

    def test_log_tail_includes_mtime_for_attribution_and_bounds_read(self):
        data=b'old line\nlatest line\n'
        with patch.object(Path,'stat',return_value=SimpleNamespace(st_size=len(data),st_mtime=1)), \
                patch.object(Path,'open',return_value=io.BytesIO(data)):
            report=tail_file(Path('engine.log'),12)
        self.assertEqual(report['tail'],'latest line\n')
        self.assertEqual(report['mtime_epoch'],1)

    def test_engine_search_does_not_include_unrelated_files(self):
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_mtime=1)), \
                patch('diagnose_hv_carla_startup.os.walk',return_value=[('Saved/Logs',[],['engine.log','binary.pak'])]), \
                patch('diagnose_hv_carla_startup.tail_file',side_effect=lambda p:{'path':str(p)}):
            report=existing_logs([Path('Saved/Logs')])
        self.assertEqual(report['files'],[{'path':str(Path('Saved/Logs/engine.log'))}])


if __name__=='__main__':
    unittest.main()
