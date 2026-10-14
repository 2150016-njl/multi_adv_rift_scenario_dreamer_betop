"""CARLA source-path regression tests without CARLA or filesystem writes."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import audit_hv_snapshot_experiment  # Adds the project module paths.
from high_value_init.carla_audit_environment import configure_carla_pythonapi


class NativeEnvironment(unittest.TestCase):
    def test_installed_carla_still_gets_agents_sources(self):
        paths=[]
        with patch('high_value_init.carla_audit_environment.sys.path',paths), \
                patch.dict('os.environ',{'CARLA_ROOT':'sdk'},clear=True), \
                patch.object(Path,'is_dir',return_value=True),patch.object(Path,'resolve',lambda p:p), \
                patch('importlib.util.find_spec',return_value=object()), \
                patch('importlib.invalidate_caches'), \
                patch('importlib.import_module',side_effect=lambda name:SimpleNamespace(__file__=name+'.py')) as imported:
            result=configure_carla_pythonapi('project',validate=True)
        self.assertEqual(paths[0],str(Path('sdk/PythonAPI/carla')))
        self.assertIn(str(Path('sdk/PythonAPI')),paths)
        self.assertEqual(len(imported.call_args_list),3)
        self.assertIn('agents.navigation.local_planner_source',result)
        self.assertFalse(any(p.endswith('.egg') for p in paths))

    def test_explicit_egg_is_used_only_when_binary_missing(self):
        paths=[]
        with patch('high_value_init.carla_audit_environment.sys.path',paths), \
                patch.dict('os.environ',{'CARLA_ROOT':'sdk','CARLA_EGG':'sdk/carla.egg'},clear=True), \
                patch.object(Path,'is_dir',return_value=False),patch.object(Path,'exists',return_value=True), \
                patch.object(Path,'resolve',lambda p:p),patch('importlib.util.find_spec',return_value=None), \
                patch('importlib.invalidate_caches'):
            configure_carla_pythonapi('project')
        self.assertEqual(paths,[str(Path('sdk/carla.egg'))])

    def test_missing_agents_reports_source_directory(self):
        with patch('high_value_init.carla_audit_environment.sys.path',[]), \
                patch.dict('os.environ',{'CARLA_ROOT':'sdk'},clear=True), \
                patch.object(Path,'is_dir',return_value=False),patch('importlib.util.find_spec',return_value=object()), \
                patch('importlib.invalidate_caches'), \
                patch('importlib.import_module',side_effect=ModuleNotFoundError('agents')):
            with self.assertRaisesRegex(RuntimeError,'PythonAPI'):
                configure_carla_pythonapi('project',validate=True)


if __name__=='__main__':unittest.main()
