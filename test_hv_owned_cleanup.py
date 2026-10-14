import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parent/'scenario-dreamer'))
from high_value_init.native_state_audit import destroy_owned_without_tick


class OwnedCleanup(unittest.TestCase):
    def test_deletes_only_created_ids_without_tick_or_cached_actor_state(self):
        actors=[MagicMock(id=101),MagicMock(id=102)]
        client=MagicMock()
        client.apply_batch_sync.return_value=[SimpleNamespace(has_error=lambda:False)]*2
        carla=SimpleNamespace(command=SimpleNamespace(DestroyActor=lambda aid:('destroy',aid)))
        errors,evidence=destroy_owned_without_tick(client,carla,actors)
        self.assertEqual(errors,[])
        client.apply_batch_sync.assert_called_once_with([('destroy',102),('destroy',101)],False)
        self.assertEqual(evidence['actor_ids'],[102,101])
        for actor in actors:actor.destroy.assert_not_called()

    def test_server_rejection_and_missing_acknowledgment_are_not_suppressed(self):
        client=MagicMock()
        client.apply_batch_sync.return_value=[SimpleNamespace(has_error=lambda:True,error='not found')]
        carla=SimpleNamespace(command=SimpleNamespace(DestroyActor=lambda aid:aid))
        errors,evidence=destroy_owned_without_tick(client,carla,[SimpleNamespace(id=1),SimpleNamespace(id=2)])
        self.assertEqual(len(errors),2)
        self.assertIn('not found',errors[1])

    def test_empty_cohort_does_not_send_commands(self):
        client=MagicMock()
        errors,evidence=destroy_owned_without_tick(client,None,[])
        self.assertEqual(errors,[])
        client.apply_batch_sync.assert_not_called()


if __name__=='__main__':unittest.main()
