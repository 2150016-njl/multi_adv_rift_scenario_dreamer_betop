"""Server readiness, owned cleanup and repeated difference-report regression."""
import hashlib
import errno
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch
from run_hv_native_diagnostics import wait_until_ready,stop_owned_server,passive_listeners,launch_command,startup_evidence,prepare_output
from diagnose_hv_pair_ambiguity import saved_result
import run_hv_native_diagnostics as managed


class ManagedReadiness(unittest.TestCase):
    def test_auto_port_skips_occupied_block_and_closes_every_probe(self):
        sockets=[MagicMock() for _ in range(5)]
        sockets[1].bind.side_effect=OSError(errno.EADDRINUSE,'occupied streaming port')
        with patch.object(managed.socket,'socket',side_effect=sockets):
            self.assertEqual(managed.choose_free_port(31200),31203)
        for sock in sockets:sock.close.assert_called_once()
        sockets[4].bind.assert_called_once_with(('0.0.0.0',31205))

    def test_auto_port_search_is_bounded_and_does_not_contact_servers(self):
        sock=MagicMock();sock.bind.side_effect=OSError(errno.EADDRINUSE,'occupied')
        with patch.object(managed.socket,'socket',return_value=sock):
            with self.assertRaisesRegex(RuntimeError,'No free'):
                managed.choose_free_port(31200,attempts=2)
        self.assertEqual(sock.close.call_count,2)
        sock.connect.assert_not_called()

    def test_auto_port_leaves_reserved_service_unprobed(self):
        sock=MagicMock()
        with patch.object(managed.socket,'socket',return_value=sock):
            self.assertEqual(managed.choose_free_port(38000),38003)
        self.assertEqual([call.args[0][1] for call in sock.bind.call_args_list],[38003,38004,38005])

    def test_auto_port_does_not_hide_permission_errors(self):
        sock=MagicMock();sock.bind.side_effect=OSError(errno.EACCES,'permission denied')
        with patch.object(managed.socket,'socket',return_value=sock):
            with self.assertRaises(PermissionError):managed.choose_free_port(31200)
        sock.close.assert_called_once()

    def test_input_failure_is_recorded_before_dependency_or_server_launch(self):
        written=[]
        def write(path,content,**kwargs):
            written.append((path.name,json.loads(content)))
        with patch('sys.argv',['runner','--pair-report','missing.json','--output-dir','memory-output']), \
                patch.object(managed.sys,'platform','linux'), \
                patch.object(managed.os,'geteuid',return_value=1000,create=True), \
                patch.object(managed,'prepare_output'), \
                patch.object(Path,'write_text',autospec=True,side_effect=write), \
                patch.object(Path,'is_file',return_value=False), \
                patch.object(managed,'startup_evidence',return_value={}), \
                patch.object(managed,'run_logged') as preflight, \
                patch.object(managed.subprocess,'Popen') as launch,patch('builtins.print'):
            with self.assertRaises(FileNotFoundError):managed.main()
        states=[value for name,value in written if name=='status.json']
        self.assertEqual(states[0]['status'],'initializing')
        self.assertEqual(states[1]['status'],'checking_pair_report_path')
        self.assertEqual(states[-1]['failure_stage'],'checking_pair_report_path')
        self.assertEqual(states[-1]['status'],'failed_before_or_during_startup')
        preflight.assert_not_called();launch.assert_not_called()

    def test_startup_attempts_replace_client_created_before_readiness(self):
        clock=[0.];early=MagicMock();ready=MagicMock()
        early.get_server_version.side_effect=RuntimeError('starting')
        ready.get_server_version.return_value='0.9.15'
        ready.get_client_version.return_value='0.9.15'
        ready.get_world.return_value.get_map.return_value.name='Town03'
        factory=MagicMock(side_effect=[early,ready])
        with patch('builtins.print'):
            result=wait_until_ready(None,None,10.,interval=1.,now=lambda:clock[0],
                sleep=lambda t:clock.__setitem__(0,clock[0]+t),client_factory=factory)
        self.assertEqual(factory.call_count,2)
        self.assertEqual(result['map_name'],'Town03')
        early.get_world.assert_not_called()

    def test_default_launch_uses_existing_no_rendering_contract(self):
        self.assertEqual(launch_command(Path('carla'),31100),
                         [str(Path('carla/CarlaUE4.sh')),'-nosound','-carla-rpc-port=31100','-nullrhi'])

    def test_prepared_output_refuses_nonempty_directory_without_mutating_it(self):
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'is_symlink',return_value=False), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_uid=1000)), \
                patch.object(Path,'iterdir',return_value=iter([Path('status.json')])), \
                patch('run_hv_native_diagnostics.os.geteuid',return_value=1000,create=True), \
                patch.object(Path,'mkdir') as mkdir:
            with self.assertRaises(FileExistsError):prepare_output(Path('output'),True)
        mkdir.assert_not_called()

    def test_prepared_output_refuses_wrong_owner(self):
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'is_symlink',return_value=False), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_uid=0)), \
                patch('run_hv_native_diagnostics.os.geteuid',return_value=1000,create=True):
            with self.assertRaises(PermissionError):prepare_output(Path('output'),True)

    def test_prepared_output_accepts_only_empty_user_owned_directory(self):
        with patch.object(Path,'is_dir',return_value=True), \
                patch.object(Path,'is_symlink',return_value=False), \
                patch.object(Path,'stat',return_value=SimpleNamespace(st_uid=1000)), \
                patch.object(Path,'iterdir',return_value=iter([])), \
                patch('run_hv_native_diagnostics.os.geteuid',return_value=1000,create=True), \
                patch.object(Path,'mkdir') as mkdir:
            prepare_output(Path('output'),True)
        mkdir.assert_not_called()

    def test_passive_listener_does_not_connect_to_server(self):
        table='header\n 0: 00000000:7918 00000000:0000 0A 0:0 0:0 0 1000 0 123\n'
        with patch.object(Path,'read_text',return_value=table), \
                patch('run_hv_native_diagnostics.socket.create_connection') as connect:
            result=passive_listeners(31000)
        self.assertTrue(result['tcp_listening'])
        self.assertEqual(result['listeners'][0]['socket_inode'],'123')
        connect.assert_not_called()

    def test_unreadable_listener_table_is_unknown_not_false(self):
        with patch.object(Path,'read_text',side_effect=PermissionError('blocked')):
            self.assertIsNone(passive_listeners(31000)['tcp_listening'])

    def test_startup_evidence_never_opens_probe_connection(self):
        with patch.object(Path,'read_text',return_value='header\n'), \
                patch.object(Path,'is_file',return_value=False), \
                patch('run_hv_native_diagnostics.shutil.which',return_value=None), \
                patch('run_hv_native_diagnostics.socket.create_connection') as connect:
            result=startup_evidence(31000,None,Path('server.log'))
        self.assertFalse(result['tcp_listening'])
        connect.assert_not_called()

    def test_offscreen_and_nullrhi_are_explicit_separate_launch_modes(self):
        offscreen=launch_command(Path('carla'),31000,'offscreen',0)
        self.assertIn('-RenderOffScreen',offscreen)
        self.assertIn('-graphicsadapter=0',offscreen)
        self.assertNotIn('-nullrhi',offscreen)
        null=launch_command(Path('carla'),31000,'nullrhi')
        self.assertIn('-nullrhi',null)
        self.assertNotIn('-RenderOffScreen',null)

    def test_reports_rpc_stage_and_error_to_progress_callback(self):
        clock=[0.];client=MagicMock();reports=[]
        client.get_server_version.return_value='0.9.15'
        client.get_world.side_effect=RuntimeError('world unavailable')
        with patch('builtins.print'),self.assertRaises(TimeoutError):
            wait_until_ready(client,None,1.,interval=1.,now=lambda:clock[0],
                             sleep=lambda t:clock.__setitem__(0,clock[0]+t),
                             on_progress=reports.append)
        self.assertEqual(reports[0]['rpc_stage'],'get_world')
        self.assertIn('world unavailable',reports[0]['last_error'])

    def test_failed_version_probe_is_distinguished_from_world_probe(self):
        clock=[0.];client=MagicMock();reports=[]
        client.get_server_version.side_effect=RuntimeError('version unavailable')
        with patch('builtins.print'),self.assertRaises(TimeoutError):
            wait_until_ready(client,None,1.,interval=1.,now=lambda:clock[0],
                             sleep=lambda t:clock.__setitem__(0,clock[0]+t),
                             on_progress=reports.append)
        self.assertEqual(reports[0]['rpc_stage'],'get_server_version')
        client.get_world.assert_not_called()

    def test_waits_for_rpc_and_world_instead_of_fixed_sleep(self):
        clock=[0.];client=MagicMock()
        client.get_server_version.side_effect=[RuntimeError('not ready'),RuntimeError('not ready'),'0.9.15']
        client.get_client_version.return_value='0.9.15'
        client.get_world.return_value.get_map.return_value.name='Town03'
        process=SimpleNamespace(poll=lambda:None)
        with patch('builtins.print'):
            r=wait_until_ready(client,process,10.,interval=1.,now=lambda:clock[0],
                               sleep=lambda t:clock.__setitem__(0,clock[0]+t))
        self.assertEqual(r['startup_wait_seconds'],2.)
        self.assertEqual(r['map_name'],'Town03')

    def test_dead_server_fails_without_repeated_rpc_wait(self):
        client=MagicMock();process=SimpleNamespace(poll=lambda:1,returncode=1)
        with self.assertRaisesRegex(RuntimeError,'server.log'):
            wait_until_ready(client,process,600.)
        client.get_server_version.assert_not_called()

    def test_unresponsive_server_has_bounded_wait(self):
        clock=[0.];client=MagicMock();client.get_server_version.side_effect=RuntimeError('no RPC')
        with patch('builtins.print'),self.assertRaises(TimeoutError):
            wait_until_ready(client,None,3.,interval=1.,now=lambda:clock[0],
                             sleep=lambda t:clock.__setitem__(0,clock[0]+t))
        self.assertEqual(clock[0],3.)

    def test_cleanup_only_targets_owned_process_group(self):
        process=MagicMock(pid=4321);process.poll.return_value=None
        with patch('run_hv_native_diagnostics.os.killpg',create=True) as kill:
            stop_owned_server(process)
        self.assertEqual(kill.call_args.args[0],4321)
        process.wait.assert_called_once()

    def test_existing_server_is_not_stopped(self):
        with patch('run_hv_native_diagnostics.os.killpg',create=True) as kill:
            stop_owned_server(None)
        kill.assert_not_called()


class CachedDifferences(unittest.TestCase):
    def test_same_report_can_be_read_again_without_overwriting(self):
        data=b'original';value={'source_pair_report_sha256':hashlib.sha256(data).hexdigest(),'records':{}}
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=json.dumps(value)), \
                patch.object(Path,'write_text') as write:
            self.assertEqual(saved_result(Path('diff.json'),Path('report.json'),data),value)
        write.assert_not_called()

    def test_changed_source_report_does_not_reuse_stale_result(self):
        value={'source_pair_report_sha256':hashlib.sha256(b'old').hexdigest(),'records':{}}
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=json.dumps(value)):
            with self.assertRaisesRegex(FileExistsError,'changed'):
                saved_result(Path('diff.json'),Path('report.json'),b'new')

    def test_legacy_report_without_hash_is_still_readable_at_original_path(self):
        source=Path('report.json');value={'source_pair_report':str(source.resolve()),'records':{}}
        with patch.object(Path,'exists',return_value=True),patch.object(Path,'read_text',return_value=json.dumps(value)):
            self.assertEqual(saved_result(Path('diff.json'),source,b'original'),value)


if __name__=='__main__':unittest.main()
