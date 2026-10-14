#!/usr/bin/env python3
"""Start an owned CARLA server, wait for RPC readiness, run audits, then stop it."""
import argparse
import datetime
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time


def choose_free_port(start, attempts=100):
    """Find a free TCP block for RPC/streaming without contacting any server."""
    for port in range(start, min(start+3*attempts,65534),3):
        # The user's existing service is reserved, including its adjacent ports.
        if set(range(port,port+3)).intersection(range(38000,38003)):
            continue
        checks=[]
        try:
            for candidate in range(port,port+3):
                check=socket.socket();checks.append(check)
                check.bind(('0.0.0.0',candidate))
            return port
        except OSError as exc:
            if exc.errno!=errno.EADDRINUSE:raise
        finally:
            for check in checks:check.close()
    raise RuntimeError('No free three-port TCP block starting at %d; choose another --port'%start)


def wait_until_ready(client,process,timeout,interval=2.,now=None,sleep=None,on_progress=None,client_factory=None):
    now=now or time.monotonic;sleep=sleep or time.sleep
    start=now();deadline=start+timeout;next_message=start;last='no response'
    while now()<deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError('Owned CARLA exited with code %s; inspect server.log' % process.returncode)
        stage='get_server_version'
        try:
            if client_factory is not None:client=client_factory()
            server=client.get_server_version()
            stage='get_world'
            world=client.get_world()
            stage='get_map'
            map_name=world.get_map().name
            return {'server_version':server,'client_version':client.get_client_version(),
                    'map_name':map_name,'startup_wait_seconds':now()-start}
        except Exception as exc:
            last='%s: %s' % (type(exc).__name__,exc)
        if now()>=next_message:
            progress={'elapsed_seconds':now()-start,'rpc_stage':stage,'last_error':last}
            print('Waiting for CARLA RPC (%.0fs), stage=%s: %s' % (now()-start,stage,last),flush=True)
            if on_progress is not None:on_progress(progress)
            next_message=now()+15.
        remaining=deadline-now()
        if remaining>0:sleep(min(interval,remaining))
    raise TimeoutError('CARLA did not become ready within %.0fs: %s; inspect server.log' % (timeout,last))


def startup_evidence(port,process,log_path):
    """Read-only evidence collected before owned-server cleanup changes its state."""
    evidence={'server_returncode':process.poll() if process is not None else None}
    evidence.update(passive_listeners(port))
    ss=shutil.which('ss')
    if ss:
        try:
            result=subprocess.run([ss,'-ltnp','sport = :%d'%port],capture_output=True,
                                  text=True,timeout=3.)
            evidence['listener_details']=result.stdout[-8000:]
        except (OSError,subprocess.TimeoutExpired) as exc:
            evidence['listener_error']=str(exc)
    if log_path.is_file():
        with log_path.open('rb') as log:
            log.seek(0,2);log.seek(max(0,log.tell()-16000))
            evidence['server_log_tail']=log.read().decode('utf-8',errors='replace')
    return evidence


def passive_listeners(port,proc_root=Path('/proc')):
    """Observe Linux socket tables; never open/close a probe connection to CARLA."""
    listeners=[];errors=[];tables_read=0
    for family in ('tcp','tcp6'):
        try:
            lines=(proc_root/'net'/family).read_text(encoding='ascii').splitlines()[1:]
            tables_read+=1
            for line in lines:
                fields=line.split()
                if len(fields)>9 and fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==port:
                    listeners.append({'family':family,'local_address_hex':fields[1],
                                      'uid':fields[7],'socket_inode':fields[9]})
        except (OSError,ValueError) as exc:
            errors.append('%s: %s'%(family,exc))
    return {'listener_observation':'passive_proc_net',
            'tcp_listening':True if listeners else (False if tables_read==2 else None),
            'listeners':listeners,'listener_observation_errors':errors}


def launch_command(carla_root,port,render_mode='nullrhi',graphics_adapter=None):
    command=[str(carla_root/'CarlaUE4.sh'),'-nosound','-carla-rpc-port=%d'%port]
    if render_mode=='offscreen':
        command+=['-RenderOffScreen','-quality-level=Low']
        if graphics_adapter is not None:command.append('-graphicsadapter=%d'%graphics_adapter)
    elif render_mode=='nullrhi':
        command.append('-nullrhi')
    else:
        raise ValueError('Unknown render mode: '+str(render_mode))
    return command


def prepare_output(out,allow_precreated_empty=False):
    """Permit the wrapper's newly created, user-owned empty directory only."""
    if not allow_precreated_empty:
        out.mkdir(parents=True,exist_ok=False)
        return
    if not out.is_dir() or out.is_symlink():
        raise ValueError('Prepared output must be an existing real directory: '+str(out))
    if out.stat().st_uid!=os.geteuid():
        raise PermissionError('Prepared output must belong to the current audit user: '+str(out))
    if any(out.iterdir()):
        raise FileExistsError('Refuse to reuse nonempty output: '+str(out))


def stop_owned_server(process):
    if process is None or process.poll() is not None:return
    # start_new_session=True makes this group exclusively ours. Never search
    # for or terminate other CARLA servers by executable name or port.
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=15.)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=15.)


def run_logged(command,path,cwd):
    with path.open('w',encoding='utf-8') as log:
        log.write(json.dumps(command)+'\n');log.flush()
        process=subprocess.Popen(command,cwd=str(cwd),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace')
        try:
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:process.wait(timeout=15.)
                except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=15.)
            raise


def main():
    print('Audit Python entered main.',flush=True)
    root=Path(__file__).resolve().parent
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-report',type=Path,required=True)
    p.add_argument('--port',type=int,default=31000)
    p.add_argument('--auto-port',action='store_true',help='Find a free RPC/streaming TCP block for a new owned server')
    p.add_argument('--native-records',type=int,default=20)
    p.add_argument('--pool-records',type=int,default=100)
    p.add_argument('--probe-ticks',type=int,default=20)
    p.add_argument('--startup-timeout',type=float,default=600.)
    p.add_argument('--audit-timeout',type=float,default=300.)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--prepared-output',action='store_true',
                   help='Use the wrapper-created empty output directory owned by the audit user')
    p.add_argument('--startup-only',action='store_true',help='Validate RPC/world/map and stop; do not run scene audits')
    p.add_argument('--mainline',action='store_true',help='Run matched GT/V2/V3 diagnostics with automatic failure analysis')
    p.add_argument('--with-physics',action='store_true',help='Opt into dynamic tick probes (disabled by default)')
    p.add_argument('--resume-batch-from',type=Path,help='Old mainline experiment directory; requires --mainline')
    p.add_argument('--render-mode',choices=('offscreen','nullrhi'),default='nullrhi',
                   help='Default: existing RIFT server no-rendering path (-nullrhi); offscreen is opt-in')
    p.add_argument('--graphics-adapter',type=int,help='Vulkan adapter index, only for offscreen mode')
    p.add_argument('--use-existing-server',action='store_true',help='Only for an already ready, dedicated empty server')
    a=p.parse_args()
    if not 1<=a.port<=65533:p.error('--port must be between 1 and 65533')
    if a.auto_port and a.use_existing_server:p.error('--auto-port requires a new owned server')
    if not a.use_existing_server and set(range(a.port,a.port+3)).intersection(range(38000,38003)) and not a.auto_port:
        p.error('Ports 38000-38002 are reserved for the existing service; choose another --port')
    if a.resume_batch_from and not a.mainline:p.error('--resume-batch-from requires --mainline')
    if not 0<a.native_records<=a.pool_records or a.probe_ticks<=0 or a.startup_timeout<=0:
        p.error('Require 0 < native-records <= pool-records, probe-ticks/startup-timeout > 0')
    if a.graphics_adapter is not None and (a.graphics_adapter<0 or a.render_mode!='offscreen'):
        p.error('--graphics-adapter requires offscreen mode and a nonnegative index')
    if sys.platform!='linux':raise SystemExit('Managed CARLA launch is for the Linux server; use the existing audit CLI elsewhere')
    out=a.output_dir or root/'data/hv_init/eval'/('native_managed_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out=out.absolute()
    prepare_output(out,a.prepared_output)
    print('RESULT_DIR='+str(out),flush=True)
    status={'server_owned':not a.use_existing_server,'port':a.port,'server_command':[],
            'server_cwd':str(root/'RIFT'),
            'pair_report':str(a.pair_report.absolute()),'status':'initializing','experiment_exit_code':None,
            'startup_only':a.startup_only,'audit_started':False,
            'audit_uid':os.geteuid(),'render_mode':a.render_mode}
    (out/'run.json').write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
    def save_status():
        (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    save_status()
    def stage(name):
        status['status']=name
        save_status()
        print('Stage: '+name,flush=True)
    server=None;server_log=None
    try:
        # Persist status BEFORE potentially slow network-filesystem reads.
        # Previously all of these checks ran while the output was still empty.
        stage('checking_pair_report_path')
        if not a.pair_report.is_file():raise FileNotFoundError(a.pair_report)
        from prepare_hv_diagnostic_subset import validate_subset
        stage('reading_pair_report')
        report=json.loads(a.pair_report.read_text(encoding='utf-8'))
        partial=a.pair_report.parent/Path(report['partial_manifest_path']).name
        stage('reading_partial_manifest')
        data=partial.read_bytes()
        stage('validating_partial_manifest')
        validate_subset(report,data)
        stage('checking_server_launch')
        if a.auto_port:
            requested_port=a.port
            a.port=choose_free_port(a.port)
            status.update(requested_port=requested_port,port=a.port)
            (out/'run.json').write_text(json.dumps(vars(a),default=str,indent=2),encoding='utf-8')
            save_status()
            print('Owned CARLA selected RPC port %d (requested %d); existing services untouched.'%
                  (a.port,requested_port),flush=True)
        carla_root=Path(os.environ.get('CARLA_ROOT') or root/'carla').resolve()
        command=launch_command(carla_root,a.port,a.render_mode,a.graphics_adapter)
        if not a.use_existing_server:
            if not Path(command[0]).is_file():raise FileNotFoundError(command[0])
            with socket.socket() as check:
                try:check.bind(('127.0.0.1',a.port))
                except OSError as exc:
                    raise RuntimeError('Port %d is occupied. Use another free port, or --use-existing-server only '
                                       'if you own that dedicated empty server.' % a.port) from exc
            if os.geteuid()==0:
                runuser=shutil.which('runuser')
                if runuser is None:raise RuntimeError('Root requires runuser -u riftuser to start CARLA')
                command=[runuser,'-u','riftuser','--']+command
        status['server_command']=command
        stage('preflight')
        code=run_logged([sys.executable,'-u',str(root/'check_hv_native_environment.py')],out/'preflight.log',root)
        if code:raise RuntimeError('Native dependency preflight failed; inspect preflight.log')
        print('Dependency imports passed. CARLA RPC has NOT been checked yet.',flush=True)
        sys.path.insert(0,str(root/'scenario-dreamer'))
        from high_value_init.carla_audit_environment import configure_carla_pythonapi
        configure_carla_pythonapi(root,validate=True)
        import carla
        if not a.use_existing_server:
            server_log=(out/'server.log').open('w',encoding='utf-8')
            print('Starting owned CARLA:',command,flush=True)
            server=subprocess.Popen(command,cwd=str(root/'RIFT'),stdout=server_log,stderr=subprocess.STDOUT,
                                    start_new_session=True)
            status['server_pid']=server.pid
        status['status']='waiting_for_rpc'
        save_status()
        def progress(info):
            status['startup_progress']=info
            status['startup_evidence']=startup_evidence(a.port,server,out/'server.log')
            save_status()
            print('TCP listening (passive observation):',status['startup_evidence']['tcp_listening'],flush=True)
            if status['startup_evidence'].get('server_log_tail'):
                print('server.log tail:\n'+status['startup_evidence']['server_log_tail'][-3000:],flush=True)
        def client_factory():
            client=carla.Client('127.0.0.1',a.port);client.set_timeout(5.)
            return client
        # Like RIFT/scripts/run.py, create a fresh client on each startup
        # attempt instead of retaining a connection made before RPC is ready.
        status['readiness']=wait_until_ready(None,server,a.startup_timeout if server else 5.,
                                           on_progress=progress,client_factory=client_factory)
        print('CARLA ready:',json.dumps(status['readiness']),flush=True)
        if a.startup_only:
            status['status']='startup_verified'
            save_status()
            print('STARTUP VERIFIED. Scene audits were not run.',flush=True)
            return
        status['status']='running'
        status['audit_started']=True
        save_status()
        if a.mainline:
            command=[sys.executable,'-u',str(root/'run_hv_mainline_batch.py'),
                     '--pair-report',str(a.pair_report),'--pool-records',str(a.pool_records),
                     '--native-records',str(a.native_records),'--probe-ticks',str(a.probe_ticks),
                     '--port',str(a.port),'--timeout',str(a.audit_timeout),'--output-dir',str(out/'experiment')]
            if a.resume_batch_from:command+=['--resume-from',str(a.resume_batch_from)]
            if a.with_physics:command+=['--with-physics']
        else:
            command=[sys.executable,'-u',str(root/'run_hv_snapshot_diagnostics.py'),
                     '--diagnostic-pair-report',str(a.pair_report),'--with-native','--with-spawn',
                     '--native-records',str(a.native_records),'--max-records',str(a.pool_records),
                     '--probe-ticks',str(a.probe_ticks),'--port',str(a.port),'--timeout',str(a.audit_timeout),
                     '--output-dir',str(out/'experiment')]
            if a.with_physics:command+=['--with-physics']
        code=run_logged(command,out/'experiment.log',root)
        status['experiment_exit_code']=code;status['status']='completed' if code==0 else 'experiment_failed'
        if code:raise SystemExit(code)
    except BaseException as exc:
        status['failure_stage']=status['status']
        if status['status']=='running':status['status']='experiment_interrupted' if isinstance(exc,KeyboardInterrupt) else 'experiment_failed'
        elif status['status'] not in ('completed','experiment_failed'):status['status']='failed_before_or_during_startup'
        status['error']='%s: %s' % (type(exc).__name__,exc)
        if 'readiness' not in status:
            status['startup_evidence']=startup_evidence(a.port,server,out/'server.log')
            print('Startup evidence before cleanup:\n'+json.dumps(status['startup_evidence'],indent=2),flush=True)
        save_status()
        raise
    finally:
        stop_owned_server(server)
        if server_log is not None:server_log.close()
        save_status()
        print('Logs and results:',out,flush=True)


if __name__=='__main__':main()
