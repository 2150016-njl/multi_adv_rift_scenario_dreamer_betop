#!/usr/bin/env python3
"""Collect existing CARLA startup evidence. Never launch CARLA or change settings."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ENV_KEYS = ('HOME', 'USER', 'LOGNAME', 'DISPLAY', 'XDG_RUNTIME_DIR',
            'SDL_VIDEODRIVER', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
            'NVIDIA_DRIVER_CAPABILITIES', 'VK_ICD_FILENAMES', 'VK_DRIVER_FILES',
            'LD_LIBRARY_PATH')


def command_result(command, timeout=12):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        return {'command':command, 'returncode':result.returncode,
                'stdout':result.stdout[-32000:], 'stderr':result.stderr[-16000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command':command, 'error':str(exc)}


def tail_file(path, limit=24000):
    try:
        stat = path.stat()
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size-limit))
            tail = stream.read(limit).decode('utf-8', errors='replace')
        return {'path':str(path), 'size':stat.st_size, 'mtime_epoch':stat.st_mtime,
                'mtime_utc':datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat(),
                'tail':tail}
    except OSError as exc:
        return {'path':str(path), 'error':str(exc)}


def existing_logs(roots, max_files=12):
    """Bound traversal to known Saved/Logs and Saved/Crashes directories."""
    files = {}; errors = []
    def on_error(exc):
        errors.append(str(exc))
    for root in roots:
        try:
            if not root.is_dir():
                continue
            for directory, dirs, names in os.walk(str(root), followlinks=False, onerror=on_error):
                if len(Path(directory).relative_to(root).parts) >= 3:
                    dirs[:] = []
                for name in names:
                    path = Path(directory)/name
                    if path.suffix.lower() not in ('.log', '.txt', '.xml'):
                        continue
                    try:
                        files[str(path)] = (path.stat().st_mtime, path)
                    except OSError as exc:
                        errors.append(str(exc))
        except OSError as exc:
            errors.append(str(exc))
    ordered = sorted(files.values(), key=lambda item:item[0], reverse=True)
    return {'searched_directories':[str(p) for p in roots], 'errors':errors,
            'note':'Logs may belong to earlier runs; compare mtime and command line with the failed run.',
            'files':[tail_file(p) for _, p in ordered[:max_files]]}


def user_probe_code():
    # Executed under the same user used by the managed CARLA launcher. No writes.
    return '''import os,json,pwd,glob,stat
from pathlib import Path
root=Path(__import__('sys').argv[1])
keys=__import__('sys').argv[2:]
paths=[root,root/'CarlaUE4.sh',root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping',
       root/'CarlaUE4/Saved',root/'CarlaUE4/Saved/Logs',Path.home(),Path.home()/'.config',Path('/tmp')]
paths += [Path(p) for pattern in ('/dev/nvidia*','/dev/dri/*') for p in glob.glob(pattern)]
def probe(p):
    result={'path':str(p),'exists':p.exists(),'readable':os.access(str(p),os.R_OK),
            'writable':os.access(str(p),os.W_OK),'executable_or_traversable':os.access(str(p),os.X_OK)}
    try:
        s=p.stat();result.update(uid=s.st_uid,gid=s.st_gid,mode=oct(stat.S_IMODE(s.st_mode)))
    except OSError as e:result['error']=str(e)
    return result
print(json.dumps({'uid':os.getuid(),'gid':os.getgid(),'groups':os.getgroups(),
    'passwd_home':pwd.getpwuid(os.getuid()).pw_dir,'environment':{k:os.environ.get(k) for k in keys},
    'paths':[probe(p) for p in paths]},indent=2))
'''


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--carla-root', type=Path, default=Path(os.environ.get('CARLA_ROOT') or root/'carla'))
    parser.add_argument('--server-user', default='riftuser')
    parser.add_argument('--output-json', type=Path)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run this collector on the Linux CARLA server')
    carla_root = args.carla_root.resolve()
    if not args.run_dir.is_dir():
        parser.error('Run directory does not exist: '+str(args.run_dir))
    output = args.output_json or args.run_dir/('startup_environment_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.json')
    report = {'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'carla_root':str(carla_root), 'run_dir':str(args.run_dir.resolve()),
              'read_only_checks':True, 'carla_launched':False,
              'current_user':command_result(['id']),
              'current_environment':{k:os.environ.get(k) for k in ENV_KEYS},
              'run_files':[tail_file(args.run_dir/name) for name in ('run.json','status.json','server.log')],
              'launcher':tail_file(carla_root/'CarlaUE4.sh')}
    prefix = []
    if os.geteuid() == 0:
        runuser = shutil.which('runuser')
        if not runuser:
            parser.error('runuser is missing; cannot reproduce the CARLA server user')
        prefix = [runuser, '-u', args.server_user, '--']
    else:
        import pwd
        actual = pwd.getpwuid(os.getuid()).pw_name
        if actual != args.server_user:
            parser.error('Run as root or as --server-user; actual user is '+actual)
    python = '/usr/bin/python3' if Path('/usr/bin/python3').is_file() else sys.executable
    probe = command_result(prefix+[python, '-c', user_probe_code(), str(carla_root)]+list(ENV_KEYS))
    report['server_user_probe'] = probe
    home = None
    if probe.get('returncode') == 0:
        try:
            report['server_user'] = json.loads(probe['stdout'])
            home = Path(report['server_user']['passwd_home'])
        except (ValueError, KeyError) as exc:
            report['server_user_parse_error'] = str(exc)
    report['gpu_and_libraries'] = {}
    binary = carla_root/'CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping'
    for name, command in (
            ('nvidia_smi_current', ['nvidia-smi']),
            ('nvidia_smi_server_user', prefix+['nvidia-smi']),
            ('vulkan_server_user', prefix+['vulkaninfo','--summary']),
            ('binary_libraries_server_user', prefix+['ldd',str(binary)])):
        print('Collecting '+name+' (bounded to 12 seconds)...', flush=True)
        report['gpu_and_libraries'][name] = command_result(command)
    saved_roots = [carla_root/'CarlaUE4/Saved', carla_root/'Saved']
    if home:
        saved_roots += [home/'.config/Epic/CarlaUE4/Saved', home/'.local/share/CarlaUE4/Saved']
    log_roots = [p/name for p in saved_roots for name in ('Logs','Crashes')]
    report['engine_logs'] = existing_logs(log_roots)
    report['vulkan_icd'] = []
    for directory in (Path('/usr/share/vulkan/icd.d'), Path('/etc/vulkan/icd.d')):
        try:
            report['vulkan_icd'].extend(tail_file(p,4000) for p in directory.glob('*.json'))
        except OSError as exc:
            report['vulkan_icd'].append({'path':str(directory),'error':str(exc)})
    report['process_limits'] = tail_file(Path('/proc/self/limits'),8000)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
    print('REPORT='+str(output.resolve()), flush=True)
    print('Existing engine log files collected: '+str(len(report['engine_logs']['files'])))
    print('CARLA was not launched. Upload the report JSON for analysis.')


if __name__ == '__main__':
    main()
