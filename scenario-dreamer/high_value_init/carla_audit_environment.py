"""Configure CARLA PythonAPI before importing native RIFT dependencies."""
import importlib
import importlib.util
import os
from pathlib import Path
import sys


def configure_carla_pythonapi(project_root=None,validate=False):
    root=Path(project_root or Path(__file__).resolve().parents[2])
    carla_root=Path(os.environ.get('CARLA_ROOT') or root/'carla').expanduser()
    api=carla_root/'PythonAPI'
    added=[]
    for path in reversed([api/'carla',api]):
        if path.is_dir():
            value=str(path.resolve())
            if value not in sys.path:sys.path.insert(0,value);added.append(value)
    # Keep an already installed CARLA binary. The agents sources are separate
    # from that binary distribution, and still require PythonAPI/carla.
    if importlib.util.find_spec('carla') is None:
        explicit=os.environ.get('CARLA_EGG')
        tag='-py%d.%d-' % sys.version_info[:2]
        platform='win-amd64' if sys.platform=='win32' else 'linux-x86_64'
        candidates=([Path(explicit)] if explicit else
                    sorted((api/'carla/dist').glob('carla-*%s%s.egg' % (tag,platform))))
        if len(candidates)==1 and candidates[0].exists():
            value=str(candidates[0].resolve())
            if value not in sys.path:sys.path.append(value);added.append(value)
    importlib.invalidate_caches()
    result={'carla_root':str(carla_root),'pythonapi_directory':str(api),'added_paths':added}
    if validate:
        for name in ('carla','agents.navigation.local_planner','agents.tools.misc'):
            try:
                module=importlib.import_module(name)
                result[name+'_source']=getattr(module,'__file__',None)
            except ImportError as exc:
                raise RuntimeError('Native audit dependency %s cannot import. CARLA_ROOT=%s; '
                    'expected agents/navigation/local_planner.py under %s. Install/use the matching CARLA '
                    'PythonAPI sources; the carla binary alone does not supply agents. Original error: %s'
                    % (name,carla_root,api/'carla',exc)) from exc
    return result
