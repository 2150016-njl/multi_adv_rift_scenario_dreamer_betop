#!/usr/bin/env python3
"""Import native dependencies without contacting CARLA or loading model weights."""
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scenario-dreamer'))
sys.path.insert(0,str(ROOT/'RIFT'))
from high_value_init.carla_audit_environment import configure_carla_pythonapi


def main():
    result=configure_carla_pythonapi(ROOT,validate=True)
    for name in ('rift.cbv.recognition.base_cbv','rift.cbv.planning.route_planner.base_planner',
                 'rift.scenario.tools.global_route_planner'):
        result[name+'_source']=importlib.import_module(name).__file__
    result['server_contacted']=False
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
