from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from .common import ContractError, read_json, read_jsonl, require, write_json


def validate_scene(scene: dict, where: str) -> list[str]:
    errors = []
    def req(cond, msg):
        if not cond: errors.append(msg)
    req(scene.get("schema_version") == "hv_v8_scene_v1", "schema_version must be hv_v8_scene_v1")
    for k in ("record_index","scene_id","town","ego_route","actors"):
        req(k in scene, f"missing {k}")
    ego=scene.get("ego",{})
    for k in ("x","y","z","yaw_deg","speed_mps"):
        req(k in ego, f"ego missing {k}")
        if k in ego:
            req(math.isfinite(float(ego[k])), f"ego.{k} not finite")
    route=scene.get("ego_route",[])
    req(len(route)>=2, "ego_route must have >=2 points")
    for i,r in enumerate(route):
        for k in ("x","y","z","yaw_deg","road_id","section_id","lane_id","s"):
            req(k in r, f"ego_route[{i}] missing {k}")
    ids=set()
    for i,a in enumerate(scene.get("actors",[])):
        aid=a.get("actor_id")
        req(aid not in ids, f"duplicate actor_id={aid}")
        ids.add(aid)
        for k in ("actor_id","type_id","x","y","z","yaw_deg","speed_mps"):
            req(k in a, f"actors[{i}] missing {k}")
    return [f"{where}: {x}" for x in errors]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True,help="JSONL rows with snapshot_path")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    rows=read_jsonl(Path(args.manifest))
    errors=[]
    for i,row in enumerate(rows):
        try:
            scene=read_json(Path(row["snapshot_path"]))
            errors.extend(validate_scene(scene,f"row={i},record={row.get('record_index')}"))
        except Exception as e:
            errors.append(f"row={i}: {type(e).__name__}: {e}")
    report={"schema_version":"hv_v8_scene_schema_audit_v1","rows":len(rows),
            "error_count":len(errors),"errors":errors,
            "pass":not errors}
    write_json(Path(args.output),report)
    print(report)
    raise SystemExit(0 if not errors else 3)


if __name__=="__main__":
    main()
