#!/usr/bin/env python3
"""Build reproducible P1 candidate from exact parent frozen SHA, no in-place edit."""
from __future__ import annotations
import hashlib, ast, json
from pathlib import Path

BASE=Path(__file__).resolve().parent
PARENT=Path(__file__).resolve().parent/'parent_p1_native_audit.py'
EXPECTED='b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb'

def sha(b):return hashlib.sha256(b).hexdigest()

def replace_function(src,name,new):
    tree=ast.parse(src); hits=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef)) and x.name==name]
    assert len(hits)==1,(name,len(hits))
    node=hits[0];lines=src.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.strip('\n')+'\n\n'+''.join(lines[node.end_lineno:])

src=PARENT.read_text()
assert sha(PARENT.read_bytes())==EXPECTED, 'parent SHA mismatch; refuse patch'

src=replace_function(src,'route_waypoints', '''
def route_waypoints(carla: Any, carla_map: Any, scene: dict[str, Any]):
    # Native geometry chosen from independently verified historical OpenDRIVE
    # identity; exact xyz inverse lookup is preserved as a non-unique probe.
    return route_waypoints_v2(carla,carla_map,scene)
''')

src=src.replace('from .p1_world_guard import load_verified_world, assert_same_episode_map',
                'from .p1_world_guard import load_verified_world, assert_same_episode_map\nfrom .p1_science_contract_v2 import route_waypoints_v2')

src=replace_function(src,'is_within_distance_ahead', '''
def is_within_distance_ahead(target_transform: Any, ref_transform: Any,
                             max_distance: float, max_fov_deg: float) -> bool:
    # Exact RIFT utils.misc helper geometry: 3D distance, CARLA vector angle
    # (radians), degrees conversion. Zero vector/NaN never treated as ahead.
    target_loc=target_transform.location
    ref_loc=ref_transform.location
    relative=target_loc-ref_loc
    if target_loc.distance(ref_loc)>max_distance:
        return False
    forward=ref_transform.rotation.get_forward_vector()
    angle_deg=math.degrees(forward.get_vector_angle(relative))
    return angle_deg < max_fov_deg
''')
# Original reference uses //2, not /2.0
assert '''DEFAULT[
                "max_interaction_fov_deg"
            ] / 2.0''' in src
src=src.replace('''DEFAULT[
                "max_interaction_fov_deg"
            ] / 2.0''','''DEFAULT[
                "max_interaction_fov_deg"
            ] // 2''')

src=replace_function(src,'candidate_semantics', '''
def candidate_semantics(carla: Any, carla_map: Any,
                        actor: dict[str, Any]) -> dict[str, Any]:
    tr=carla_transform(carla,actor)
    dist=float(actor['_ego_location'].distance(tr.location))
    wp=preview_waypoint(carla,carla_map,tr)
    out={'distance_m':dist,'distance_pass':bool(math.isfinite(dist) and
          DEFAULT['min_distance_m'] < dist <= DEFAULT['search_radius_m']),
         'preview_exists':wp is not None,'junction_heading_pass':True,
         'heading_numeric_status':'not_required'}
    if wp is not None and wp.is_junction:
        a=tr.get_forward_vector();b=wp.transform.get_forward_vector()
        rad=float(a.get_vector_angle(b))
        out['junction_heading_angle_rad']=rad if math.isfinite(rad) else None
        out['junction_forward_vector']=[float(a.x),float(a.y),float(a.z)]
        out['road_forward_vector']=[float(b.x),float(b.y),float(b.z)]
        if not math.isfinite(rad):
            out['heading_numeric_status']='unknown_nonfinite'
            out['junction_heading_error_deg']=None
            out['junction_heading_pass']=None
        else:
            deg=math.degrees(rad)
            out['junction_heading_error_deg']=deg
            out['heading_numeric_status']='finite'
            # RIFT source uses "if angle > 20: continue". Equal 20 passes.
            out['junction_heading_pass']=bool(deg<=20.0)
    if out['junction_heading_pass'] is None:
        out['candidate_filter_pass']=None
        out['candidate_filter_status']='unknown_numeric'
    else:
        out['candidate_filter_pass']=bool(out['distance_pass'] and
                                          out['preview_exists'] and out['junction_heading_pass'])
        out['candidate_filter_status']='evaluated'
    return out
''')

src=replace_function(src,'native_truth_status', '''
def native_truth_status(candidate: dict[str, Any], interaction: dict[str, Any]) -> tuple[str,bool|None]:
    if interaction.get('status')!='complete' or candidate.get('candidate_filter_pass') is None:
        return 'unknown',None
    passed=bool(candidate['candidate_filter_pass'] and interaction.get('native_interaction') is True)
    return ('pass' if passed else 'fail'),passed
''')
# Extend raw pose evidence to full 3-D and roll/pitch. Preserve existing tolerances.
old='''position_error = math.hypot(
                actual_pose["x"] - expected["x"],
                actual_pose["y"] - expected["y"],
            )'''
new='''position_error = math.sqrt(sum((actual_pose[a]-expected[a])**2 for a in ('x','y','z')))
            pitch_error = abs((actual_pose['pitch']-expected['pitch']+180.0)%360.0-180.0)
            roll_error = abs((actual_pose['roll']-expected['roll']+180.0)%360.0-180.0)'''
assert old in src
src=src.replace(old,new)
assert 'elif position_error <= 0.30 and yaw_error <= 2.0:' in src
src=src.replace('elif position_error <= 0.30 and yaw_error <= 2.0:',
                "elif position_error <= 0.30 and yaw_error <= 2.0 and pitch_error <= 2.0 and roll_error <= 2.0:")
# Make post-spawn angular evidence explicit.
mark='''"yaw_error_deg": yaw_error,
            "position_tolerance_m": 0.30,'''
assert mark in src
src=src.replace(mark,'''"yaw_error_deg": yaw_error,
            "pitch_error_deg": (pitch_error if actual_pose is not None else None),
            "roll_error_deg": (roll_error if actual_pose is not None else None),
            "full_xyz_pose_validation": True,
            "position_tolerance_m": 0.30,''')
# Ensure not using unbound pitch/roll for cases actual none: guarded ternary yes.
# Strict model3 full-scene independent experiment is a separate process, not
# incorrectly inferred from raw spawning in this native audit.

out=BASE/'hv_v8/p1_native_audit.py'
out.write_text(src,encoding='utf-8')
print('P1_PARENT_SHA256='+EXPECTED)
print('P1_CANDIDATE_SHA256='+sha(out.read_bytes()))
print('P1_CANDIDATE_MODULE='+str(out))
