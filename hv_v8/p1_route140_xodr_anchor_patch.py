#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guarded, immutable P1 Route140 XODR-anchor contract revision.

IMPORTANT: does not contact CARLA, edit historical snapshots/manifests, or
modify the live hv_v8/p1_native_audit.py. Only creates a NEW frozen capsule
with --create. On any uncertainty, fail closed before writing.

The patch retains exact-location s errors as measured diagnostics, but uses
XODR (road/lane/original-s) as the strict source-state anchor. For verified
road-end XODR misses it retains the original exact-only endpoint exception.
No thresholds are relaxed from 0.05 to 0.10, and no GT states are moved.
"""
import argparse
import ast
import difflib
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

OLD_SHA = '34a80666b1030434c528d0b91f7bb7f2994eb65bba21b5b4c8540fa972898d2c'
EXPECTED_MANIFEST_SHA = '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
EXPECTED_MAP_SHA = '9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf'

# Exact original predicate from the historical V3 frozen-source integrator.
OLD_BLOCK = '        # Separate core route identity checks from the XODR cross-check.\n        # A verified road-end exception does not make the XODR lookup succeed.\n        route_identity_pass = (\n            int(wp.road_id)\n            == int(r["road_id"])\n            and int(wp.section_id)\n            == int(r["section_id"])\n            and int(wp.lane_id)\n            == int(r["lane_id"])\n            and check["s_abs_error_m"] <= 0.05\n            and check["position_error_m"] <= 0.30\n            and check["yaw_error_deg"] <= 2.0\n        )\n\n        check["route_identity_pass"] = bool(route_identity_pass)\n\n        if xodr_wp is not None:\n            xodr_position_error = check["xodr_exact_position_error_m"]\n            xodr_yaw_error = check["xodr_exact_yaw_error_deg"]\n            xodr_comparison_pass = (\n                xodr_position_error is not None\n                and xodr_position_error <= 0.30\n                and xodr_yaw_error is not None\n                and xodr_yaw_error <= 2.0\n            )\n            check["xodr_exact_comparison_status"] = (\n                "pass" if xodr_comparison_pass else "fail"\n            )\n        else:\n            # The earlier guard must have verified the road-end exception.\n            # Never compare None to a number or report XODR as found.\n            xodr_comparison_pass = bool(xodr_boundary_verified)\n            check["xodr_exact_comparison_status"] = (\n                "not_evaluable_verified_endpoint_exception"\n                if xodr_boundary_verified\n                else "unknown"\n            )\n\n        check["pass"] = bool(\n            route_identity_pass and xodr_comparison_pass\n        )\n'

NEW_BLOCK = '''        # P1_ROUTE140_XODR_SOURCE_ANCHOR_V1
        # Preserve the independent exact-location probe, including its measured
        # s deviation. A CARLA location lookup is not the source waypoint.
        check["exact_location_s_abs_error_m"] = check["s_abs_error_m"]
        check["exact_location_xyz_error_m"] = float(
            exact_wp.transform.location.distance(expected.location)
        )
        check["exact_location_s_over_005"] = bool(
            check["s_abs_error_m"] > 0.05
        )

        if xodr_wp is not None:
            check["xodr_source_road_equal"] = (
                int(xodr_wp.road_id) == int(r["road_id"])
            )
            check["xodr_source_section_equal"] = (
                int(xodr_wp.section_id) == int(r["section_id"])
            )
            check["xodr_source_lane_equal"] = (
                int(xodr_wp.lane_id) == int(r["lane_id"])
            )
            check["xodr_source_s_abs_error_m"] = abs(
                float(xodr_wp.s) - float(r["s"])
            )
            check["xodr_source_xyz_error_m"] = float(
                xodr_wp.transform.location.distance(expected.location)
            )
            check["xodr_source_yaw_error_deg"] = abs(
                (float(xodr_wp.transform.rotation.yaw)
                 - float(r["yaw_deg"]) + 180.0) % 360.0 - 180.0
            )
            xodr_source_anchor_pass = bool(
                check["xodr_source_road_equal"]
                and check["xodr_source_section_equal"]
                and check["xodr_source_lane_equal"]
                and check["xodr_source_s_abs_error_m"] <= 1e-4
                and check["xodr_source_xyz_error_m"] <= 0.05
                and check["xodr_source_yaw_error_deg"] <= 0.5
            )
        else:
            # None remains None; only the earlier verified endpoint may pass.
            check["xodr_source_road_equal"] = None
            check["xodr_source_section_equal"] = None
            check["xodr_source_lane_equal"] = None
            check["xodr_source_s_abs_error_m"] = None
            check["xodr_source_xyz_error_m"] = None
            check["xodr_source_yaw_error_deg"] = None
            xodr_source_anchor_pass = False

        check["xodr_source_anchor_pass"] = bool(xodr_source_anchor_pass)
        check["route_reference_kind"] = (
            "xodr_original_s" if xodr_wp is not None
            else "verified_road_end_exact"
        )
        # Exact-location query is a lane-membership/cross-check only; its s
        # is not substituted for the serialized historical route's s.
        exact_location_membership_pass = bool(
            int(exact_wp.road_id) == int(r["road_id"])
            and int(exact_wp.section_id) == int(r["section_id"])
            and int(exact_wp.lane_id) == int(r["lane_id"])
            and check["position_error_m"] <= 0.30
            and check["exact_location_xyz_error_m"] <= 0.30
            and check["yaw_error_deg"] <= 2.0
        )
        check["exact_location_membership_pass"] = exact_location_membership_pass
        route_identity_pass = bool(
            exact_location_membership_pass
            and (
                xodr_source_anchor_pass if xodr_wp is not None
                else xodr_boundary_verified
            )
        )

        check["route_identity_pass"] = route_identity_pass

        if xodr_wp is not None:
            xodr_position_error = check["xodr_exact_position_error_m"]
            xodr_yaw_error = check["xodr_exact_yaw_error_deg"]
            xodr_comparison_pass = (
                xodr_position_error is not None
                and xodr_position_error <= 0.30
                and xodr_yaw_error is not None
                and xodr_yaw_error <= 2.0
            )
            check["xodr_exact_comparison_status"] = (
                "pass" if xodr_comparison_pass else "fail"
            )
        else:
            xodr_comparison_pass = bool(xodr_boundary_verified)
            check["xodr_exact_comparison_status"] = (
                "not_evaluable_verified_endpoint_exception"
                if xodr_boundary_verified else "unknown"
            )

        check["pass"] = bool(route_identity_pass and xodr_comparison_pass)
'''

SOURCE_REL = Path('hv_v8/p1_native_audit.py')
OLD_APPEND = '        out.append(wp)\n'
NEW_APPEND = '''        # Preserve the original serialized route geometry in native replay.
        # No snap, projection, GT adjustment or candidate substitution.
        out.append(xodr_wp if xodr_wp is not None else exact_wp)
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def patch(text):
    require('P1_ROUTE140_XODR_SOURCE_ANCHOR_V1' not in text,
            'refusing to reapply an already-present patch')
    require(text.count(OLD_BLOCK) == 1,
            'frozen route identity block differs from verified V3 version')
    start = text.index('def route_waypoints(')
    end = text.find('\ndef cumulative_route(', start)
    require(end > start, 'route_waypoints function boundary not found')
    route_code = text[start:end]
    require(route_code.count(OLD_APPEND) == 1,
            'route output append differs from verified V3 source')
    require(route_code.count(OLD_BLOCK) == 1,
            'route identity predicate not within route_waypoints')
    require('xodr_endpoint_exception_verified' in route_code and
            'xodr_boundary_verified' in route_code and
            'exact_wp = carla_map.get_waypoint(' in route_code and
            'xodr_wp = carla_map.get_waypoint_xodr(' in route_code,
            'frozen source lacks verified endpoint / XODR / exact contracts')
    new_route = route_code.replace(OLD_BLOCK, NEW_BLOCK).replace(OLD_APPEND, NEW_APPEND)
    result = text[:start] + new_route + text[end:]
    compile(result, 'new_frozen/hv_v8/p1_native_audit.py', 'exec')
    fn = next(n for n in ast.walk(ast.parse(result))
              if isinstance(n, ast.FunctionDef) and n.name == 'route_waypoints')
    require('P1_ROUTE140_XODR_SOURCE_ANCHOR_V1' in ast.get_source_segment(result, fn),
            'replaced code is not in route_waypoints')
    return result


def validate_original(root, capsule):
    original = capsule / SOURCE_REL
    require(original.is_file(), 'frozen module missing: {}'.format(original))
    raw = original.read_bytes()
    require(sha(raw) == OLD_SHA, 'original capsule SHA mismatch; refuse patch: {}'.format(sha(raw)))
    base = root / 'data/hv_init/v7/manifests'
    for p, expected in [
        (base / 'p1_calibration95.jsonl', EXPECTED_MANIFEST_SHA),
        (base / 'p1_map_identity_manifest_v2_20261017_154737.json', EXPECTED_MAP_SHA),
    ]:
        require(p.is_file(), 'input missing: {}'.format(p))
        require(sha(p.read_bytes()) == expected, 'input changed: {}'.format(p))
    return raw.decode('utf-8')


def run(root, make):
    original_capsule = root / 'data/hv_init/v7/p1_frozen_capsule_20261018_033841_9535'
    original_text = validate_original(root, original_capsule)
    updated = patch(original_text)
    updated_bytes = updated.encode('utf-8')
    new_sha = sha(updated_bytes)
    difference = ''.join(difflib.unified_diff(
        original_text.splitlines(True), updated.splitlines(True),
        fromfile='frozen-original/hv_v8/p1_native_audit.py',
        tofile='new-frozen/hv_v8/p1_native_audit.py'))
    require(difference and 'xodr_source_anchor_pass' in difference,
            'unexpected empty patch')
    print('P1_ORIGINAL_FROZEN_SHA256={}'.format(OLD_SHA))
    print('P1_ROUTE140_CANDIDATE_FROZEN_SHA256={}'.format(new_sha))
    print('P1_ROUTE140_CHANGED_LINES={}'.format(len(difference.splitlines())))
    print('P1_ROUTE140_OFFLINE_PATCH_CONTRACT_PASS (NOT CARLA PASS)')
    if not make:
        return
    new_capsule = root / 'data/hv_init/v7/p1_frozen_capsule_route140_{}_{}'.format(
        datetime.now().strftime('%Y%m%d_%H%M%S'), os.getpid())
    require(not new_capsule.exists(), 'new capsule already exists: {}'.format(new_capsule))
    shutil.copytree(str(original_capsule), str(new_capsule))
    new_src = new_capsule / SOURCE_REL
    try:
        require(sha(new_src.read_bytes()) == OLD_SHA,
                'copied original capsule differs from expected SHA')
        new_src.write_bytes(updated_bytes)
        require(sha(new_src.read_bytes()) == new_sha,
                'new capsule SHA check failed')
        (new_capsule / 'route140_xodr_anchor.diff').write_text(difference, encoding='utf-8')
        # The cloned original provenance must NOT masquerade as the new source.
        meta_path = new_capsule / 'p1_source_provenance.json'
        require(meta_path.is_file(), 'original capsule provenance missing')
        meta_original = json.loads(meta_path.read_text(encoding='utf-8'))
        require(meta_original.get('frozen_sha256') == OLD_SHA,
                'original provenance has unexpected frozen SHA')
        (new_capsule / 'p1_parent_source_provenance.json').write_text(
            json.dumps(meta_original, indent=2, ensure_ascii=False), encoding='utf-8')
        meta_original['parent_frozen_sha256'] = OLD_SHA
        meta_original['frozen_sha256'] = new_sha
        meta_original['reconciliation_steps'] = list(meta_original.get('reconciliation_steps') or []) + [
            'P1_ROUTE140_XODR_SOURCE_ANCHOR_V1_on_new_frozen_capsule_only']
        meta_original.setdefault('contracts', {})['xodr_source_anchor_contract_ok'] = True
        meta_original['previous_smoke_result_directory'] = meta_original.get('result_directory')
        meta_original['result_directory'] = None  # no scientific run has yet occurred
        meta_original['diff'] = str(new_capsule / 'route140_xodr_anchor.diff')
        meta_path.write_text(json.dumps(meta_original, indent=2, ensure_ascii=False), encoding='utf-8')
        (new_capsule / 'route140_xodr_anchor_provenance.json').write_text(json.dumps({
            'status': 'CANDIDATE_ONLY_NOT_CARLA_VALIDATED',
            'old_capsule': str(original_capsule),
            'old_frozen_sha256': OLD_SHA,
            'new_frozen_sha256': new_sha,
            'manifest_sha256': EXPECTED_MANIFEST_SHA,
            'map_manifest_sha256': EXPECTED_MAP_SHA,
            'source_data_modified': False,
            'threshold_005_to_010': False,
            'new_route_reference': 'xodr-original-s when available; else verified road-end exact',
        }, indent=2, ensure_ascii=False), encoding='utf-8')
    except Exception:
        # Preserve failed staging for investigation; never erase evidence.
        raise
    require(sha(original_capsule.joinpath(SOURCE_REL).read_bytes()) == OLD_SHA,
            'original capsule unexpectedly modified')
    print('P1_ROUTE140_NEW_CAPSULE={}'.format(new_capsule))
    print('P1_ROUTE140_DIFF={}'.format(new_capsule / 'route140_xodr_anchor.diff'))
    print('P1_ROUTE140_INPUTS_AND_ORIGINAL_UNCHANGED_PASS')
    print('P1_ROUTE140_NEW_CAPSULE_CANDIDATE_ONLY_NOT_CARLA_PASS')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--create', action='store_true', help='write new capsule, never overwrite old')
    args = p.parse_args()
    try:
        run(Path(args.root).resolve(), args.create)
    except Exception as exc:
        print('P1_ROUTE140_PATCH_FAIL_CLOSED={}: {}'.format(type(exc).__name__, exc),
              file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
