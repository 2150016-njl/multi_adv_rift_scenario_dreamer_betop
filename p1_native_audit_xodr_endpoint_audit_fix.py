#!/usr/bin/env python3
"""Patch P1 route auditing for a verified CARLA OpenDRIVE road-end lookup miss.

This does NOT alter scenario-generation output or mutate route s values.
It permits continuation only when:
  * get_waypoint_xodr returned None,
  * route s is at the runtime OpenDRIVE road endpoint,
  * exact serialized-location lookup returns the already-validated same
    road/section/lane waypoint,
  * exact waypoint s, position, and yaw agree with the serialized route point.

The exception is explicitly recorded as xodr_found=False and
xodr_endpoint_exception_verified=True. All other XODR misses remain fatal.
"""
from __future__ import annotations

import datetime
import shutil
import sys
from pathlib import Path

TARGET_DEFAULT = "hv_v8/p1_native_audit.py"

OLD_XODR_BLOCK = """        xodr_wp = carla_map.get_waypoint_xodr(
            int(r["road_id"]),
            int(r["lane_id"]),
            float(r["s"]),
        )

        require(
            xodr_wp is not None,
            "ego_route[%d] "
            "get_waypoint_xodr returned None"
            % i,
        )

        expected = carla.Transform(
"""

NEW_XODR_BLOCK = """        xodr_wp = carla_map.get_waypoint_xodr(
            int(r["road_id"]),
            int(r["lane_id"]),
            float(r["s"]),
        )

        # Construct the serialized transform before classifying an XODR miss.
        # Do not modify r["s"] or substitute a nearby waypoint.
        expected = carla.Transform(
"""

IDENTITY_ANCHOR = """        require(
            (
                int(exact_wp.road_id)
                == int(r["road_id"])
                and int(exact_wp.section_id)
                == int(r["section_id"])
                and int(exact_wp.lane_id)
                == int(r["lane_id"])
            ),
            (
                f"ego_route[{i}] exact lane "
                "identity mismatch"
            ),
        )

        wp = exact_wp
"""

IDENTITY_REPLACEMENT = """        require(
            (
                int(exact_wp.road_id)
                == int(r["road_id"])
                and int(exact_wp.section_id)
                == int(r["section_id"])
                and int(exact_wp.lane_id)
                == int(r["lane_id"])
            ),
            (
                f"ego_route[{i}] exact lane "
                "identity mismatch"
            ),
        )

        # Classify only the proven road-end boundary case. The map lookup
        # remains visibly failed in the audit output; this is not an XODR pass.
        xodr_road_length_m = _p1_road_lengths.get(int(r["road_id"]))
        xodr_boundary_verified = False
        xodr_endpoint_delta_m = None
        exact_s_error_m = abs(float(exact_wp.s) - float(r["s"]))
        exact_position_error_m = exact_wp.transform.location.distance(
            expected.location
        )
        exact_yaw_error_deg = abs((
            float(exact_wp.transform.rotation.yaw)
            - float(expected.rotation.yaw) + 180.0
        ) % 360.0 - 180.0)

        if xodr_wp is None:
            if xodr_road_length_m is not None:
                xodr_endpoint_delta_m = (
                    float(r["s"]) - float(xodr_road_length_m)
                )

            require(
                xodr_road_length_m is not None
                and abs(xodr_endpoint_delta_m) <= 1e-4
                and exact_s_error_m <= 1e-4
                and exact_position_error_m <= 0.05
                and exact_yaw_error_deg <= 0.5,
                (
                    f"ego_route[{i}] get_waypoint_xodr returned None; "
                    "not a verified road-end boundary: "
                    f"road_length={xodr_road_length_m}, "
                    f"s_minus_road_end={xodr_endpoint_delta_m}, "
                    f"exact_s_error={exact_s_error_m}, "
                    f"exact_position_error_m={exact_position_error_m}, "
                    f"exact_yaw_error_deg={exact_yaw_error_deg}"
                ),
            )
            xodr_boundary_verified = True

        wp = exact_wp
"""

ROWS_ANCHOR = """    rows = scene.get("ego_route", [])

    require(
"""
ROWS_REPLACEMENT = """    rows = scene.get("ego_route", [])

    # Read road lengths from the same runtime OpenDRIVE used by CARLA. This
    # is used only to classify an exact road-end XODR lookup miss.
    import xml.etree.ElementTree as _p1_ET

    _p1_road_lengths = {}
    _p1_opendrive_error = None
    try:
        _p1_root = _p1_ET.fromstring(carla_map.to_opendrive())
        for _p1_road in _p1_root.findall("road"):
            _p1_road_lengths[int(_p1_road.attrib["id"])] = float(
                _p1_road.attrib["length"]
            )
    except Exception as _p1_exc:
        _p1_opendrive_error = repr(_p1_exc)

    require(
"""

CHECK_ANCHOR = """        check["route_index"] = i

        check["expected_road_id"] = int(
"""
CHECK_REPLACEMENT = """        check["route_index"] = i

        check["xodr_found"] = xodr_wp is not None
        check["xodr_lookup_status"] = (
            "ok" if xodr_wp is not None
            else "verified_road_end_boundary"
        )
        check["xodr_endpoint_exception_verified"] = xodr_boundary_verified
        check["xodr_road_length_m"] = xodr_road_length_m
        check["xodr_s_minus_road_end_m"] = xodr_endpoint_delta_m
        check["xodr_exact_s_error_m"] = exact_s_error_m
        check["xodr_exact_position_error_m"] = exact_position_error_m
        check["xodr_exact_yaw_error_deg"] = exact_yaw_error_deg
        check["xodr_opendrive_parse_error"] = _p1_opendrive_error

        check["expected_road_id"] = int(
"""


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one {label} anchor, found {count}; no source changes written"
        )
    return source.replace(old, new, 1)


def main() -> int:
    target = Path(sys.argv[1] if len(sys.argv) > 1 else TARGET_DEFAULT).resolve()
    if not target.is_file():
        print(f"ERROR: target file not found: {target}", file=sys.stderr)
        return 2

    source = target.read_text(encoding="utf-8")
    if "xodr_endpoint_exception_verified" in source:
        print("P1_XODR_ENDPOINT_AUDIT_PATCH_ALREADY_PRESENT")
        print(f"file = {target}")
        return 0

    try:
        patched = source
        patched = replace_once(
            patched, ROWS_ANCHOR, ROWS_REPLACEMENT, "route rows/OpenDRIVE"
        )
        patched = replace_once(
            patched, OLD_XODR_BLOCK, NEW_XODR_BLOCK, "XODR immediate require"
        )
        patched = replace_once(
            patched, IDENTITY_ANCHOR, IDENTITY_REPLACEMENT, "exact identity check"
        )
        patched = replace_once(
            patched, CHECK_ANCHOR, CHECK_REPLACEMENT, "route check annotation"
        )
        compile(patched, str(target), "exec")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_xodr_endpoint_{stamp}")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print("P1_XODR_ENDPOINT_AUDIT_PATCH_APPLIED")
    print(f"file = {target}")
    print(f"backup = {backup}")
    print("route s values unchanged")
    print("XODR misses are recorded explicitly; only verified road-end misses may continue")
    print("non-endpoint XODR misses remain fatal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
