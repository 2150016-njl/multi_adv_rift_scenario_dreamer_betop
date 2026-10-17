#!/usr/bin/env python3
"""Fail-closed patch for xodr_wp=None dereferences in hv_v8/p1_native_audit.py."""
import datetime
import shutil
import sys
from pathlib import Path

POSITION_OLD = """        check[
            "xodr_exact_position_error_m"
        ] = distance2(
            xodr_wp.transform.location,
            exact_wp.transform.location,
        )
"""
POSITION_NEW = """        if xodr_wp is not None:
            check[
                "xodr_exact_position_error_m"
            ] = distance2(
                xodr_wp.transform.location,
                exact_wp.transform.location,
            )
        else:
            # XODR lookup failed; keep the comparison explicitly unavailable.
            check["xodr_exact_position_error_m"] = None
"""
YAW_OLD = """        check[
            "xodr_exact_yaw_error_deg"
        ] = abs(
            (
                (
                    float(
                        xodr_wp.transform.rotation.yaw
                    )
                    - float(
                        exact_wp.transform.rotation.yaw
                    )
                    + 180.0
                )
                % 360.0
            )
            - 180.0
        )
"""
YAW_NEW = """        if xodr_wp is not None:
            check[
                "xodr_exact_yaw_error_deg"
            ] = abs(
                (
                    (
                        float(
                            xodr_wp.transform.rotation.yaw
                        )
                        - float(
                            exact_wp.transform.rotation.yaw
                        )
                        + 180.0
                    )
                    % 360.0
                )
                - 180.0
            )
        else:
            # XODR lookup failed; keep the comparison explicitly unavailable.
            check["xodr_exact_yaw_error_deg"] = None
"""

def replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise RuntimeError(f"expected exactly one {label} block, found {n}; no file modified")
    return source.replace(old, new, 1)

def main():
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "hv_v8/p1_native_audit.py").resolve()
    if not target.is_file():
        print(f"ERROR: target not found: {target}", file=sys.stderr)
        return 2
    source = target.read_text(encoding="utf-8")
    if ('check["xodr_exact_position_error_m"] = None' in source
            and 'check["xodr_exact_yaw_error_deg"] = None' in source):
        print("P1_XODR_NONE_GUARD_ALREADY_PRESENT")
        print(f"file = {target}")
        return 0
    try:
        result = replace_once(source, POSITION_OLD, POSITION_NEW, "XODR position comparison")
        result = replace_once(result, YAW_OLD, YAW_NEW, "XODR yaw comparison")
        compile(result, str(target), "exec")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_xodr_none_guard_{stamp}")
    shutil.copy2(target, backup)
    target.write_text(result, encoding="utf-8")
    print("P1_XODR_NONE_GUARD_PATCH_APPLIED")
    print(f"file = {target}")
    print(f"backup = {backup}")
    print("xodr_found remains false when get_waypoint_xodr returns None")
    print("XODR-vs-exact comparison fields are null when XODR waypoint is unavailable")
    print("route snapshot and s values are unchanged")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
