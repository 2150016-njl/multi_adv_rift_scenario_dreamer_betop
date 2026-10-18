#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Narrow P1 route pass-contract patch for verified XODR road-end exception."""
from __future__ import annotations
import datetime
import shutil
import sys
from pathlib import Path

OLD = '''        check["pass"] = (
            int(wp.road_id)
            == int(r["road_id"])
            and int(wp.section_id)
            == int(r["section_id"])
            and int(wp.lane_id)
            == int(r["lane_id"])
            and check["s_abs_error_m"] <= 0.05
            and check["position_error_m"] <= 0.30
            and check["yaw_error_deg"] <= 2.0
            and check[
                "xodr_exact_position_error_m"
            ] <= 0.30
            and check[
                "xodr_exact_yaw_error_deg"
            ] <= 2.0
        )
'''

NEW = '''        # Separate core route identity checks from the XODR cross-check.
        # A verified road-end exception does not make the XODR lookup succeed.
        route_identity_pass = (
            int(wp.road_id)
            == int(r["road_id"])
            and int(wp.section_id)
            == int(r["section_id"])
            and int(wp.lane_id)
            == int(r["lane_id"])
            and check["s_abs_error_m"] <= 0.05
            and check["position_error_m"] <= 0.30
            and check["yaw_error_deg"] <= 2.0
        )

        check["route_identity_pass"] = bool(route_identity_pass)

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
            # The earlier guard must have verified the road-end exception.
            # Never compare None to a number or report XODR as found.
            xodr_comparison_pass = bool(xodr_boundary_verified)
            check["xodr_exact_comparison_status"] = (
                "not_evaluable_verified_endpoint_exception"
                if xodr_boundary_verified
                else "unknown"
            )

        check["pass"] = bool(
            route_identity_pass and xodr_comparison_pass
        )
'''

def main() -> int:
    target = Path(
        sys.argv[1] if len(sys.argv) > 1 else "hv_v8/p1_native_audit.py"
    ).resolve()
    if not target.is_file():
        print(f"ERROR: target file not found: {target}", file=sys.stderr)
        return 2

    source = target.read_text(encoding="utf-8")
    if ('check["xodr_exact_comparison_status"]' in source
            and 'route_identity_pass = (' in source):
        print("P1_ROUTE_PASS_CONTRACT_PATCH_ALREADY_PRESENT")
        print(f"file = {target}")
        return 0

    count = source.count(OLD)
    if count != 1:
        print(
            f"ERROR: expected exactly one old route pass predicate; found {count}. "
            "No changes made.",
            file=sys.stderr,
        )
        return 2

    patched = source.replace(OLD, NEW, 1)
    try:
        compile(patched, str(target), "exec")
    except Exception as exc:
        print(f"ERROR: patched source failed compile check: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_route_pass_contract_{stamp}")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print("P1_ROUTE_PASS_CONTRACT_PATCH_APPLIED")
    print(f"file = {target}")
    print(f"backup = {backup}")
    print("normal XODR lookup: comparison thresholds remain mandatory")
    print("XODR None: allowed only when xodr_boundary_verified=True")
    print("XODR remains xodr_found=False; comparison status is explicit")
    print("route snapshot and route coordinates are unchanged")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
