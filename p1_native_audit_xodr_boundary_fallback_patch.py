#!/usr/bin/env python3
"""Safely patch hv_v8/p1_native_audit.py to tolerate verified XODR endpoint lookup misses.

The fallback is accepted only when map.get_waypoint(expected.location) returns a waypoint
whose road/lane/section identity, s coordinate, position, and yaw match the snapshot route
point. The original get_waypoint_xodr miss remains recorded as xodr_found=False and the
route check is annotated with fallback_used=True.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import re
import shutil
import sys
from pathlib import Path


PHRASE = "get_waypoint_xodr returned None"


def attr_call(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    )


def target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for item in node.elts:
            out.extend(target_names(item))
        return out
    return []


def is_var_is_none(node: ast.If, var_name: str) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == var_name
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Is)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value is None
    )


def find_route_checks_return(fn: ast.FunctionDef) -> ast.Return | None:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Tuple):
            continue
        names = [
            item.id if isinstance(item, ast.Name) else None
            for item in node.value.elts
        ]
        if "route" in names and "route_checks" in names:
            return node
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="hv_v8/p1_native_audit.py",
        help="Path to p1_native_audit.py (default: hv_v8/p1_native_audit.py)",
    )
    args = parser.parse_args()

    target = Path(args.file).resolve()
    if not target.is_file():
        print(f"ERROR: target file does not exist: {target}", file=sys.stderr)
        return 2

    source = target.read_text(encoding="utf-8")
    if "_p1_xodr_fallback_indices" in source and "fallback_used" in source:
        print("P1_XODR_FALLBACK_PATCH_ALREADY_PRESENT")
        print(f"file = {target}")
        return 0

    try:
        tree = ast.parse(source, filename=str(target))
    except SyntaxError as exc:
        print(f"ERROR: target file has syntax error before patch: {exc}", file=sys.stderr)
        return 2

    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "route_waypoints"
    ]
    if len(functions) != 1:
        print(
            f"ERROR: expected exactly one route_waypoints() function, found {len(functions)}",
            file=sys.stderr,
        )
        return 2
    fn = functions[0]
    if not {"carla", "carla_map", "scene"}.issubset(
        {arg.arg for arg in fn.args.args}
    ):
        print("ERROR: route_waypoints() signature differs from expected; no changes made.", file=sys.stderr)
        return 2

    lines = source.splitlines(keepends=True)

    # Identify the variable assigned from get_waypoint_xodr().
    xodr_assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and attr_call(node.value, "get_waypoint_xodr"):
            for target_node in node.targets:
                for name in target_names(target_node):
                    xodr_assignments.append((name, node))
        elif isinstance(node, ast.AnnAssign) and attr_call(node.value, "get_waypoint_xodr"):
            for name in target_names(node.target):
                xodr_assignments.append((name, node))

    unique_vars = sorted(set(name for name, _ in xodr_assignments))
    if len(unique_vars) != 1:
        print(
            "ERROR: expected one simple variable assigned from get_waypoint_xodr(); "
            f"found {unique_vars}. No changes made.",
            file=sys.stderr,
        )
        return 2
    wp_var = unique_vars[0]

    # Locate the exact None-guard that currently raises the known fatal error.
    none_guards: list[ast.If] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not is_var_is_none(node, wp_var):
            continue
        segment = "\n".join(lines[node.lineno - 1:node.end_lineno])
        if PHRASE in segment:
            none_guards.append(node)
    if len(none_guards) != 1:
        print(
            f"ERROR: expected one `if {wp_var} is None` guard containing "
            f"{PHRASE!r}; found {len(none_guards)}. No changes made.",
            file=sys.stderr,
        )
        return 2
    guard = none_guards[0]
    guard_text = "\n".join(lines[guard.lineno - 1:guard.end_lineno])
    idx_match = re.search(r"ego_route\[\{([A-Za-z_][A-Za-z0-9_]*)\}\]", guard_text)
    idx_name = idx_match.group(1) if idx_match else "idx"

    ret = find_route_checks_return(fn)
    if ret is None:
        print(
            "ERROR: could not locate `return route, route_checks`; no changes made.",
            file=sys.stderr,
        )
        return 2

    # Initialize the fallback-index set immediately after the function docstring
    # (or before its first statement if it has no docstring).
    first_stmt = fn.body[0]
    if (
        isinstance(first_stmt, ast.Expr)
        and isinstance(first_stmt.value, ast.Constant)
        and isinstance(first_stmt.value.value, str)
    ):
        init_after_line = first_stmt.end_lineno
    else:
        init_after_line = first_stmt.lineno - 1
    fn_indent = " " * (fn.col_offset + 4)

    guard_indent = lines[guard.lineno - 1][:len(lines[guard.lineno - 1]) -
                                              len(lines[guard.lineno - 1].lstrip())]

    replacement_body = [
        f"{guard_indent}if {wp_var} is None:\n",
        f"{guard_indent}    _p1_route_point = scene['ego_route'][{idx_name}]\n",
        f"{guard_indent}    _p1_fallback_wp = None\n",
        f"{guard_indent}    _p1_fallback_method = None\n",
        f"{guard_indent}    try:\n",
        f"{guard_indent}        _p1_fallback_wp = carla_map.get_waypoint(\n",
        f"{guard_indent}            expected.location, project_to_road=False,\n",
        f"{guard_indent}            lane_type=carla.LaneType.Any,\n",
        f"{guard_indent}        )\n",
        f"{guard_indent}    except TypeError:\n",
        f"{guard_indent}        _p1_fallback_wp = carla_map.get_waypoint(\n",
        f"{guard_indent}            expected.location, project_to_road=False\n",
        f"{guard_indent}        )\n",
        f"{guard_indent}    if _p1_fallback_wp is not None:\n",
        f"{guard_indent}        _p1_fallback_method = 'get_waypoint_location_exact'\n",
        f"{guard_indent}    if _p1_fallback_wp is None:\n",
        f"{guard_indent}        try:\n",
        f"{guard_indent}            _p1_fallback_wp = carla_map.get_waypoint(\n",
        f"{guard_indent}                expected.location, project_to_road=True,\n",
        f"{guard_indent}                lane_type=carla.LaneType.Any,\n",
        f"{guard_indent}            )\n",
        f"{guard_indent}        except TypeError:\n",
        f"{guard_indent}            _p1_fallback_wp = carla_map.get_waypoint(\n",
        f"{guard_indent}                expected.location, project_to_road=True\n",
        f"{guard_indent}            )\n",
        f"{guard_indent}        if _p1_fallback_wp is not None:\n",
        f"{guard_indent}            _p1_fallback_method = 'get_waypoint_location_projected'\n",
        f"{guard_indent}    if _p1_fallback_wp is None:\n",
        f"{guard_indent}        raise RuntimeError(\n",
        f"{guard_indent}            f'ego_route[{{{idx_name}}}] get_waypoint_xodr returned None; '\n",
        f"{guard_indent}            'exact and projected location lookup also returned None'\n",
        f"{guard_indent}        )\n",
        f"{guard_indent}    _p1_expected_road = int(_p1_route_point['road_id'])\n",
        f"{guard_indent}    _p1_expected_lane = int(_p1_route_point['lane_id'])\n",
        f"{guard_indent}    _p1_section_raw = _p1_route_point.get('section_id')\n",
        f"{guard_indent}    _p1_s_raw = _p1_route_point.get('s')\n",
        f"{guard_indent}    _p1_identity_ok = (\n",
        f"{guard_indent}        int(_p1_fallback_wp.road_id) == _p1_expected_road\n",
        f"{guard_indent}        and int(_p1_fallback_wp.lane_id) == _p1_expected_lane\n",
        f"{guard_indent}        and (\n",
        f"{guard_indent}            _p1_section_raw is None\n",
        f"{guard_indent}            or int(_p1_fallback_wp.section_id) == int(_p1_section_raw)\n",
        f"{guard_indent}        )\n",
        f"{guard_indent}    )\n",
        f"{guard_indent}    _p1_position_error_m = _p1_fallback_wp.transform.location.distance(\n",
        f"{guard_indent}        expected.location\n",
        f"{guard_indent}    )\n",
        f"{guard_indent}    _p1_yaw_error_deg = abs((\n",
        f"{guard_indent}        float(_p1_fallback_wp.transform.rotation.yaw)\n",
        f"{guard_indent}        - float(expected.rotation.yaw) + 180.0\n",
        f"{guard_indent}    ) % 360.0 - 180.0)\n",
        f"{guard_indent}    _p1_s_error_m = (\n",
        f"{guard_indent}        abs(float(_p1_fallback_wp.s) - float(_p1_s_raw))\n",
        f"{guard_indent}        if _p1_s_raw is not None else 0.0\n",
        f"{guard_indent}    )\n",
        f"{guard_indent}    if (\n",
        f"{guard_indent}        not _p1_identity_ok\n",
        f"{guard_indent}        or _p1_position_error_m > 0.25\n",
        f"{guard_indent}        or _p1_yaw_error_deg > 5.0\n",
        f"{guard_indent}        or _p1_s_error_m > 0.25\n",
        f"{guard_indent}    ):\n",
        f"{guard_indent}        raise RuntimeError(\n",
        f"{guard_indent}            f'ego_route[{{{idx_name}}}] XODR lookup failed and location fallback '\n",
        f"{guard_indent}            f'failed identity/tolerance checks: method={{_p1_fallback_method}}, '\n",
        f"{guard_indent}            f'road={{_p1_fallback_wp.road_id}}/{{_p1_expected_road}}, '\n",
        f"{guard_indent}            f'lane={{_p1_fallback_wp.lane_id}}/{{_p1_expected_lane}}, '\n",
        f"{guard_indent}            f'position_error_m={{_p1_position_error_m:.6f}}, '\n",
        f"{guard_indent}            f'yaw_error_deg={{_p1_yaw_error_deg:.6f}}, '\n",
        f"{guard_indent}            f's_error_m={{_p1_s_error_m:.6f}}'\n",
        f"{guard_indent}        )\n",
        f"{guard_indent}    {wp_var} = _p1_fallback_wp\n",
        f"{guard_indent}    _p1_xodr_fallback_indices.add({idx_name})\n",
    ]

    # Add a final annotation to route_checks while preserving the fact that XODR failed.
    return_indent = lines[ret.lineno - 1][:len(lines[ret.lineno - 1]) -
                                           len(lines[ret.lineno - 1].lstrip())]
    annotation = [
        f"{return_indent}for _p1_check in route_checks:\n",
        f"{return_indent}    if (\n",
        f"{return_indent}        isinstance(_p1_check, dict)\n",
        f"{return_indent}        and _p1_check.get('route_index') in _p1_xodr_fallback_indices\n",
        f"{return_indent}    ):\n",
        f"{return_indent}        _p1_check['xodr_found'] = False\n",
        f"{return_indent}        _p1_check['fallback_used'] = True\n",
        f"{return_indent}        _p1_check['fallback_method'] = 'get_waypoint(location)'\n",
        f"{return_indent}        _p1_check['fallback_identity_match'] = True\n",
    ]

    edits = [
        (guard.lineno - 1, guard.end_lineno, replacement_body),
        (init_after_line, init_after_line, [fn_indent + "_p1_xodr_fallback_indices = set()\n"]),
        (ret.lineno - 1, ret.lineno - 1, annotation),
    ]
    # Apply from bottom to top so the original AST line numbers remain valid.
    for start, end, replacement in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        lines[start:end] = replacement

    patched = "".join(lines)
    try:
        ast.parse(patched, filename=str(target))
        compile(patched, str(target), "exec")
    except Exception as exc:
        print(f"ERROR: generated patch failed syntax validation: {exc}", file=sys.stderr)
        return 2

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + f".bak_xodr_fallback_{stamp}")
    shutil.copy2(target, backup)
    target.write_text(patched, encoding="utf-8")

    print("P1_XODR_BOUNDARY_FALLBACK_PATCH_APPLIED")
    print(f"file = {target}")
    print(f"backup = {backup}")
    print(f"xodr_variable = {wp_var}")
    print(f"route_index_variable = {idx_name}")
    print("guards = road_id/lane_id/section_id, position<=0.25m, yaw<=5deg, s<=0.25m")
    print("audit_semantics = xodr_found remains false; fallback_used=true on affected route_checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
