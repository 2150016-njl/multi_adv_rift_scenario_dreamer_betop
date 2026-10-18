#!/usr/bin/env python3
"""Fail-closed patch for the P1 route_waypoints carla-scope bug.

The patch passes the existing CARLA PythonAPI module explicitly from
audit_record() into route_waypoints(), and makes it an explicit parameter.
It supports a self-test, a no-write dry run, and an idempotent application.
"""
import argparse
import ast
import datetime
import shutil
import sys
from pathlib import Path


MARKER = "P1_ROUTE_CARLA_SCOPE_FIX_V1"


class PatchError(RuntimeError):
    pass


def _functions(tree, name):
    return [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]


def _positional_args(func):
    return list(getattr(func.args, "posonlyargs", [])) + list(func.args.args)


def _find_route_call(audit_func):
    found = []
    for node in ast.walk(audit_func):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "route_waypoints":
            found.append(node)
    if len(found) != 1:
        raise PatchError(
            "expected exactly one route_waypoints(...) call inside audit_record; "
            "found {}".format(len(found))
        )
    return found[0]


def _contract(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PatchError("source does not parse: {}".format(exc))

    route_funcs = _functions(tree, "route_waypoints")
    audit_funcs = _functions(tree, "audit_record")
    if len(route_funcs) != 1:
        raise PatchError("expected exactly one route_waypoints function; found {}".format(len(route_funcs)))
    if len(audit_funcs) != 1:
        raise PatchError("expected exactly one audit_record function; found {}".format(len(audit_funcs)))

    route_func = route_funcs[0]
    audit_func = audit_funcs[0]
    route_args = _positional_args(route_func)
    audit_args = _positional_args(audit_func)
    if not any(arg.arg == "carla" for arg in audit_args):
        raise PatchError("audit_record does not have an explicit carla parameter")

    call = _find_route_call(audit_func)
    route_has_carla_first = bool(route_args and route_args[0].arg == "carla")
    call_has_carla_first = bool(
        call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "carla"
    )
    return {
        "route_func": route_func,
        "audit_func": audit_func,
        "call": call,
        "route_has_carla_first": route_has_carla_first,
        "call_has_carla_first": call_has_carla_first,
        "route_args": [arg.arg for arg in route_args],
        "call_arg_count": len(call.args),
    }


def _line_start_offset(source, line_no):
    lines = source.splitlines(keepends=True)
    if line_no < 1 or line_no > len(lines) + 1:
        raise PatchError("invalid source line number {}".format(line_no))
    return sum(len(line) for line in lines[:line_no - 1])


def _open_paren_after(source, start, token):
    token_at = source.find(token, start)
    if token_at < 0:
        raise PatchError("could not find {!r} at expected source location".format(token))
    open_at = source.find("(", token_at + len(token))
    if open_at < 0:
        raise PatchError("could not find opening parenthesis after {!r}".format(token))
    return open_at


def _insert_first_item(source, open_at, item):
    """Return (offset, insertion_text), preserving multiline formatting."""
    after = open_at + 1
    newline = source.find("\n", after)
    if newline >= 0 and source[after:newline].strip() == "":
        # Existing arguments start on the next line. Insert a sibling line.
        arg_line_start = newline + 1
        next_newline = source.find("\n", arg_line_start)
        if next_newline < 0:
            next_newline = len(source)
        first_arg_line = source[arg_line_start:next_newline]
        indent = first_arg_line[:len(first_arg_line) - len(first_arg_line.lstrip(" \t"))]
        if not first_arg_line.strip():
            raise PatchError("cannot determine indentation for multiline argument list")
        return arg_line_start, indent + item + "\n"
    return after, item + " "


def _build_patched_source(source):
    if MARKER in source:
        state = _contract(source)
        if not state["route_has_carla_first"] or not state["call_has_carla_first"]:
            raise PatchError("patch marker exists, but the carla scope contract is incomplete")
        compile(source, "<p1_native_audit.py>", "exec")
        return source, False

    state = _contract(source)
    route_func = state["route_func"]
    call = state["call"]
    edits = []

    if not state["route_has_carla_first"]:
        route_args = _positional_args(route_func)
        if any(arg.arg == "carla" for arg in route_args):
            raise PatchError("route_waypoints has carla, but not as its first positional parameter")
        func_line_start = _line_start_offset(source, route_func.lineno)
        func_line_end = source.find("\n", func_line_start)
        if func_line_end < 0:
            func_line_end = len(source)
        def_token = "def route_waypoints"
        def_at = source.find(def_token, func_line_start, func_line_end)
        if def_at < 0:
            raise PatchError("could not locate route_waypoints definition text")
        open_at = _open_paren_after(source, def_at, "route_waypoints")
        offset, insertion = _insert_first_item(source, open_at, "carla: Any,")
        edits.append((offset, insertion))

    if not state["call_has_carla_first"]:
        if any(isinstance(arg, ast.Name) and arg.id == "carla" for arg in call.args):
            raise PatchError("route_waypoints call contains carla, but not as its first argument")
        call_line_start = _line_start_offset(source, call.lineno)
        call_line_end = source.find("\n", call_line_start)
        if call_line_end < 0:
            call_line_end = len(source)
        call_at = source.find("route_waypoints", call_line_start, call_line_end)
        if call_at < 0:
            raise PatchError("could not locate route_waypoints call text")
        open_at = _open_paren_after(source, call_at, "route_waypoints")
        offset, insertion = _insert_first_item(source, open_at, "carla,")
        edits.append((offset, insertion))

    marker_line = route_func.lineno
    if route_func.decorator_list:
        marker_line = min([d.lineno for d in route_func.decorator_list] + [route_func.lineno])
    marker_offset = _line_start_offset(source, marker_line)
    edits.append((marker_offset, "# {}\n".format(MARKER)))

    patched = source
    for offset, insertion in sorted(edits, key=lambda item: item[0], reverse=True):
        patched = patched[:offset] + insertion + patched[offset:]

    try:
        compile(patched, "<p1_native_audit.py>", "exec")
    except SyntaxError as exc:
        raise PatchError("patched source does not compile: {}".format(exc))

    final_state = _contract(patched)
    if not final_state["route_has_carla_first"] or not final_state["call_has_carla_first"]:
        raise PatchError("internal verification failed: carla is not explicit at both ends")
    if MARKER not in patched:
        raise PatchError("internal verification failed: patch marker missing")
    return patched, True


def _self_test():
    fixture = """from typing import Any

def route_waypoints(
    carla_map: Any,
    scene: dict,
):
    return carla.Transform(
        carla.Location(0, 0, 0),
        carla.Rotation(yaw=0),
    )

def audit_record(carla: Any, carla_map: Any, scene: dict):
    return route_waypoints(
        carla_map,
        scene,
    )
"""
    patched, changed = _build_patched_source(fixture)
    assert changed
    compile(patched, "<self-test>", "exec")
    state = _contract(patched)
    assert state["route_has_carla_first"]
    assert state["call_has_carla_first"]
    patched_again, changed_again = _build_patched_source(patched)
    assert not changed_again
    assert patched_again == patched
    assert "carla: Any," in patched
    assert "carla," in patched
    print("P1_ROUTE_CARLA_SCOPE_SELF_TEST_PASS")
    print("explicit carla parameter => PASS")
    print("explicit carla call argument => PASS")
    print("multiline signature/call => PASS")
    print("repeat application is idempotent => PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--target")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return 0
    if not args.target:
        parser.error("--target is required unless --self-test is used")

    target = Path(args.target).resolve()
    if not target.is_file():
        print("ERROR: target file does not exist: {}".format(target), file=sys.stderr)
        return 2

    original = target.read_text(encoding="utf-8")
    try:
        patched, changed = _build_patched_source(original)
    except PatchError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 2

    state = _contract(patched)
    if args.dry_run:
        print("P1_ROUTE_CARLA_SCOPE_DRY_RUN_PASS")
        print("target = {}".format(target))
        print("route_waypoints first parameters = {}".format(state["route_args"][:4]))
        print("route_waypoints call first argument = carla")
        print("source compiles in memory; no file was changed")
        return 0

    if not changed:
        print("P1_ROUTE_CARLA_SCOPE_ALREADY_APPLIED")
        print("file = {}".format(target))
        print("explicit function/call contract verified; no file changed")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + ".bak_route_carla_scope_" + stamp)
    shutil.copy2(str(target), str(backup))
    target.write_text(patched, encoding="utf-8")

    persisted = target.read_text(encoding="utf-8")
    compile(persisted, str(target), "exec")
    persisted_state = _contract(persisted)
    if not (persisted_state["route_has_carla_first"] and persisted_state["call_has_carla_first"]):
        print("ERROR: persisted contract verification failed", file=sys.stderr)
        return 2

    print("P1_ROUTE_CARLA_SCOPE_PATCH_APPLIED")
    print("file = {}".format(target))
    print("backup = {}".format(backup))
    print("route_waypoints receives carla explicitly")
    print("audit_record passes carla explicitly")
    print("persisted source compiles and contract is verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
