#!/usr/bin/env python3
from pathlib import Path
import ast
import shutil
import sys

ROOT = Path("/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop")
TARGET = ROOT / "hv_v8" / "p1_native_audit.py"
BACKUP = TARGET.with_suffix(".py.bak_before_route_carla_scope_fix")

src = TARGET.read_text(encoding="utf-8")
tree = ast.parse(src)

# Find the exact route_waypoints definition.
defs = [
    n for n in ast.walk(tree)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    and n.name == "route_waypoints"
]
if len(defs) != 1:
    raise SystemExit(f"Expected exactly one route_waypoints definition, found {len(defs)}")

fn = defs[0]
arg_names = [a.arg for a in fn.args.args]
if arg_names[:2] == ["carla", "carla_map"]:
    # Already fixed.
    print("P1_ROUTE_CARLA_SCOPE_ALREADY_FIXED")
    print("file =", TARGET)
    raise SystemExit(0)

if arg_names[:2] != ["carla_map", "scene"]:
    raise SystemExit(
        "Unexpected route_waypoints signature: "
        + repr(arg_names)
        + ". Refusing blind modification."
    )

# Add `carla` as the first positional argument without touching the function body.
# Reconstruct only the argument list text by using AST line/column locations.
# The local file uses a simple two-argument signature, but annotations/return
# annotation may span multiple lines. We therefore replace only the first
# argument token in the source span.
lines = src.splitlines(keepends=True)
start_line = fn.lineno - 1
start_col = fn.col_offset

# Locate the `def route_waypoints(` line and insert `carla: Any, ` before
# the existing `carla_map` argument. This is intentionally lexical and narrow.
line = lines[start_line]
needle = "def route_waypoints("
pos = line.find(needle, start_col)
if pos < 0:
    raise SystemExit("Could not locate route_waypoints definition text safely.")

insert_at = pos + len(needle)
lines[start_line] = line[:insert_at] + "carla: Any, " + line[insert_at:]
src2 = "".join(lines)

tree2 = ast.parse(src2)

# Find calls to route_waypoints and identify the call inside audit_record.
calls = []
for n in ast.walk(tree2):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "route_waypoints":
        calls.append(n)

if len(calls) != 1:
    raise SystemExit(
        f"Expected exactly one route_waypoints call, found {len(calls)}. "
        "Refusing blind modification."
    )

call = calls[0]
if [getattr(a, "id", None) for a in call.args[:2]] == ["carla", "carla_map"]:
    pass
elif [getattr(a, "id", None) for a in call.args[:2]] == ["carla_map", "scene"]:
    # Replace the call's first argument token by inserting `carla, `.
    lines2 = src2.splitlines(keepends=True)
    c_line = call.lineno - 1
    # Find route_waypoints( on the call's starting line.
    route_pos = lines2[c_line].find("route_waypoints(", call.col_offset)
    if route_pos < 0:
        raise SystemExit("Could not locate route_waypoints call text safely.")
    insert = route_pos + len("route_waypoints(")
    lines2[c_line] = lines2[c_line][:insert] + "carla, " + lines2[c_line][insert:]
    src2 = "".join(lines2)
else:
    raise SystemExit(
        "Unexpected route_waypoints call arguments: "
        + repr([getattr(a, "id", None) for a in call.args])
        + ". Refusing blind modification."
    )

# Final AST contract check.
tree3 = ast.parse(src2)
fn3 = next(
    n for n in ast.walk(tree3)
    if isinstance(n, ast.FunctionDef) and n.name == "route_waypoints"
)
args3 = [a.arg for a in fn3.args.args]
if args3[:3] != ["carla", "carla_map", "scene"]:
    raise SystemExit(f"Post-edit signature check failed: {args3}")

calls3 = [
    n for n in ast.walk(tree3)
    if isinstance(n, ast.Call)
    and isinstance(n.func, ast.Name)
    and n.func.id == "route_waypoints"
]
if len(calls3) != 1:
    raise SystemExit("Post-edit call-count check failed.")

call3 = calls3[0]
call_args3 = [getattr(a, "id", None) for a in call3.args]
if call_args3[:3] != ["carla", "carla_map", "scene"]:
    raise SystemExit(f"Post-edit call check failed: {call_args3}")

if not BACKUP.exists():
    shutil.copy2(TARGET, BACKUP)

TARGET.write_text(src2, encoding="utf-8")
print("P1_ROUTE_CARLA_SCOPE_FIX_APPLIED")
print("file =", TARGET)
print("backup =", BACKUP)
print("signature =", args3[:3])
print("call_args =", call_args3[:3])
