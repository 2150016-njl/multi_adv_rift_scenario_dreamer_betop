#!/usr/bin/env bash
# One-record P1 run. This script always invokes hv_v8.p1_native_audit as a module.
# Never runs p1_native_audit.py by file path (its relative imports require a package).
# Keeps logs outside the P1 output directory and never pre-creates --output.

set +e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${P1_PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
TARGET="$ROOT/hv_v8/p1_native_audit.py"
MAP_PATCH="$ROOT/hv_v8/p1_map_identity_manifest_runtime_v2_patch.py"
P1_MANIFEST="$ROOT/data/hv_init/v7/manifests/p1_calibration95.jsonl"
MAP_IDENTITY_MANIFEST="$ROOT/data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json"
RIFT_CONFIG="$ROOT/RIFT/rift/cbv/recognition/config/attention.yaml"
CARLA_ROOT="$ROOT/carla"
CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
CARLA_EGG="$CARLA_PYTHONAPI/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$ROOT/data/hv_init/v7"
P1_LOG="$LOG_DIR/p1_smoke_module_fixed_${STAMP}.log"
P1_SMOKE="$LOG_DIR/p1_smoke_module_fixed_${STAMP}"
HELP_FILE="$LOG_DIR/p1_module_help_${STAMP}.txt"

if [[ ! -d "$LOG_DIR" ]]; then
  mkdir -p "$LOG_DIR" || exit 90
fi

exec > >(tee -a "$P1_LOG") 2>&1

finish() {
  local rc=$?
  trap - EXIT
  echo
  echo "RUNNER_EXIT=$rc"
  echo "P1_LOG=$P1_LOG"
  echo "P1_SMOKE=$P1_SMOKE"
  echo "All output above was saved to P1_LOG."
  if [[ -t 0 ]]; then
    printf '运行结束；按 Enter 返回原来的终端（不会再启动新的 Shell）：'
    IFS= read -r _
  fi
}
trap finish EXIT

stop() {
  echo "P1_STOP: $*"
  exit 2
}

cd "$ROOT" || stop "cannot enter project root"
export PYTHONPATH="$ROOT:$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
export TARGET

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "ACTIVE_MODULE=hv_v8.p1_native_audit"
echo "TARGET=$TARGET"
echo "MAP_IDENTITY_MANIFEST=$MAP_IDENTITY_MANIFEST"
echo "P1_LOG=$P1_LOG"
echo "P1_SMOKE=$P1_SMOKE"

for f in "$TARGET" "$P1_MANIFEST" "$MAP_IDENTITY_MANIFEST" "$RIFT_CONFIG"; do
  [[ -f "$f" ]] || stop "missing required file: $f"
done
[[ -x "$PY" ]] || stop "Python executable not found: $PY"
[[ -d "$CARLA_PYTHONAPI" ]] || stop "CARLA PythonAPI directory missing: $CARLA_PYTHONAPI"

# Verify -m uses THIS exact local module, not a different environment's copy.
"$PY" - <<'PY'
import importlib.util
import os
from pathlib import Path
spec = importlib.util.find_spec('hv_v8.p1_native_audit')
assert spec is not None and spec.origin, 'Cannot find module hv_v8.p1_native_audit'
actual = Path(spec.origin).resolve()
expected = Path(os.environ['TARGET']).resolve()
print('MODULE_PATH=' + str(actual))
assert actual == expected, 'Wrong Python module loaded: {} != {}'.format(actual, expected)
print('P1_MODULE_IDENTITY_PASS')
PY
rc=$?
[[ "$rc" -eq 0 ]] || stop "Python package/module path mismatch"

# Critical fix: use -m for argparse help, NEVER "$PY" "$TARGET" --help.
echo
 echo '=== MODULE HELP AND CLI PRECHECK ==='
"$PY" -m hv_v8.p1_native_audit --help > "$HELP_FILE" 2>&1
rc=$?
if [[ "$rc" -ne 0 ]]; then
  cat "$HELP_FILE"
  stop "module --help failed (rc=$rc)"
fi

if ! grep -q -- '--map-identity-manifest' "$HELP_FILE"; then
  echo 'Map identity CLI option missing; checking existing patch before smoke.'
  if grep -q 'P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2' "$TARGET"; then
    stop 'patch marker exists but CLI flag is absent; do not run CARLA'
  fi
  [[ -f "$MAP_PATCH" ]] || stop "missing runtime patch: $MAP_PATCH"
  "$PY" "$MAP_PATCH" --target "$TARGET" --dry-run
  rc=$?
  [[ "$rc" -eq 0 ]] || stop "map identity patch dry-run failed (rc=$rc)"
  "$PY" "$MAP_PATCH" --target "$TARGET"
  rc=$?
  [[ "$rc" -eq 0 ]] || stop "map identity patch apply failed (rc=$rc)"
  "$PY" -m py_compile "$TARGET"
  rc=$?
  [[ "$rc" -eq 0 ]] || stop "P1 compile failed after map identity patch"
  "$PY" -m hv_v8.p1_native_audit --help > "$HELP_FILE" 2>&1
  rc=$?
  [[ "$rc" -eq 0 ]] || { cat "$HELP_FILE"; stop "module --help failed after patch"; }
fi

grep -F -- '--map-identity-manifest' "$HELP_FILE" || stop 'module CLI did not expose --map-identity-manifest'
"$PY" -m py_compile "$TARGET" || stop 'active P1 source does not compile'

# Verify real route_waypoints function/call contract in the current active source.
"$PY" - <<'PY'
import ast
from pathlib import Path
root = ast.parse(Path('hv_v8/p1_native_audit.py').read_text(encoding='utf-8'))
routes = [n for n in ast.walk(root) if isinstance(n, ast.FunctionDef) and n.name == 'route_waypoints']
audits = [n for n in ast.walk(root) if isinstance(n, ast.FunctionDef) and n.name == 'audit_record']
assert len(routes) == len(audits) == 1, 'Expected one route_waypoints and one audit_record'
rargs = routes[0].args.args
assert rargs and rargs[0].arg == 'carla', 'route_waypoints does not take carla as first argument'
calls = [n for n in ast.walk(audits[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'route_waypoints']
assert len(calls) == 1 and calls[0].args and isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == 'carla', 'audit_record does not pass carla'
print('P1_ROUTE_CARLA_SCOPE_CONTRACT_PASS')
PY
rc=$?
[[ "$rc" -eq 0 ]] || stop 'route call contract incomplete; smoke not started'
echo 'P1_RUNTIME_ENTRY_PREFLIGHT_PASS'

echo
 echo '=== REAL CARLA P1 SMOKE (ONE RECORD) ==='
[[ ! -e "$P1_SMOKE" ]] || stop "output already exists: $P1_SMOKE"
"$PY" -m hv_v8.p1_native_audit \
    --manifest "$P1_MANIFEST" \
    --rift-root "$ROOT/RIFT" \
    --rift-recognition-config "$RIFT_CONFIG" \
    --rift-mode eval \
    --carla-pythonapi "$CARLA_PYTHONAPI" \
    --map-identity-manifest "$MAP_IDENTITY_MANIFEST" \
    --host 127.0.0.1 \
    --port 31120 \
    --startup-timeout 600 \
    --startup-rpc-timeout 5 \
    --startup-poll-interval 2 \
    --rpc-timeout 120 \
    --expected-carla-version 0.9.15 \
    --limit 1 \
    --output "$P1_SMOKE"
P1_RC=$?
echo "P1_SMOKE_EXIT=$P1_RC"

if [[ -f "$P1_SMOKE/summary.json" ]]; then
  echo '=== REAL SUMMARY.JSON ==='
  "$PY" - "$P1_SMOKE" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
record_file = out / 'record_0000.json'
record = json.loads(record_file.read_text(encoding='utf-8')) if record_file.exists() else {}
identity = record.get('map_identity') or {}
reported = {
  'p1_pass': summary.get('p1_pass'),
  'map_identity_all_pass': summary.get('map_identity_all_pass'),
  'target_selection_all_pass': summary.get('target_selection_all_pass'),
  'coordinate_roundtrip_all_pass': summary.get('coordinate_roundtrip_all_pass'),
  'rift_reference_parity_all_pass': summary.get('rift_reference_parity_all_pass'),
  'historical_native_unknown_count': summary.get('historical_native_unknown_count'),
  'interaction_error_count': summary.get('interaction_error_count'),
  'map_identity': identity,
  'raw_pose_statuses': [
      {
        'actor_id': x.get('actor_id'),
        'status': x.get('status'),
        'pose_validation_status': x.get('pose_validation_status'),
        'position_error_m': x.get('position_error_m'),
      }
      for x in record.get('fixed_model3_raw_pose_spawn', [])
  ],
}
print(json.dumps(reported, indent=2, ensure_ascii=False))
PY
else
  echo 'P1_SUMMARY_NOT_GENERATED'
  [[ ! -d "$P1_SMOKE" ]] || find "$P1_SMOKE" -maxdepth 2 -type f -print | sort
fi

exit "$P1_RC"
