#!/usr/bin/env bash
# Run P1 patch/static checks without losing the terminal or diagnostic output.
# Optional CARLA smoke test: RUN_P1_SMOKE=1 RIFT_CONFIG=/path/to/config.yaml bash this_script.sh

set +e
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
TARGET="$ROOT/hv_v8/p1_native_audit.py"
PATCH="$ROOT/hv_v8/p1_map_identity_manifest_runtime_v2_patch.py"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/data/hv_init/v7"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/p1_keep_terminal_${STAMP}.log"
FAILED=0

exec > >(tee -a "$LOG") 2>&1

finish() {
    rc=$?
    echo
    echo "RUNNER_EXIT=$rc"
    echo "RUNNER_LOG=$LOG"
    echo "Keep this log if any step failed."
    if [ -e /dev/tty ]; then
        read -r -p "Execution finished. Press Enter to close this terminal..." _ </dev/tty
    fi
}
trap finish EXIT

fail_now() {
    echo "ERROR: $*"
    exit 1
}

cd "$ROOT" || fail_now "cannot cd to project root: $ROOT"

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "TARGET=$TARGET"
echo "LOG=$LOG"

[ -x "$PY" ] || fail_now "Python executable unavailable: $PY"
[ -f "$TARGET" ] || fail_now "active P1 source missing: $TARGET"
[ -f "$PATCH" ] || fail_now "patch file missing; upload it to: $PATCH"

echo
echo "=== STEP 1: compile patch file ==="
"$PY" -m py_compile "$PATCH"
rc=$?
echo "PATCH_PYCOMPILE_RC=$rc"
[ "$rc" -eq 0 ] || fail_now "patch script did not compile"

echo
echo "=== STEP 2: patch self-test ==="
"$PY" "$PATCH" --self-test
rc=$?
echo "PATCH_SELF_TEST_RC=$rc"
[ "$rc" -eq 0 ] || fail_now "patch self-test failed"

echo
echo "=== STEP 3: apply patch only if not already applied ==="
if grep -q 'P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2' "$TARGET"; then
    echo "PATCH_ALREADY_PRESENT_SKIP_REAPPLY"
else
    "$PY" "$PATCH" --target "$TARGET"
    rc=$?
    echo "PATCH_APPLY_RC=$rc"
    [ "$rc" -eq 0 ] || fail_now "patch application failed"
fi

echo
echo "=== STEP 4: compile active P1 source ==="
"$PY" -m py_compile "$TARGET"
rc=$?
echo "ACTIVE_P1_PYCOMPILE_RC=$rc"
[ "$rc" -eq 0 ] || fail_now "active P1 source did not compile"
echo "P1_PYCOMPILE_PASS"

echo
echo "=== STEP 5: verify CLI wiring ==="
HELP_FILE="$LOG_DIR/p1_help_${STAMP}.txt"
"$PY" -m hv_v8.p1_native_audit --help > "$HELP_FILE" 2>&1
help_rc=$?
echo "P1_HELP_RC=$help_rc"
grep -F -- '--map-identity-manifest' "$HELP_FILE"
grep_rc=$?
echo "P1_MAP_IDENTITY_CLI_GREP_RC=$grep_rc"
[ "$help_rc" -eq 0 ] || fail_now "P1 --help command failed; see $HELP_FILE"
[ "$grep_rc" -eq 0 ] || fail_now "CLI option --map-identity-manifest not registered"
grep -n -E 'P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2|_p1_load_map_identity_manifest|_p1_resolve_expected_opendrive_sha256|opendrive_expected_source|map_identity_manifest' "$TARGET"
echo "P1_MAP_IDENTITY_RUNTIME_STATIC_ACCEPTANCE_PASS"

if [ "${RUN_P1_SMOKE:-0}" != "1" ]; then
    echo
    echo "SMOKE_SKIPPED"
    echo "To run CARLA smoke after static checks, set RUN_P1_SMOKE=1 and RIFT_CONFIG=/absolute/path/to/your/working_RIFT_config.yaml."
    exit 0
fi

echo
echo "=== STEP 6: CARLA smoke test ==="
CARLA_ROOT="$ROOT/carla"
CARLA_PYTHONAPI="${CARLA_PYTHONAPI:-$CARLA_ROOT/PythonAPI/carla}"
CARLA_EGG="${CARLA_EGG:-$CARLA_PYTHONAPI/dist/carla-0.9.15-py3.7-linux-x86_64.egg}"
P1_MANIFEST="${P1_MANIFEST:-$ROOT/data/hv_init/v7/manifests/p1_calibration95.jsonl}"
MAP_IDENTITY_MANIFEST="${MAP_IDENTITY_MANIFEST:-$ROOT/data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json}"
RIFT_CONFIG="${RIFT_CONFIG:-}"

[ -f "$P1_MANIFEST" ] || fail_now "P1 manifest missing: $P1_MANIFEST"
[ -f "$MAP_IDENTITY_MANIFEST" ] || fail_now "map identity manifest missing: $MAP_IDENTITY_MANIFEST"
[ -n "$RIFT_CONFIG" ] || fail_now "set RIFT_CONFIG to the actual recognition config path before smoke"
[ -f "$RIFT_CONFIG" ] || fail_now "RIFT_CONFIG does not exist: $RIFT_CONFIG"

export PYTHONPATH="$ROOT:$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
P1_SMOKE="$ROOT/data/hv_init/v7/p1_smoke_map_identity_runtime_v2_${STAMP}"
echo "P1_SMOKE=$P1_SMOKE"
"$PY" -m hv_v8.p1_native_audit \
  --manifest "$P1_MANIFEST" \
  --rift-root "$ROOT/RIFT" \
  --rift-recognition-config "$RIFT_CONFIG" \
  --rift-mode eval \
  --carla-pythonapi "$CARLA_PYTHONAPI" \
  --host 127.0.0.1 \
  --port 31120 \
  --startup-timeout 600 \
  --startup-rpc-timeout 5 \
  --startup-poll-interval 2 \
  --rpc-timeout 120 \
  --expected-carla-version 0.9.15 \
  --limit 1 \
  --output "$P1_SMOKE" \
  --map-identity-manifest "$MAP_IDENTITY_MANIFEST"
P1_RC=$?
echo "P1_EXIT=$P1_RC"
echo "P1_SMOKE=$P1_SMOKE"

echo
echo "=== STEP 7: map identity acceptance ==="
"$PY" - "$P1_SMOKE" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary_path = out / "summary.json"
record_path = out / "record_0000.json"
if not summary_path.is_file() or not record_path.is_file():
    print("P1_MAP_IDENTITY_RUNTIME_ACCEPTANCE_FAIL: missing output record/summary")
    sys.exit(1)

summary = json.loads(summary_path.read_text(encoding="utf-8"))
record = json.loads(record_path.read_text(encoding="utf-8"))
identity = record.get("map_identity") or {}
report = {
    "town_expected": identity.get("town_expected"),
    "town_actual": identity.get("town_actual"),
    "expected_source": identity.get("opendrive_expected_source"),
    "expected_reason": identity.get("opendrive_expected_reason"),
    "expected_sha256": identity.get("opendrive_sha256_expected"),
    "actual_sha256": identity.get("opendrive_sha256_actual"),
    "opendrive_status": identity.get("opendrive_status"),
    "opendrive_pass": identity.get("opendrive_pass"),
    "map_identity_all_pass": summary.get("map_identity_all_pass"),
    "p1_pass": summary.get("p1_pass"),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
checks = [
    identity.get("opendrive_expected_source") == "map_identity_manifest",
    identity.get("opendrive_status") == "pass",
    identity.get("opendrive_sha256_expected") == identity.get("opendrive_sha256_actual"),
    summary.get("map_identity_all_pass") is True,
]
if not all(checks):
    print("P1_MAP_IDENTITY_RUNTIME_ACCEPTANCE_FAIL")
    sys.exit(1)
print("P1_MAP_IDENTITY_RUNTIME_ACCEPTANCE_PASS")
PY
ACCEPT_RC=$?
echo "P1_MAP_IDENTITY_ACCEPTANCE_RC=$ACCEPT_RC"
[ "$ACCEPT_RC" -eq 0 ] || fail_now "map identity acceptance failed; inspect log and smoke outputs"

if [ "$P1_RC" -eq 0 ]; then
    echo "P1_FULL_SMOKE_EXIT_ZERO"
else
    echo "NOTE: P1 exit code $P1_RC may mean another P1 contract failed; inspect summary.json. Map identity acceptance is checked separately."
fi

exit 0
