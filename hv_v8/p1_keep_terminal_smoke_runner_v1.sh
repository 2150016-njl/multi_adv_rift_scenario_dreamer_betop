#!/usr/bin/env bash
# P1 keep-terminal runner: preflight active source, wire map manifest only if absent,
# run one-record smoke, preserve all output, and leave an interactive shell open.
# Place this file at: <repo>/hv_v8/p1_keep_terminal_smoke_runner_v1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PY="/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python"

TARGET="$ROOT/hv_v8/p1_native_audit.py"
MAP_PATCH="$ROOT/hv_v8/p1_map_identity_manifest_runtime_v2_patch.py"
MAP_IDENTITY_MANIFEST="$ROOT/data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json"
P1_MANIFEST="$ROOT/data/hv_init/v7/manifests/p1_calibration95.jsonl"
RIFT_CONFIG="$ROOT/RIFT/rift/cbv/recognition/config/attention.yaml"
CARLA_ROOT="$ROOT/carla"
CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG_DIR="$ROOT/data/hv_init/v7"
LOG="$LOG_DIR/p1_keep_terminal_smoke_$STAMP.log"
P1_SMOKE="$LOG_DIR/p1_smoke_keep_terminal_$STAMP"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "ROOT=$ROOT"
echo "PY=$PY"
echo "TARGET=$TARGET"
echo "LOG=$LOG"
echo "P1_SMOKE=$P1_SMOKE"

run_main() {
    cd "$ROOT" || return 10

    for f in "$TARGET" "$MAP_PATCH" "$MAP_IDENTITY_MANIFEST" \
             "$P1_MANIFEST" "$RIFT_CONFIG"; do
        if [[ ! -f "$f" ]]; then
            echo "P1_STOP_MISSING_FILE=$f"
            return 10
        fi
    done
    if [[ ! -d "$CARLA_PYTHONAPI" ]]; then
        echo "P1_STOP_CARLA_PYTHONAPI_MISSING=$CARLA_PYTHONAPI"
        return 10
    fi

    echo
    echo "=== STEP 1: map-manifest patch self-test and manifest validation ==="
    "$PY" -m py_compile "$MAP_PATCH" || return 11
    "$PY" "$MAP_PATCH" --self-test || return 11
    "$PY" "$MAP_PATCH" --validate-manifest "$MAP_IDENTITY_MANIFEST" || return 11

    echo
    echo "=== STEP 2: inspect actual entry-point CLI ==="
    local help_file
    help_file="$(mktemp)"
    if ! "$PY" "$TARGET" --help > "$help_file" 2>&1; then
        cat "$help_file"
        rm -f "$help_file"
        echo "P1_STOP_ACTIVE_SOURCE_HELP_FAILED"
        return 12
    fi

    if ! grep -q -- '--map-identity-manifest' "$help_file"; then
        if grep -q 'P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2' "$TARGET"; then
            cat "$help_file"
            rm -f "$help_file"
            echo "P1_STOP_PATCH_MARKER_PRESENT_BUT_CLI_OPTION_ABSENT"
            echo "No smoke was started. Active source and patch state are inconsistent."
            return 12
        fi

        echo "CLI option absent and patch marker absent; applying the existing V2 patch."
        "$PY" "$MAP_PATCH" --target "$TARGET" --dry-run || {
            rm -f "$help_file"
            return 13
        }
        "$PY" "$MAP_PATCH" --target "$TARGET" || {
            rm -f "$help_file"
            return 13
        }
        "$PY" -m py_compile "$TARGET" || {
            rm -f "$help_file"
            return 13
        }
        if ! "$PY" "$TARGET" --help > "$help_file" 2>&1; then
            cat "$help_file"
            rm -f "$help_file"
            return 13
        fi
    fi

    if ! grep -q -- '--map-identity-manifest' "$help_file"; then
        cat "$help_file"
        rm -f "$help_file"
        echo "P1_STOP_MAP_IDENTITY_CLI_STILL_ABSENT"
        return 14
    fi
    grep -n -- '--map-identity-manifest' "$help_file"
    rm -f "$help_file"

    "$PY" -m py_compile "$TARGET" || return 15
    echo "P1_MAP_IDENTITY_CLI_ACCEPTANCE_PASS"

    echo
    echo "=== STEP 3: launch one-record P1 smoke ==="
    export PYTHONPATH="$ROOT:$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_PYTHONAPI:${PYTHONPATH:-}"

    if [[ -e "$P1_SMOKE" ]]; then
        echo "P1_STOP_OUTPUT_EXISTS=$P1_SMOKE"
        return 16
    fi

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
    local p1_rc=$?

    echo "P1_SMOKE_EXIT=$p1_rc"
    echo "P1_SMOKE=$P1_SMOKE"
    if [[ -f "$P1_SMOKE/summary.json" ]]; then
        echo "=== SUMMARY ==="
        cat "$P1_SMOKE/summary.json"
    else
        echo "P1_SUMMARY_NOT_GENERATED"
        if [[ -d "$P1_SMOKE" ]]; then
            find "$P1_SMOKE" -maxdepth 2 -type f -print | sort
        fi
    fi
    return "$p1_rc"
}

run_main
RUNNER_RC=$?
echo
echo "RUNNER_EXIT=$RUNNER_RC"
echo "RUNNER_LOG=$LOG"
echo "The full output is saved to the log above."

if [[ -t 0 ]]; then
    echo
    echo "Terminal persistence enabled. Press Enter to return to an interactive shell."
    IFS= read -r _
    exec "${SHELL:-/bin/bash}" -i
else
    echo "WARNING: stdin is not a TTY; cannot hold this terminal open from the script."
    exit "$RUNNER_RC"
fi
