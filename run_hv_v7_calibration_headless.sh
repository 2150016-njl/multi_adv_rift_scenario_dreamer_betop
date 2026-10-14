#!/usr/bin/env bash
# Exact-record calibration on a NEW owned CARLA, under riftuser and -nullrhi.
set -euo pipefail
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
if [[ "${1:-}" != "--plan" || -z "${2:-}" ]]; then
  echo 'Usage: bash run_hv_v7_calibration_headless.sh --plan FILE [--port 31120]' >&2
  exit 2
fi
PLAN="$2"
shift 2
[[ -f "$PLAN" ]] || { echo "Missing calibration plan: $PLAN" >&2; exit 2; }
export HV_NATIVE_OUTPUT_ROOT="$ROOT/data/hv_init/v7/eval"
exec bash "$ROOT/run_hv_native_headless.sh" --v7-calibration-plan "$PLAN" --port 31120 "$@"
