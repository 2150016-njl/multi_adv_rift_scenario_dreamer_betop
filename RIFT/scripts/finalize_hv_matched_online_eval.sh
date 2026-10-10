#!/usr/bin/env bash
# Rebuild summaries for an already-finished matched HV run without CARLA.

ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
HV_ROOT="${HV_ROOT:-$ROOT/data/hv_init}"
TAG=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash RIFT/scripts/finalize_hv_matched_online_eval.sh --tag TAG'
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$TAG" ]; then
  printf '%s\n' 'Missing required --tag' >&2
  exit 2
fi

RUN_DIR="$HV_ROOT/eval/$TAG"
NO_HV_OUT="$ROOT/RIFT/no_hv_$TAG"
V2_OUT="$ROOT/RIFT/hv_v2_$TAG"
V3_OUT="$ROOT/RIFT/hv_v3_$TAG"
V4A_OUT="$ROOT/RIFT/hv_v4a_$TAG"

for path in "$RUN_DIR/v2_dump" "$RUN_DIR/v3_dump" "$RUN_DIR/v4a_dump" "$NO_HV_OUT" "$V2_OUT" "$V3_OUT" "$V4A_OUT"; do
  if [ ! -e "$path" ]; then
    printf 'Missing finished-run evidence: %s\n' "$path" >&2
    exit 2
  fi
done

export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}"
cd "$ROOT/RIFT" || exit 2

"$PY" scripts/summarize_high_value_generation_online.py \
  --dump_dir "$RUN_DIR/v2_dump" \
  --statistics_dir "$V2_OUT" \
  --output_json "$RUN_DIR/v2_dump/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi

"$PY" scripts/summarize_high_value_generation_online.py \
  --dump_dir "$RUN_DIR/v3_dump" \
  --statistics_dir "$V3_OUT" \
  --output_json "$RUN_DIR/v3_dump/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi

"$PY" scripts/summarize_high_value_generation_online.py \
  --dump_dir "$RUN_DIR/v4a_dump" \
  --statistics_dir "$V4A_OUT" \
  --output_json "$RUN_DIR/v4a_dump/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi

"$PY" scripts/compare_hv_online_runs.py \
  --run no_hv - "$NO_HV_OUT" \
  --run v2 "$RUN_DIR/v2_dump" "$V2_OUT" \
  --run v3 "$RUN_DIR/v3_dump" "$V3_OUT" \
  --run v4a "$RUN_DIR/v4a_dump" "$V4A_OUT" \
  --baseline no_hv \
  --output_json "$RUN_DIR/matched_comparison.json" \
  --output_markdown "$RUN_DIR/matched_comparison.md"
if [ "$?" -ne 0 ]; then exit 1; fi

printf 'COMPLETE: %s\n' "$RUN_DIR"
