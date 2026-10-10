#!/usr/bin/env bash
# Matched no-HV / V4C-C0 / V4C-C2 closed-loop evaluation.
#
# C0 and C2 use the V4C online entry.  V4C preserves V4A's sampling network,
# but its provenance gate accepts only checkpoints produced by the V4C trainer.
# This script never supplies an interaction waypoint to online inference.

export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
HV_ROOT="${HV_ROOT:-$ROOT/data/hv_init}"
AE_CKPT="${AE_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt}"
LDM_CKPT="${LDM_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt}"

TAG=""
ROUTES=""
SEED=""
GPU=""
C0_CHECKPOINT=""
C2_CHECKPOINT=""
GENERATION_MAX_ATTEMPTS=1
BASE_PORT=30000
BASE_TM_PORT=50000
DISABLE_EXISTING_RIFT_POSITIVE_GATE=0
INITIAL_ONLY=0

usage() {
  printf '%s\n' "Usage: /bin/bash RIFT/scripts/run_hv_v4c_matched_online_eval.sh --tag TAG --routes ROUTE_IDS --seed SEED --gpu GPU --c0-checkpoint PATH --c2-checkpoint PATH [--generation-max-attempts N] [--no-existing-rift-positive-gate] [--initial-only] [--base-port PORT] [--base-tm-port PORT]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --routes) ROUTES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --c0-checkpoint) C0_CHECKPOINT="$2"; shift 2 ;;
    --c2-checkpoint) C2_CHECKPOINT="$2"; shift 2 ;;
    --generation-max-attempts) GENERATION_MAX_ATTEMPTS="$2"; shift 2 ;;
    --no-existing-rift-positive-gate) DISABLE_EXISTING_RIFT_POSITIVE_GATE=1; shift ;;
    --initial-only) INITIAL_ONLY=1; shift ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --base-tm-port) BASE_TM_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$TAG" ] || [ -z "$ROUTES" ] || [ -z "$SEED" ] || [ -z "$GPU" ] || [ -z "$C0_CHECKPOINT" ] || [ -z "$C2_CHECKPOINT" ]; then
  usage
  exit 2
fi
case "$GENERATION_MAX_ATTEMPTS" in ''|*[!0-9]*|0) printf '%s\n' 'generation-max-attempts must be a positive integer' >&2; exit 2 ;; esac
case "$DISABLE_EXISTING_RIFT_POSITIVE_GATE" in 0|1) ;; *) printf '%s\n' 'invalid existing-RIFT gate setting' >&2; exit 2 ;; esac
case "$INITIAL_ONLY" in 0|1) ;; *) printf '%s\n' 'invalid initial-only setting' >&2; exit 2 ;; esac
case "$BASE_PORT" in ''|*[!0-9]*|0) printf '%s\n' 'base-port must be a positive integer' >&2; exit 2 ;; esac
case "$BASE_TM_PORT" in ''|*[!0-9]*|0) printf '%s\n' 'base-tm-port must be a positive integer' >&2; exit 2 ;; esac
if [ "$(id -u)" -ne 0 ]; then
  printf '%s\n' 'Run this launcher as root so each CARLA process can be started as riftuser.' >&2
  exit 2
fi
for path in "$PY" "$AE_CKPT" "$LDM_CKPT" "$C0_CHECKPOINT" "$C2_CHECKPOINT"; do
  if [ ! -e "$path" ]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 2
  fi
done

RUN_DIR="$HV_ROOT/eval/$TAG"
NO_HV_OUT="no_hv_$TAG"
C0_OUT="hv_v4c_c0_$TAG"
C2_OUT="hv_v4c_c2_$TAG"
C0_DUMP="$RUN_DIR/c0_dump"
C2_DUMP="$RUN_DIR/c2_dump"
if [ -e "$RUN_DIR" ] || [ -e "$ROOT/RIFT/$NO_HV_OUT" ] || [ -e "$ROOT/RIFT/$C0_OUT" ] || [ -e "$ROOT/RIFT/$C2_OUT" ]; then
  printf 'Refusing to overwrite an existing matched evaluation tag: %s\n' "$TAG" >&2
  exit 2
fi
mkdir -p "$RUN_DIR" "$ROOT/RIFT/$NO_HV_OUT" "$ROOT/RIFT/$C0_OUT" "$ROOT/RIFT/$C2_OUT"
chown -R riftuser:riftuser "$RUN_DIR" "$ROOT/RIFT/$NO_HV_OUT" "$ROOT/RIFT/$C0_OUT" "$ROOT/RIFT/$C2_OUT"

run_no_hv() {
  runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" OUT="$NO_HV_OUT" ROUTES="$ROUTES" SEED="$SEED" GPU="$GPU" BASE_PORT="$BASE_PORT" BASE_TM_PORT="$BASE_TM_PORT" /bin/bash -lc '
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
    export CARLA_ROOT="$ROOT/carla"
    export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
    export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)"
    export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
    cd "$ROOT/RIFT" || exit 2
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run.py \
      --mode eval --output_dir "$OUT" --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
      --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
      --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 --save_bev \
      --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
      --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
      --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits"
  '
}

run_hv() {
  local checkpoint="$1" out="$2" dump="$3"
  runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" AE_CKPT="$AE_CKPT" LDM_CKPT="$LDM_CKPT" \
    CHECKPOINT="$checkpoint" OUT="$out" DUMP="$dump" ROUTES="$ROUTES" SEED="$SEED" GPU="$GPU" \
    GENERATION_MAX_ATTEMPTS="$GENERATION_MAX_ATTEMPTS" DISABLE_EXISTING_RIFT_POSITIVE_GATE="$DISABLE_EXISTING_RIFT_POSITIVE_GATE" \
    INITIAL_ONLY="$INITIAL_ONLY" BASE_PORT="$BASE_PORT" BASE_TM_PORT="$BASE_TM_PORT" /bin/bash -lc '
      export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
      export CARLA_ROOT="$ROOT/carla"
      export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
      export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)"
      export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
      cd "$ROOT/RIFT" || exit 2
      if [ "$DISABLE_EXISTING_RIFT_POSITIVE_GATE" = "1" ]; then
        EXISTING_RIFT_GATE_ARG="--no_hv_generation_gate_on_existing_rift_positive"
      else
        EXISTING_RIFT_GATE_ARG="--hv_generation_existing_rift_positive_threshold 2"
      fi
      if [ "$INITIAL_ONLY" = "1" ]; then
        ROLLING_ARGS=""
      else
        ROLLING_ARGS="--hv_generation_rolling --hv_generation_rolling_min_step 50 --hv_generation_rolling_interval_steps 50"
      fi
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run.py \
        --mode eval --output_dir "$OUT" --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
        --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
        --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 --save_bev \
        --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
        --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
        --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \
        --high_value_init_generation \
        --hv_generation_generator_entry high_value_init.scene_generator_v4c:ScenarioDreamerHighValueSceneGeneratorV4C \
        --hv_generation_checkpoint "$CHECKPOINT" --hv_autoencoder_checkpoint "$AE_CKPT" --hv_ldm_checkpoint "$LDM_CKPT" \
        --hv_generation_mode full_fov_inpainting --hv_num_generate_agents 2 --hv_sd_fov 64 --hv_map_radius 64 \
        --hv_snapshot_map_coordinate_mode right_handed --hv_generation_max_attempts "$GENERATION_MAX_ATTEMPTS" \
        $ROLLING_ARGS \
        --hv_generation_lifetime_radius 45 $EXISTING_RIFT_GATE_ARG \
        --hv_generation_lifecycle_telemetry --no_hv_generation_fail_on_invalid --hv_generation_dump_dir "$DUMP"
    '
}

printf '\n===== no-HV baseline: routes=%s seed=%s =====\n' "$ROUTES" "$SEED"
run_no_hv || exit 1
printf '\n===== V4C C0: same data, no route-interaction loss =====\n'
run_hv "$C0_CHECKPOINT" "$C0_OUT" "$C0_DUMP" || exit 1
printf '\n===== V4C C2: direct RIFT route-progress/synchronization loss =====\n'
run_hv "$C2_CHECKPOINT" "$C2_OUT" "$C2_DUMP" || exit 1

export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}"
cd "$ROOT/RIFT" || exit 2
"$PY" scripts/summarize_high_value_generation_online.py --dump_dir "$C0_DUMP" --statistics_dir "$ROOT/RIFT/$C0_OUT" --output_json "$C0_DUMP/summary.json" || exit 1
"$PY" scripts/summarize_high_value_generation_online.py --dump_dir "$C2_DUMP" --statistics_dir "$ROOT/RIFT/$C2_OUT" --output_json "$C2_DUMP/summary.json" || exit 1
"$PY" scripts/compare_hv_online_runs.py \
  --run no_hv - "$ROOT/RIFT/$NO_HV_OUT" \
  --run c0 "$C0_DUMP" "$ROOT/RIFT/$C0_OUT" \
  --run c2 "$C2_DUMP" "$ROOT/RIFT/$C2_OUT" \
  --baseline no_hv --output_json "$RUN_DIR/matched_comparison.json" --output_markdown "$RUN_DIR/matched_comparison.md" || exit 1
"$PY" scripts/analyze_hv_eventual_rift_cbv.py --dump_dir "$C0_DUMP" \
  --source high_value --expected_hv_per_event 2 --output_json "$RUN_DIR/c0_eventual_rift_cbv_analysis.json" || exit 1
"$PY" scripts/analyze_hv_eventual_rift_cbv.py --dump_dir "$C2_DUMP" \
  --source high_value --expected_hv_per_event 2 --output_json "$RUN_DIR/c2_eventual_rift_cbv_analysis.json" || exit 1
printf '\nCOMPLETE: %s\n' "$RUN_DIR"
