#!/usr/bin/env bash
# Foreground matched online comparison for no-HV, V2, V3, and V4-A.
#
# This script intentionally runs methods sequentially and streams every CARLA
# process to this terminal.  It is an evaluation launcher only: no checkpoint,
# dataset, actor state, RIFT audit result, or controller decision is modified.

# Some launch wrappers intentionally sanitize PATH. Keep the system tools this
# script needs available without relying on the caller's interactive shell.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
HV_ROOT="${HV_ROOT:-$ROOT/data/hv_init}"
AE_CKPT="${AE_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt}"
LDM_CKPT="${LDM_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt}"
V2_BEST="${V2_BEST:-$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt}"
V3_BEST="${V3_BEST:-$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/best.ckpt}"
V4A_BEST="${V4A_BEST:-$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4a_23k_exploratory_gpu3_20261006/best.ckpt}"

TAG=""
ROUTES=""
SEED=""
GPU=""
GENERATION_MAX_ATTEMPTS=1
BASE_PORT=30000
BASE_TM_PORT=50000

usage() {
  printf '%s\n' "Usage: bash RIFT/scripts/run_hv_matched_online_eval.sh --tag TAG --routes ROUTE_IDS --seed SEED --gpu GPU [--generation-max-attempts N] [--base-port PORT] [--base-tm-port PORT]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --routes) ROUTES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --generation-max-attempts) GENERATION_MAX_ATTEMPTS="$2"; shift 2 ;;
    --base-port) BASE_PORT="$2"; shift 2 ;;
    --base-tm-port) BASE_TM_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage; exit 2 ;;
  esac
done

if [ -z "$TAG" ] || [ -z "$ROUTES" ] || [ -z "$SEED" ] || [ -z "$GPU" ]; then
  usage
  exit 2
fi
case "$GENERATION_MAX_ATTEMPTS" in
  ''|*[!0-9]*|0)
    printf 'Expected a positive integer --generation-max-attempts, got: %s\n' "$GENERATION_MAX_ATTEMPTS" >&2
    exit 2
    ;;
esac
case "$BASE_PORT" in
  ''|*[!0-9]*|0)
    printf 'Expected a positive integer --base-port, got: %s\n' "$BASE_PORT" >&2
    exit 2
    ;;
esac
case "$BASE_TM_PORT" in
  ''|*[!0-9]*|0)
    printf 'Expected a positive integer --base-tm-port, got: %s\n' "$BASE_TM_PORT" >&2
    exit 2
    ;;
esac
if [ "$(id -u)" -ne 0 ]; then
  printf '%s\n' 'Run this launcher as root so each CARLA process can be started as riftuser.' >&2
  exit 2
fi
for path in "$PY" "$AE_CKPT" "$LDM_CKPT" "$V2_BEST" "$V3_BEST" "$V4A_BEST"; do
  if [ ! -e "$path" ]; then
    printf 'Missing required path: %s\n' "$path" >&2
    exit 2
  fi
done

RUN_DIR="$HV_ROOT/eval/$TAG"
NO_HV_OUT="no_hv_$TAG"
V2_OUT="hv_v2_$TAG"
V3_OUT="hv_v3_$TAG"
V4A_OUT="hv_v4a_$TAG"
V2_DUMP="$RUN_DIR/v2_dump"
V3_DUMP="$RUN_DIR/v3_dump"
V4A_DUMP="$RUN_DIR/v4a_dump"

if [ -e "$RUN_DIR" ] || [ -e "$ROOT/RIFT/$NO_HV_OUT" ] || [ -e "$ROOT/RIFT/$V2_OUT" ] || [ -e "$ROOT/RIFT/$V3_OUT" ] || [ -e "$ROOT/RIFT/$V4A_OUT" ]; then
  printf 'Refusing to overwrite an existing matched evaluation tag: %s\n' "$TAG" >&2
  exit 2
fi

mkdir -p "$RUN_DIR" "$ROOT/RIFT/$NO_HV_OUT" "$ROOT/RIFT/$V2_OUT" "$ROOT/RIFT/$V3_OUT" "$ROOT/RIFT/$V4A_OUT"
chown -R riftuser:riftuser "$RUN_DIR" "$ROOT/RIFT/$NO_HV_OUT" "$ROOT/RIFT/$V2_OUT" "$ROOT/RIFT/$V3_OUT" "$ROOT/RIFT/$V4A_OUT"

run_no_hv() {
  runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" OUT="$NO_HV_OUT" ROUTES="$ROUTES" SEED="$SEED" GPU="$GPU" BASE_PORT="$BASE_PORT" BASE_TM_PORT="$BASE_TM_PORT" bash -lc '
    export CARLA_ROOT="$ROOT/carla"
    export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
    export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)"
    export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
    cd "$ROOT/RIFT" || exit 2
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run.py \
      --mode eval --output_dir "$OUT" \
      --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
      --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
      --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 --save_bev \
      --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
      --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
      --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits"
  '
}

run_hv() {
  local method="$1" entry="$2" checkpoint="$3" mode="$4" map_coordinate_mode="$5" out="$6" dump="$7"
  runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" AE_CKPT="$AE_CKPT" LDM_CKPT="$LDM_CKPT" \
    METHOD="$method" ENTRY="$entry" CHECKPOINT="$checkpoint" MODE="$mode" MAP_COORDINATE_MODE="$map_coordinate_mode" \
    OUT="$out" DUMP="$dump" ROUTES="$ROUTES" SEED="$SEED" GPU="$GPU" GENERATION_MAX_ATTEMPTS="$GENERATION_MAX_ATTEMPTS" BASE_PORT="$BASE_PORT" BASE_TM_PORT="$BASE_TM_PORT" bash -lc '
      export CARLA_ROOT="$ROOT/carla"
      export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
      export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)"
      export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
      cd "$ROOT/RIFT" || exit 2
      CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run.py \
        --mode eval --output_dir "$OUT" \
        --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
        --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
        --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 --save_bev \
        --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
        --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
        --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \
        --high_value_init_generation --hv_generation_generator_entry "$ENTRY" \
        --hv_generation_checkpoint "$CHECKPOINT" --hv_autoencoder_checkpoint "$AE_CKPT" --hv_ldm_checkpoint "$LDM_CKPT" \
        --hv_generation_mode "$MODE" --hv_num_generate_agents 2 --hv_sd_fov 64 --hv_map_radius 64 \
        --hv_snapshot_map_coordinate_mode "$MAP_COORDINATE_MODE" \
        --hv_generation_max_attempts "$GENERATION_MAX_ATTEMPTS" --hv_generation_rolling \
        --hv_generation_rolling_min_step 50 --hv_generation_rolling_interval_steps 50 \
        --hv_generation_lifetime_radius 45 \
        --hv_generation_existing_rift_positive_threshold 2 \
        --hv_generation_lifecycle_telemetry --no_hv_generation_fail_on_invalid \
        --hv_generation_dump_dir "$DUMP"
    '
}

printf '\n===== no-HV baseline: routes=%s seed=%s =====\n' "$ROUTES" "$SEED"
run_no_hv
if [ "$?" -ne 0 ]; then exit 1; fi

printf '\n===== V2 raw sparse-inpainting =====\n'
run_hv v2 high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator "$V2_BEST" sparse_inpainting legacy "$V2_OUT" "$V2_DUMP"
if [ "$?" -ne 0 ]; then exit 1; fi

printf '\n===== V3 raw sparse-inpainting =====\n'
run_hv v3 high_value_init.scene_generator_v3:ScenarioDreamerHighValueSceneGeneratorV3 "$V3_BEST" sparse_inpainting right_handed "$V3_OUT" "$V3_DUMP"
if [ "$?" -ne 0 ]; then exit 1; fi

printf '\n===== V4-A raw full-FOV forward =====\n'
run_hv v4a high_value_init.scene_generator_v4a:ScenarioDreamerHighValueSceneGeneratorV4A "$V4A_BEST" full_fov_inpainting right_handed "$V4A_OUT" "$V4A_DUMP"
if [ "$?" -ne 0 ]; then exit 1; fi

export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}"
cd "$ROOT/RIFT" || exit 2
"$PY" scripts/summarize_high_value_generation_online.py --dump_dir "$V2_DUMP" --statistics_dir "$ROOT/RIFT/$V2_OUT" --output_json "$V2_DUMP/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi
"$PY" scripts/summarize_high_value_generation_online.py --dump_dir "$V3_DUMP" --statistics_dir "$ROOT/RIFT/$V3_OUT" --output_json "$V3_DUMP/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi
"$PY" scripts/summarize_high_value_generation_online.py --dump_dir "$V4A_DUMP" --statistics_dir "$ROOT/RIFT/$V4A_OUT" --output_json "$V4A_DUMP/summary.json"
if [ "$?" -ne 0 ]; then exit 1; fi
"$PY" scripts/compare_hv_online_runs.py \
  --run no_hv - "$ROOT/RIFT/$NO_HV_OUT" \
  --run v2 "$V2_DUMP" "$ROOT/RIFT/$V2_OUT" \
  --run v3 "$V3_DUMP" "$ROOT/RIFT/$V3_OUT" \
  --run v4a "$V4A_DUMP" "$ROOT/RIFT/$V4A_OUT" \
  --baseline no_hv --output_json "$RUN_DIR/matched_comparison.json" --output_markdown "$RUN_DIR/matched_comparison.md"
if [ "$?" -ne 0 ]; then exit 1; fi

printf '\nCOMPLETE: %s\n' "$RUN_DIR"
