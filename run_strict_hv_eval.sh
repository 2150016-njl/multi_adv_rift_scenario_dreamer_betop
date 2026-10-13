#!/usr/bin/env bash
# Usage (root): bash run_strict_hv_eval.sh --tag v2_one_shot_1711_s0 --routes 1711 --seed 0 --gpu 0
# Optional: --checkpoint PATH --entry module:Class --generation-mode MODE --map-coordinate-mode MODE
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
HV_ROOT="${HV_ROOT:-$ROOT/data/hv_init}"
AE_CKPT="${AE_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt}"
LDM_CKPT="${LDM_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt}"
CHECKPOINT="$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt"
ENTRY="high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator"
MODE="sparse_inpainting"; MAP_COORDINATE_MODE="legacy"
TAG=""; ROUTES=""; SEED=""; GPU=""; BASE_PORT=30000; BASE_TM_PORT=50000
usage(){ echo "Usage: $0 --tag TAG --routes IDS --seed SEED --gpu GPU [--checkpoint PATH --entry module:Class --generation-mode MODE --map-coordinate-mode MODE --base-port PORT --base-tm-port PORT]"; }
while [ "$#" -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2;;
    --routes) ROUTES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --entry) ENTRY="$2"; shift 2;;
    --generation-mode) MODE="$2"; shift 2;;
    --map-coordinate-mode) MAP_COORDINATE_MODE="$2"; shift 2;;
    --base-port) BASE_PORT="$2"; shift 2;;
    --base-tm-port) BASE_TM_PORT="$2"; shift 2;;
    *) usage; exit 2;;
  esac
done
[ -n "$TAG" ] && [ -n "$ROUTES" ] && [ -n "$SEED" ] && [ -n "$GPU" ] || { usage; exit 2; }
[ "$(id -u)" = 0 ] || { echo "Must run as root (CARLA managed by riftuser)." >&2; exit 2; }
for f in "$PY" "$AE_CKPT" "$LDM_CKPT" "$CHECKPOINT"; do
  [ -e "$f" ] || { echo "Missing: $f" >&2; exit 2; }
done
OUT="strict_hv_${TAG}"
DUMP="$HV_ROOT/eval/strict_hv_${TAG}_dump"
[ ! -e "$ROOT/RIFT/$OUT" ] && [ ! -e "$DUMP" ] || { echo "Refusing to overwrite $TAG" >&2; exit 2; }
mkdir -p "$ROOT/RIFT/$OUT" "$DUMP"
chown -R riftuser:riftuser "$ROOT/RIFT/$OUT" "$DUMP"
# Deliberately NO --hv_generation_rolling, NO --hv_generation_register_cbv_from_audit,
# NO --hv_generation_clamp_generated_states and NO context hiding/traffic replacement.
set +e
runuser -u riftuser -- env ROOT="$ROOT" PY="$PY" AE_CKPT="$AE_CKPT" LDM_CKPT="$LDM_CKPT" CHECKPOINT="$CHECKPOINT" \
  ENTRY="$ENTRY" MODE="$MODE" MAP_COORDINATE_MODE="$MAP_COORDINATE_MODE" \
  OUT="$OUT" DUMP="$DUMP" ROUTES="$ROUTES" SEED="$SEED" GPU="$GPU" BASE_PORT="$BASE_PORT" BASE_TM_PORT="$BASE_TM_PORT" /bin/bash -lc '
  export CARLA_ROOT="$ROOT/carla"
  export CARLA_PYTHONAPI="$CARLA_ROOT/PythonAPI/carla"
  export CARLA_EGG="$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)"
  export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}"
  cd "$ROOT/RIFT" || exit 2
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/run.py \
    --mode eval --output_dir "$OUT" \
    --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
    --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
    --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 \
    --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
    --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \
    --high_value_init_generation --hv_generation_generator_entry "$ENTRY" \
    --hv_generation_checkpoint "$CHECKPOINT" --hv_autoencoder_checkpoint "$AE_CKPT" --hv_ldm_checkpoint "$LDM_CKPT" \
    --hv_generation_mode "$MODE" --hv_num_generate_agents 2 --hv_sd_fov 64 --hv_map_radius 64 \
    --hv_snapshot_map_coordinate_mode "$MAP_COORDINATE_MODE" \
    --hv_generation_actor_model vehicle.tesla.model3 \
    --hv_generation_max_attempts 1 --no_hv_generation_partial_context \
    --no_hv_generation_gate_on_existing_rift_positive \
    --hv_generation_lifecycle_telemetry --no_hv_generation_fail_on_invalid \
    --hv_generation_dump_dir "$DUMP"
'
RUN_CODE=$?
set -e
set +e
"$PY" "$ROOT/audit_hv_one_shot_results.py" --dump_dir "$DUMP" \
  --simulator-exit-code "$RUN_CODE" \
  --output_json "$DUMP/strict_audit_summary.json"
AUDIT_CODE=$?
set -e
echo "Simulator exit code: $RUN_CODE"
echo "Strict audit exit code: $AUDIT_CODE"
echo "Strict evaluation dump: $DUMP"
# Preserve the simulator failure as the primary error.  Never report an
# incomplete run's zero green count as an online model failure.
if [ "$RUN_CODE" -ne 0 ]; then exit "$RUN_CODE"; fi
exit "$AUDIT_CODE"
