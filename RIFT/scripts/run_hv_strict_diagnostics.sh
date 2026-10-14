#!/usr/bin/env bash
# One initial request per episode, native attention/control unchanged.
set -euo pipefail
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
GPU="${GPU:-0}"
ROUTES="${ROUTES:-28243,24333}"
SEED="${SEED:-0}"
TAG="${TAG:-strict_attention_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-33000}"
BASE_TM_PORT="${BASE_TM_PORT:-53000}"
AE_CKPT="${AE_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt}"
LDM_CKPT="${LDM_CKPT:-$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt}"
V2_CKPT="${V2_CKPT:-$ROOT/data/hv_init/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt}"
V3_CKPT="${V3_CKPT:-$ROOT/data/hv_init/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/best.ckpt}"
ATTN_CKPT="${ATTN_CKPT:-$ROOT/RIFT/rift/ego/model_ckpt/PlanT_state_encoder/checkpoints/state_encoder.ckpt}"
ATTN_HF_DIR="${ATTN_HF_DIR:-$ROOT/RIFT/rift/ego/model_ckpt/PlanT_state_encoder/checkpoints/bert-medium}"
OUT="$ROOT/data/hv_init/eval/$TAG"
test ! -e "$OUT" || { printf 'Output exists: %s\n' "$OUT"; exit 2; }
for path in "$PY" "$AE_CKPT" "$LDM_CKPT" "$V2_CKPT" "$V3_CKPT"; do
  test -f "$path" || { printf 'Missing file: %s\n' "$path"; exit 2; }
done
"$PY" "$ROOT/check_hv_attention_assets.py" --checkpoint "$ATTN_CKPT" --hf-dir "$ATTN_HF_DIR"
export CARLA_ROOT="$ROOT/carla"
CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name 'carla-*.egg' -print -quit)
export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG:${PYTHONPATH:-}"
mkdir -p "$OUT/v2/dump" "$OUT/v3/dump"
RUNNER=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY")
if [ "$(id -u)" -eq 0 ]; then
  chown -R riftuser:riftuser "$OUT"
  RUNNER=(runuser -u riftuser -- env CUDA_VISIBLE_DEVICES="$GPU" CARLA_ROOT="$CARLA_ROOT" PYTHONPATH="$PYTHONPATH" "$PY")
fi
printf 'RESULT_DIR=%s\n' "$OUT"
cd "$ROOT/RIFT"
failed=0
for method in v2 v3; do
  checkpoint="$V2_CKPT"; entry=high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator
  map_mode=legacy
  if [ "$method" = v3 ]; then
    checkpoint="$V3_CKPT"; entry=high_value_init.scene_generator_v3:ScenarioDreamerHighValueSceneGeneratorV3
    map_mode=right_handed
  fi
  set +e
  "${RUNNER[@]}" scripts/run.py --mode eval --output_dir "$OUT/$method/statistics" \
    --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
    --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
    --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --cbv_recog attention \
    --cbv_attention_checkpoint "$ATTN_CKPT" --cbv_attention_hf_dir "$ATTN_HF_DIR" --no_resume -rep 1 \
    --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
    --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \
    --high_value_init_generation --hv_generation_generator_entry "$entry" \
    --hv_generation_checkpoint "$checkpoint" --hv_autoencoder_checkpoint "$AE_CKPT" --hv_ldm_checkpoint "$LDM_CKPT" \
    --hv_generation_mode sparse_inpainting --no_hv_generation_partial_context \
    --hv_num_generate_agents 2 --hv_sd_fov 64 --hv_map_radius 64 \
    --hv_snapshot_map_coordinate_mode "$map_mode" --hv_generation_max_attempts 1 \
    --no_hv_generation_gate_on_existing_rift_positive --no_hv_generation_fail_on_invalid \
    --hv_generation_lifecycle_telemetry --hv_generation_dump_dir "$OUT/$method/dump" \
    2>&1 | tee "$OUT/$method/runner.log"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$OUT/$method/runner_exit_code.txt"
  "$PY" "$ROOT/audit_hv_one_shot_results.py" --dump_dir "$OUT/$method/dump" \
    --simulator-exit-code "$code" --output_json "$OUT/$method/one_shot_audit.json" \
    2>&1 | tee "$OUT/$method/audit.log" || failed=1
  if [ "$code" -ne 0 ]; then failed=1; fi
done
printf 'Upload entire result directory: %s\n' "$OUT"
exit "$failed"
