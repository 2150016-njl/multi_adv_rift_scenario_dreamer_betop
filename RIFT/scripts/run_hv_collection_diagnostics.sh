#!/usr/bin/env bash
# Small coverage campaign: every Town, two real traffic densities, frame 0+later.
set -euo pipefail
ROOT="${ROOT:-/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop}"
PY="${PY:-/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
TAG="${TAG:-coverage_$(date +%Y%m%d_%H%M%S)}"
BASE_PORT="${BASE_PORT:-35000}"
BASE_TM_PORT="${BASE_TM_PORT:-55000}"
OUT="$ROOT/data/hv_init/$TAG"
test ! -e "$OUT" || { printf 'Output exists: %s\n' "$OUT"; exit 2; }
# Route CLI accepts XML route IDs, not dataset-local route_index values.
ROUTES="${ROUTES:-$("$PY" -c 'from xml.etree import ElementTree as E; import sys; d={}; [d.setdefault(r.get("town"),r.get("id")) for r in E.parse(sys.argv[1]).getroot().findall("route")]; print(",".join(d.values()))' "$ROOT/RIFT/rift/scenario/route/bench2drive220.xml")}"
export CARLA_ROOT="$ROOT/carla"
CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name 'carla-*.egg' -print -quit)
export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI/carla:$CARLA_EGG:${PYTHONPATH:-}"
mkdir -p "$OUT/sparse/raw" "$OUT/normal/raw"
RUNNER=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY")
if [ "$(id -u)" -eq 0 ]; then
  chown -R riftuser:riftuser "$OUT"
  RUNNER=(runuser -u riftuser -- env CUDA_VISIBLE_DEVICES="$GPU" CARLA_ROOT="$CARLA_ROOT" PYTHONPATH="$PYTHONPATH" "$PY")
fi
printf 'RESULT_DIR=%s\nROUTES=%s\nSEED=%s\n' "$OUT" "$ROUTES" "$SEED" | tee "$OUT/collection_plan.txt"
cd "$ROOT/RIFT"
failed=0
for density in sparse normal; do
  intensity=0.2
  if [ "$density" = normal ]; then intensity=0.6; fi
  set +e
  "${RUNNER[@]}" scripts/run.py --mode collect_hv_generation --output_dir "$OUT/$density/statistics" \
    --base_port "$BASE_PORT" --base_tm_port "$BASE_TM_PORT" \
    --routes rift/scenario/route/bench2drive220.xml --routes_subset "$ROUTES" --seed "$SEED" \
    --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --cbv_recog rule --no_resume -rep 1 \
    --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
    --rift_speed_limits_dir "$ROOT/RIFT/data/speed_limits/Speed-Limits" \
    --hv_snapshot_map_coordinate_mode right_handed --hv_snapshot_native_map_export \
    --hv_collection_traffic_intensity "$intensity" --hv_generation_collect_min_forward_rift_positive 0 \
    --hv_generation_sample_interval_s 1.0 --hv_generation_max_frames_per_episode 40 \
    --hv_generation_dump_dir "$OUT/$density/raw" \
    2>&1 | tee "$OUT/$density/collection.log"
  code=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$code" > "$OUT/$density/runner_exit_code.txt"
  if [ "$code" -ne 0 ]; then failed=1; fi
done
printf 'Coverage collection output: %s\n' "$OUT"
exit "$failed"
