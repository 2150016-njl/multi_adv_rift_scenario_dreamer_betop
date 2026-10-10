

export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT=$ROOT/carla
export CARLA_PYTHONAPI=$ROOT/carla/PythonAPI/carla
export CARLA_EGG=$(find "$ROOT/carla/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}

if [ ! -x "$CARLA_ROOT/CarlaUE4.sh" ]; then
  echo "ERROR: CARLA_ROOT is invalid or CarlaUE4.sh is not executable: $CARLA_ROOT" >&2
  exit 1
fi

export CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt

export ROUTES_XML=rift/scenario/route/bench2drive220.xml
export ROUTES_XML_ABS=$ROOT/RIFT/$ROUTES_XML
export ROUTES_SUBSET=$(python - <<'PY'
import os
import xml.etree.ElementTree as ET

xml_path = os.environ["ROUTES_XML_ABS"]
ids = [route.attrib["id"] for route in ET.parse(xml_path).iter("route")]
if not ids:
    raise RuntimeError(f"No route ids parsed from {xml_path}")
print(",".join(ids))
PY
)
export SCENE_TAG=b2d220_all_xml_routes_seed0_rep1_video_phase5best_rawstate_fov64_riftgate
export REPLENISH_OUT=hvgen_replenish_life45_${SCENE_TAG}
export REPLENISH_DUMP=$HV_ROOT/online_eval/${REPLENISH_OUT}

mkdir -p "$ROOT/RIFT/$REPLENISH_OUT" "$REPLENISH_DUMP"
chown -R riftuser:riftuser "$ROOT/RIFT/$REPLENISH_OUT" "$REPLENISH_DUMP" || exit 1

cd $ROOT/RIFT

$PY -m py_compile \
  scripts/run.py \
  scripts/summarize_high_value_generation_online.py \
  rift/carla_runner.py \
  rift/cbv/initialization/hv_online_generation.py \
  rift/gym_carla/envs/carla_env.py \
  rift/gym_carla/envs/env_wrapper.py \
  rift/gym_carla/visualization/nuplan_scenario_render.py \
  rift/gym_carla/visualization/bev_render.py || exit 1

runuser -u riftuser -- bash -lc "
export ROOT=$ROOT
export HV_ROOT=$HV_ROOT
export PY=$PY
export CARLA_ROOT=$CARLA_ROOT
export CARLA_PYTHONAPI=$CARLA_PYTHONAPI
export CARLA_EGG=$CARLA_EGG
export PYTHONPATH=$PYTHONPATH
export CKPT=$CKPT
export AE_CKPT=$AE_CKPT
export LDM_CKPT=$LDM_CKPT
export ROUTES_XML=$ROUTES_XML
export ROUTES_SUBSET=$ROUTES_SUBSET
export REPLENISH_OUT=$REPLENISH_OUT
export REPLENISH_DUMP=$REPLENISH_DUMP

cd \$ROOT/RIFT

CUDA_VISIBLE_DEVICES=2 \$PY scripts/run.py \
  --mode eval \
  --output_dir \$REPLENISH_OUT \
  --routes \$ROUTES_XML \
  --routes_subset \$ROUTES_SUBSET \
  --seed 0 \
  --base_port 30600 \
  --base_tm_port 50600 \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume \
  -rep 1 \
  --save_bev \
  --video_render_bounds 32 \
  --video_render_map_radius 64 \
  --video_render_offset 0 \
  --rift_hd_map_dir \$ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir \$ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator \
  --hv_generation_checkpoint \$CKPT \
  --hv_autoencoder_checkpoint \$AE_CKPT \
  --hv_ldm_checkpoint \$LDM_CKPT \
  --hv_generation_mode sparse_inpainting \
  --hv_num_generate_agents 2 \
  --hv_generation_max_attempts 16 \
  --hv_generation_rolling \
  --hv_generation_rolling_min_step 50 \
  --hv_generation_rolling_interval_steps 50 \
  --hv_generation_lifetime_radius 45 \
  --hv_generation_register_cbv \
  --hv_generation_register_cbv_count 2 \
  --no_hv_generation_fail_on_invalid \
  --hv_generation_dump_dir \$REPLENISH_DUMP || exit 1
" || exit 1

$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir $REPLENISH_DUMP \
  --statistics_dir $ROOT/RIFT/$REPLENISH_OUT \
  --output_json $REPLENISH_DUMP/summary.json || exit 1

cat $REPLENISH_DUMP/summary.json

find $ROOT/RIFT/$REPLENISH_OUT \
  -path "*videos*route_*.mp4" -type f -print

python - <<'PY'
import json, glob, os

for name in ["REPLENISH_DUMP"]:
    dump_dir = os.environ[name]
    print("\n==", name, dump_dir)
    for p in sorted(glob.glob(dump_dir + "/*online_generation.json"))[:120]:
        j = json.load(open(p))
        r = j["result"]
        print(
            os.path.basename(p),
            "frame", r.get("frame_id"),
            "rolling", r.get("rolling_generation"),
            "spawned", r.get("spawned_actor_ids"),
            "registered", r.get("cbv_registered_actor_ids"),
            "rift_generated_cbv_count", r.get("num_generated_rift_positive_agents"),
            "rift_generated_cbv_bucket", r.get("generated_rift_cbv_count_bucket"),
        )
PY
