

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

export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
export ROUTES_XML_ABS=$ROOT/RIFT/rift/scenario/route/bench2drive220.xml
export ROUTES_XML_REL=rift/scenario/route/bench2drive220.xml

export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PHASE5_CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt

export TAG=phase7_v2_uncovered_routes_seed0_7
export COLLECT_ROOT=$HV_ROOT/rift_labeled_$TAG
export ROUTE_SPLIT_DIR=$HV_ROOT/route_splits_$TAG
export WORKER_STATUS_DIR=$HV_ROOT/worker_status_$TAG
export UNION_DIR=$HV_ROOT/rift_labeled_union_phase5_$TAG
export DATASET_DIR=$HV_ROOT/datasets/inpainting_sparse_$TAG
export FT_CKPT_DIR=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_${TAG}_ft_from_phase5
export SCRATCH_CKPT_DIR=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_${TAG}_scratch
export EVAL_DIR=$HV_ROOT/eval/$TAG

mkdir -p "$COLLECT_ROOT" "$ROUTE_SPLIT_DIR" "$WORKER_STATUS_DIR" "$EVAL_DIR"
chown -R riftuser:riftuser "$COLLECT_ROOT" "$WORKER_STATUS_DIR" || exit 1

cd $ROOT/scenario-dreamer

$PY -m py_compile \
  build_high_value_scene_dataset_v2.py \
  train_high_value_scene_generator_v2.py \
  eval_high_value_scene_generator_v2.py \
  audit_high_value_scene_dataset.py || exit 1

python - <<'PY'
import json, glob, os, re, xml.etree.ElementTree as ET
from collections import Counter

old_root=os.environ["OLD_COMBINED"]
xml_path=os.environ["ROUTES_XML_ABS"]
out_dir=os.environ["ROUTE_SPLIT_DIR"]
os.makedirs(out_dir, exist_ok=True)

old_routes=Counter()
for p in glob.glob(old_root+"/**/*.json", recursive=True):
    try:
        j=json.load(open(p))
        scene=j.get("scene", j)
        sid=str(scene.get("scene_id",""))
        ep=str(scene.get("episode_key",""))
        m=re.search(r"RouteScenario_(\d+)", sid or ep)
        if m:
            old_routes[m.group(1)] += 1
    except Exception:
        pass

tree=ET.parse(xml_path)
xml_routes=[]
for elem in tree.iter():
    if elem.tag == "route":
        rid = elem.attrib.get("id") or elem.attrib.get("name")
        if rid is not None:
            xml_routes.append(str(rid))

if not xml_routes:
    raise RuntimeError("No route ids parsed from routes XML")

uncovered=[rid for rid in xml_routes if rid not in old_routes]
covered_low=sorted(
    [rid for rid in xml_routes if rid in old_routes],
    key=lambda rid: old_routes[rid],
)

selected=uncovered + covered_low
selected=list(dict.fromkeys(selected))

print("old_num_routes", len(old_routes))
print("xml_num_routes", len(xml_routes))
print("uncovered_num_routes", len(uncovered))
print("selected_num_routes", len(selected))
print("first_selected", selected[:50])

num_workers=8
chunks=[[] for _ in range(num_workers)]
for i, rid in enumerate(selected):
    chunks[i % num_workers].append(rid)

for idx, chunk in enumerate(chunks):
    path=os.path.join(out_dir, f"worker{idx}.txt")
    with open(path, "w") as f:
        f.write(",".join(chunk))
    print(path, len(chunk), chunk[:10])
PY
if [ $? -ne 0 ]; then exit 1; fi
chmod -R a+rX "$ROUTE_SPLIT_DIR" || exit 1

run_collect_worker_loop () {
  local WORKER_ID=$1
  local GPU_ID=$2
  local BASE_PORT=$3
  local TM_PORT=$4
  local ROUTE_FILE=$ROUTE_SPLIT_DIR/worker${WORKER_ID}.txt
  local ROUTES_SUBSET
  ROUTES_SUBSET=$(cat "$ROUTE_FILE")

  runuser -u riftuser -- bash -lc "
export ROOT=$ROOT
export PY=$PY
export CARLA_ROOT=$CARLA_ROOT
export CARLA_PYTHONAPI=$CARLA_PYTHONAPI
export CARLA_EGG=$CARLA_EGG
export PYTHONPATH=$PYTHONPATH
export PYTHONUNBUFFERED=1
cd \$ROOT/RIFT

for SEED in 0 1 2 3 4 5 6 7; do
  OUT_DIR=$COLLECT_ROOT/worker${WORKER_ID}_seed\${SEED}
  PORT=\$(( $BASE_PORT + SEED * 20 ))
  TMPORT=\$(( $TM_PORT + SEED * 20 ))

  echo \"[worker $WORKER_ID] seed=\${SEED} gpu=$GPU_ID port=\${PORT} tm=\${TMPORT} routes_file=$ROUTE_FILE out=\${OUT_DIR}\"

  CUDA_VISIBLE_DEVICES=$GPU_ID \$PY scripts/run.py \
    --mode collect_hv_generation \
    --routes $ROUTES_XML_REL \
    --routes_subset $ROUTES_SUBSET \
    --seed \${SEED} \
    --base_port \${PORT} \
    --base_tm_port \${TMPORT} \
    --ego_cfg pdm_lite.yaml \
    --cbv_cfg rift_pluto.yaml \
    --no_resume \
    -rep 1 \
    --rift_hd_map_dir \$ROOT/RIFT/data/map_data/HD-Map \
    --rift_speed_limits_dir \$ROOT/RIFT/data/speed_limits/Speed-Limits \
    --hv_generation_dump_dir \${OUT_DIR} \
    --hv_generation_sample_interval_s 0.5 \
    --hv_generation_max_frames_per_episode 300 \
    --hv_generation_max_snapshots_per_episode 1000000 || exit 1
done
"
}

echo "============================================================"
echo "Starting parallel data collection. Output is streamed live below."
echo "Logs are also saved under: $COLLECT_ROOT"
echo "============================================================"

COLLECT_PIDS=()
rm -f "$WORKER_STATUS_DIR"/worker*.status
for WORKER_ID in 0 1 2 3 4 5 6 7; do
  GPU_ID=$((WORKER_ID / 2))
  BASE_PORT=$((32000 + WORKER_ID * 200))
  TM_PORT=$((52000 + WORKER_ID * 200))
  LOG_FILE="$COLLECT_ROOT/worker${WORKER_ID}_gpu${GPU_ID}.log"

  echo "[launcher] worker=$WORKER_ID gpu=$GPU_ID base_port=$BASE_PORT tm_port=$TM_PORT log=$LOG_FILE"

  # Keep workers parallel, but do NOT hide their output in log files.
  {
    run_collect_worker_loop "$WORKER_ID" "$GPU_ID" "$BASE_PORT" "$TM_PORT" 2>&1
    STATUS=$?
    echo "$STATUS" > "$WORKER_STATUS_DIR/worker${WORKER_ID}.status"
    exit "$STATUS"
  } \
    | sed -u "s/^/[worker${WORKER_ID} gpu${GPU_ID}] /" \
    | tee "$LOG_FILE" &
  COLLECT_PIDS+=("$!")
done

COLLECT_FAILED=0
for PID in "${COLLECT_PIDS[@]}"; do
  if ! wait "$PID"; then
    echo "[launcher] WARNING: collection pipeline pid=$PID exited with non-zero status" >&2
    COLLECT_FAILED=1
  fi
done

for WORKER_ID in 0 1 2 3 4 5 6 7; do
  STATUS_FILE="$WORKER_STATUS_DIR/worker${WORKER_ID}.status"
  if [ ! -f "$STATUS_FILE" ]; then
    echo "[launcher] ERROR: missing worker status file: $STATUS_FILE" >&2
    COLLECT_FAILED=1
  elif [ "$(cat "$STATUS_FILE")" != "0" ]; then
    echo "[launcher] ERROR: worker $WORKER_ID failed with status $(cat "$STATUS_FILE")" >&2
    COLLECT_FAILED=1
  fi
done

echo "============================================================"
if [ "$COLLECT_FAILED" -eq 0 ]; then
  echo "All collection workers finished."
else
  echo "Collection finished, but at least one worker reported an error. Check the terminal output/logs above." >&2
  exit 1
fi
echo "============================================================"

python - <<'PY'
import json, glob, os, re
from collections import Counter

root=os.environ["COLLECT_ROOT"]
files=glob.glob(root+"/**/*.json", recursive=True)
routes=Counter()
for p in files:
    try:
        j=json.load(open(p))
        scene=j.get("scene", j)
        sid=str(scene.get("scene_id",""))
        ep=str(scene.get("episode_key",""))
        m=re.search(r"RouteScenario_(\d+)", sid or ep)
        if m:
            routes[m.group(1)] += 1
    except Exception:
        pass

print("collect_root", root)
print("num_files", len(files))
print("num_routes", len(routes))
print("top20", routes.most_common(20))
print("routes_with_lt10_snapshots", sum(1 for _, c in routes.items() if c < 10))
PY
if [ $? -ne 0 ]; then exit 1; fi

rm -rf "$UNION_DIR" || exit 1
mkdir -p "$UNION_DIR"

i=0
find -L "$OLD_COMBINED" "$COLLECT_ROOT" -type f -name "*.json" -print0 | while IFS= read -r -d '' f; do
  i=$((i+1))
  ln -sf "$f" "$UNION_DIR/$(printf "%09d" "$i")_$(basename "$f")"
done

cd $ROOT/scenario-dreamer

$PY build_high_value_scene_dataset_v2.py \
  --input_dir "$UNION_DIR" \
  --output_dir "$DATASET_DIR" \
  --mode inpainting \
  --min_rift_positive_agents 2 \
  --num_generate_agents 2 \
  --actor_radius 80 \
  --fov 64 \
  --positive_target_policy route_s_diversity \
  --sparse_context_copies 4 \
  --sparse_context_min_observed 0 \
  --sparse_context_max_observed 4 \
  --val_ratio 0.10 \
  --test_ratio 0.10 \
  --seed 42 \
  --source real_positive_phase7_v2_uncovered_multiseed \
  --quality_filter_v2 \
  --min_target_speed 0.5 \
  --max_target_speed 25.0 \
  --min_target_ego_distance 3.0 \
  --max_target_ego_distance 60.0 \
  --target_fov_margin 2.0 \
  --min_target_pair_distance 2.0 \
  --dedup_grid_xy 1.0 || exit 1

$PY audit_high_value_scene_dataset.py \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --test_jsonl "$DATASET_DIR/test.jsonl" \
  --expected_mode inpainting \
  --expected_num_generate_agents 2 \
  --fov 64 \
  --output_json "$DATASET_DIR/audit.json" || exit 1

cat "$DATASET_DIR/quality_report.json"
cat "$DATASET_DIR/audit.json"

CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$FT_CKPT_DIR" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$PHASE5_CKPT" \
  --num_generate_agents 2 \
  --epochs 30 \
  --batch_size 16 \
  --lr 2e-5 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1 || exit 1

CUDA_VISIBLE_DEVICES=1 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$SCRATCH_CKPT_DIR" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 \
  --epochs 50 \
  --batch_size 16 \
  --lr 5e-5 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1 || exit 1

for NAME in phase5 ft scratch; do
  if [ "$NAME" = "phase5" ]; then CKPT="$PHASE5_CKPT"; GPU=0; fi
  if [ "$NAME" = "ft" ]; then CKPT="$FT_CKPT_DIR/best.ckpt"; GPU=1; fi
  if [ "$NAME" = "scratch" ]; then CKPT="$SCRATCH_CKPT_DIR/best.ckpt"; GPU=2; fi

  CUDA_VISIBLE_DEVICES=$GPU $PY eval_high_value_scene_generator_v2.py \
    --mode inpainting \
    --jsonl "$DATASET_DIR/test.jsonl" \
    --checkpoint "$CKPT" \
    --autoencoder_checkpoint "$AE_CKPT" \
    --ldm_checkpoint "$LDM_CKPT" \
    --output_json "$EVAL_DIR/${NAME}_test1000_clamp_bestof4.json" \
    --output_generated_jsonl "$EVAL_DIR/${NAME}_test1000_clamp_bestof4_generated.jsonl" \
    --num_generate_agents 2 \
    --num_samples_per_condition 4 \
    --max_records 1000 \
    --clamp_generated_states || exit 1
done

cat "$EVAL_DIR/phase5_test1000_clamp_bestof4.json"
cat "$EVAL_DIR/ft_test1000_clamp_bestof4.json"
cat "$EVAL_DIR/scratch_test1000_clamp_bestof4.json"
