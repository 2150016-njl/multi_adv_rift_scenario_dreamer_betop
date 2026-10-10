export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT=$ROOT/carla
export CARLA_PYTHONAPI=$ROOT/carla/PythonAPI/carla
export CARLA_EGG=$(find "$ROOT/carla/PythonAPI/carla/dist" -maxdepth 1 -name "carla-*.egg" | head -n 1)
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}

export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PHASE5_CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt

export TAG=phase7_v2_uncovered_routes_seed0_7
export COLLECT_ROOT=$HV_ROOT/rift_labeled_$TAG
export UNION_DIR=$HV_ROOT/rift_labeled_union_phase5_$TAG
export DATASET_DIR=$HV_ROOT/datasets/inpainting_sparse_$TAG
export FT_CKPT_DIR=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_${TAG}_ft_from_phase5
export SCRATCH_CKPT_DIR=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_${TAG}_scratch
export EVAL_DIR=$HV_ROOT/eval/$TAG

mkdir -p "$EVAL_DIR"

cd $ROOT/scenario-dreamer

$PY -m py_compile \
  build_high_value_scene_dataset_v2.py \
  train_high_value_scene_generator_v2.py \
  eval_high_value_scene_generator_v2.py \
  audit_high_value_scene_dataset.py || exit 1

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
if len(files) == 0:
    raise RuntimeError("No collected json files found. Do not continue.")
PY
if [ $? -ne 0 ]; then exit 1; fi

rm -rf "$UNION_DIR" || exit 1
mkdir -p "$UNION_DIR"

i=0
find -L "$OLD_COMBINED" "$COLLECT_ROOT" -type f -name "*.json" -print0 | while IFS= read -r -d '' f; do
  i=$((i+1))
  ln -sf "$f" "$UNION_DIR/$(printf "%09d" "$i")_$(basename "$f")"
done

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
