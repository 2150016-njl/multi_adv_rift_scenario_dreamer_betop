# Phase7 v2 Reproduction Record

## Scope and conventions

This document records the commands, assets, results, and known failures from the Phase7 v2 experiment round. It is a reproduction record, not a claim that every command below should be re-run immediately.

The online objective is to generate anonymous surrounding-vehicle initial states in the running CARLA scene. During closed-loop execution, a generated orange vehicle remains Traffic-Manager controlled until normal RIFT recognition may activate it as a green RIFT-CBV controlled by Pluto. RIFT labels are not an inference input and post-spawn audit does not delete or modify generated states.

All paths below are server paths.

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT=$ROOT/carla
export CARLA_PYTHONAPI=$CARLA_ROOT/PythonAPI/carla
export CARLA_EGG=$(find "$CARLA_ROOT/dist" "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name 'carla-*.egg' 2>/dev/null | head -n 1)
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}
```

The valid environment check is:

```bash
cd $ROOT/RIFT
$PY - <<'PY'
from agents.navigation.local_planner import RoadOption
from rift.carla_runner import CarlaRunner
import high_value_init
print('RIFT and high_value_init import ok')
PY
```

## 1. Data collection

### 1.1 Completed collection

The Phase7 v2 collection directory is:

```bash
export TAG=phase7_v2_uncovered_routes_seed0_7
export COLLECT_ROOT=$HV_ROOT/rift_labeled_$TAG
```

It collected 54,936 JSON snapshots from 220 routes using eight workers, GPUs 0-3, and seeds 0-7. Workers each received a disjoint route-id list. The route lists were built from `bench2drive220.xml`, prioritizing routes absent from the old recursive collection, then low-count old routes.

The original complete live-output launcher is [`xunlian.sh`](xunlian.sh). It prefixes each worker's terminal stream and writes a matching per-worker log. Its intended invocation is:

```bash
cd $ROOT
bash xunlian.sh 2>&1 | tee xunlian_$(date +%Y%m%d_%H%M%S).log
```

Do not re-run the full script merely to train: it includes collection, union construction, dataset construction, training, and evaluation.

Collection integrity check:

```bash
python - <<'PY'
import glob, json, os, re
from collections import Counter

root=os.environ['COLLECT_ROOT']
files=glob.glob(root+'/**/*.json', recursive=True)
routes=Counter()
for path in files:
    try:
        scene=json.load(open(path)).get('scene', {})
        text=str(scene.get('scene_id',''))+' '+str(scene.get('episode_key',''))
        match=re.search(r'RouteScenario_(\d+)', text)
        if match:
            routes[match.group(1)] += 1
    except Exception:
        pass
print('num_files', len(files))
print('num_routes', len(routes))
print('routes_with_lt10_snapshots', sum(v < 10 for v in routes.values()))
print('top20', routes.most_common(20))
PY
```

Observed result: `num_files=54936`, `num_routes=220`, and no route had fewer than ten snapshots.

### 1.2 Merge without basename collisions

The historical recursive collection contains symbolic links. Use `find -L` whenever counting or merging it.

```bash
export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
find -L "$OLD_COMBINED" -type f -name '*.json' | wc -l
```

The v2 dataset was built from a union containing both old and new data. A flat symlink directory is acceptable only when every link name is prefixed by a unique sequence number; never link raw basenames directly because same-name snapshots overwrite one another.

```bash
export UNION_DIR=$HV_ROOT/rift_labeled_union_phase5_phase7_v2_uncovered_routes_seed0_7
mkdir -p "$UNION_DIR"

i=0
find -L "$OLD_COMBINED" "$COLLECT_ROOT" -type f -name '*.json' -print0 |
while IFS= read -r -d '' file; do
  i=$((i + 1))
  ln -sfn "$file" "$UNION_DIR/$(printf '%09d' "$i")_$(basename "$file")"
done

find -L "$UNION_DIR" -type f -name '*.json' | wc -l
```

## 2. Dataset construction and audit

The final v2 JSONL directory used by the completed B32 runs is:

```bash
export DATASET_DIR=$HV_ROOT/datasets/inpainting_sparse_phase7_v2_uncovered_routes_seed0_7
```

Construction command:

```bash
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
  --max_target_speed 25 \
  --min_target_ego_distance 3 \
  --max_target_ego_distance 60 \
  --target_fov_margin 2 \
  --min_target_pair_distance 2 \
  --dedup_grid_xy 1

$PY audit_high_value_scene_dataset.py \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --test_jsonl "$DATASET_DIR/test.jsonl" \
  --expected_mode inpainting \
  --expected_num_generate_agents 2 \
  --fov 64 \
  --output_json "$DATASET_DIR/audit.json"
```

Verified final record counts:

| split | records |
|---|---:|
| train | 58,168 |
| val | 6,431 |
| test | dataset-dependent; use `wc -l` to verify before evaluation |

The `full` name of the B32 checkpoints refers to the full v2 train JSONL above. It does not mean every historical dataset on disk was automatically included.

## 3. Model training

Common assets:

```bash
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PHASE5_CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt
```

### 3.1 B32 scratch training

```bash
export SCRATCH_B32=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_scratch_20260922

cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=1 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$SCRATCH_B32" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 \
  --epochs 60 \
  --batch_size 32 \
  --lr 5e-5 \
  --seed 0 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

Observed result: best epoch 12/60, `val_decoded_position_l1=0.1499497757`; final epoch degraded to `0.1617999601`. Use `best.ckpt`, not `last.ckpt`.

### 3.2 B32 fine-tuning from Phase5

```bash
export FT_B32=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922

cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=3 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$FT_B32" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$PHASE5_CKPT" \
  --num_generate_agents 2 \
  --epochs 40 \
  --batch_size 32 \
  --lr 2e-5 \
  --seed 0 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

Observed result: best epoch 12/40, `val_decoded_position_l1=0.1463771265`, `val_decoded_state_loss=0.5635170419`; final epoch degraded to `0.1613097740`. This is the current offline-best checkpoint:

```bash
export CURRENT_CKPT=$FT_B32/best.ckpt
```

## 4. Offline generator evaluation

The v2 evaluator reports reconstruction errors and raw-state feasibility diagnostics. It does not prove closed-loop RIFT-CBV quality.

```bash
export EVAL_DIR=$HV_ROOT/eval/phase7_v2_full_b32_ft_from_phase5
mkdir -p "$EVAL_DIR"

cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting \
  --jsonl "$DATASET_DIR/test.jsonl" \
  --checkpoint "$CURRENT_CKPT" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --output_json "$EVAL_DIR/test1000_bestof4_raw.json" \
  --output_generated_jsonl "$EVAL_DIR/test1000_bestof4_raw_generated.jsonl" \
  --num_generate_agents 2 \
  --num_samples_per_condition 4 \
  --max_records 1000 \
  --fov 64 \
  --boundary_margin 2 \
  --min_pair_distance 2
```

Use raw evaluation first. `--clamp_generated_states` is an ablation only; it must not be treated as model quality because it changes the model output after generation.

## 5. Online closed-loop experiments

### 5.1 Runtime invariants

- `sparse_inpainting` uses ego, route, map, and visible actors in the 64m square FOV.
- Model output remains raw `x/y/yaw/speed`; online code queries road height only and does not snap position or heading to a lane.
- `--hv_generation_existing_rift_positive_threshold 2` suppresses generation only when the current scene already has at least two RIFT-positive actors.
- Do not use `--hv_generation_rolling_force_update` for the normal replenish setting.
- `--hv_generation_register_cbv` enables normal RIFT recognition flow. Post-spawn audit is observational and does not immediately register a generated actor unless the separate legacy audit flag is deliberately enabled.
- Saved BEV videos are cropped to the ego-local 64m square: bounds 32m, zero offset.

### 5.2 Successful cross-Town debug run

The successful result directory is:

```bash
export OUT=hvgen_debug_cross_town_fov64_ftb32_seed0_rerun1
export DUMP=$HV_ROOT/online_debug/$OUT
```

Routes: `24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333` (one route each from Town01/02/03/04/05/06/07/10HD/11/12/13/15).

Reproducible foreground command:

```bash
mkdir -p "$DUMP" "$ROOT/RIFT/$OUT"
chown -R riftuser:riftuser "$DUMP" "$ROOT/RIFT/$OUT"

cd $ROOT/RIFT
runuser -u riftuser -- bash -lc "
export ROOT=$ROOT
export PY=$PY
export CARLA_ROOT=\$ROOT/carla
export CARLA_PYTHONAPI=\$ROOT/carla/PythonAPI/carla
export CARLA_EGG=\$(find \$ROOT/carla/PythonAPI/carla/dist -maxdepth 1 -name 'carla-*.egg' | head -n 1)
export PYTHONPATH=\$ROOT/RIFT:\$ROOT/scenario-dreamer:\$CARLA_PYTHONAPI:\$CARLA_EGG:\${PYTHONPATH:-}
cd \$ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 \$PY scripts/run.py \
  --mode eval \
  --output_dir $OUT \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset 24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333 \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume -rep 1 --save_bev \
  --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
  --rift_hd_map_dir \$ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir \$ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator \
  --hv_generation_checkpoint $CURRENT_CKPT \
  --hv_autoencoder_checkpoint $AE_CKPT \
  --hv_ldm_checkpoint $LDM_CKPT \
  --hv_generation_mode sparse_inpainting \
  --hv_num_generate_agents 2 \
  --hv_generation_max_attempts 16 \
  --hv_generation_rolling \
  --hv_generation_rolling_min_step 50 \
  --hv_generation_rolling_interval_steps 50 \
  --hv_generation_existing_rift_positive_threshold 2 \
  --hv_generation_register_cbv \
  --no_hv_generation_fail_on_invalid \
  --hv_generation_dump_dir $DUMP
"

runuser -u riftuser -- bash -lc "
cd $ROOT/RIFT
$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir $DUMP \
  --statistics_dir $ROOT/RIFT/$OUT \
  --output_json $DUMP/summary.json
"
```

Observed result:

| metric | value |
|---|---:|
| completed routes | 12 |
| ego mean driving score | 86.2505 |
| ego mean route completion | 93.7508 |
| collision-route ratio | 0.25 |
| online dumps | 143 |
| model calls | 19 |
| model calls that spawned actors | 17 / 19 = 89.5% |
| generated 2 RIFT-positive actors | 1 / 19 = 5.3% |
| generated 1 RIFT-positive actor | 8 / 19 = 42.1% |
| generated 0 RIFT-positive actors | 10 / 19 = 52.6% |

The summary's `generation_success_rate=17/143` uses all checks as denominator. For model quality, use the conditional denominator `generator_called_count=19`.

### 5.3 Known online issues from this run

1. Rolling inspection occurred repeatedly on adjacent frames after frame 50. This is a cadence bug: skipped checks do not update `_last_rolling_step`. It inflates dumps and runtime but does not mean the model was invoked 143 times.
2. Two generation calls ended in CARLA spawn failure. Raw output was not snapped to a lane; this is a model feasibility failure that must remain visible in diagnostics.
3. `cbv_registration_count=0` means no immediate audit-driven registration. It is not a measure of later normal RIFT recognition. Lifecycle telemetry is needed to measure orange-to-green conversion.
4. Do not interpret the ego driving score as a direct model score. A matched no-HV baseline on the same routes and seed is needed for causal comparison.

## 6. Reproduction safeguards

- Run CARLA through `runuser -u riftuser`; CARLA refuses root execution.
- Pre-create and chown both `--output_dir` and `--hv_generation_dump_dir`. A root-owned `online_debug` parent caused a run to stop before route completion.
- Do not run the summary command when `simulation_results.json` still has `entry_status=Started` or zero records.
- Do not use checkpoint `last.ckpt` when `history.json` shows validation degradation after the best epoch.
- Record each run in a new `OUT` directory; never overwrite an unfinished output directory.
