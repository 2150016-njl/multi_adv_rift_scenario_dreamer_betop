# V3 Technical Route: Closed-Loop High-Value State Generation

## 1. Exact objective

At a running CARLA/RIFT scene state, generate two anonymous surrounding-vehicle initial states whose raw position, heading, speed, and size are consistent with the local map, road topology, ego route, and traffic context. The generated vehicles then run in the closed loop. A high-value result is measured by the generated actors forming RIFT-CBV interactions during the rollout.

The generator does not select an existing actor, does not consume RIFT labels at inference, and does not receive an online RIFT-audit correction. Traffic Manager initially controls an orange generated vehicle. Normal RIFT recognition may later activate it as a green Pluto-CBV; that transition is a closed-loop outcome, not a post-processing operation on the model state.

## 2. Non-negotiable boundaries

- Keep model output raw: no lane-center snap, heading replacement, speed replacement, or state clamp in the main experiment.
- Keep height-only placement because the state schema has no z coordinate. The nearest driving waypoint provides terrain z only; it must not change x/y/yaw.
- RIFT is allowed for offline data labeling, the existing `>=2` generation gate, and evaluation. It is not a generator input, online rejection oracle, or state editor.
- Do not use force-update in normal replenish experiments.
- Preserve Phase5 and v2 scripts. New work must use v3 modules or explicit v3 switches.
- Separate model metrics from ego route scores. A high ego score alone is not generator success.

## 3. Current evidence and diagnosis

The B32 fine-tuned v2 checkpoint obtains `val_decoded_position_l1=0.14638`, better than the scratch B32 checkpoint at `0.14995`. The 12-Town closed-loop run nevertheless produced two RIFT-positive generated actors only once in 19 model calls.

This identifies a transfer gap: reconstruction of RIFT-labeled target states is better than closed-loop generation of map-consistent high-value states. The evidence does not support adding online lane snapping or audit-based deletion. It supports improving the model's map representation, data coverage, and raw-state feasibility supervision.

The present condition path is especially weak for topology:

1. Snapshots already carry `layer`, `road_id`, `lane_id`, `lane_type`, `topology_type`, topology edges, and centerlines.
2. `scene_generator.py` forwards only lane polylines.
3. `rift_dataset.py` and `scenario_dreamer_adapter.py` encode every retained lane as generic `lane`.

Thus lane versus connector, direction, lane identity, and graph topology are dropped before the LDM receives the condition.

## 4. V3 implementation stages

### Current implementation status

The first executable v3 ablation is implemented in separate files:

- `build_high_value_scene_dataset_v3.py`: uses snapshot lane metadata when valid or explicitly enriches old snapshots from RIFT's HD map, associates each target with its nearest source lane centerline, rejects only records exceeding the configured association distance, and reports Town/topology counts.
- `high_value_init/high_value_ldm_v3.py`: reuses the v2 LDM and autoencoder weights but adds raw decoded-state lane-corridor, lane-heading, and initial-clearance losses.
- `train_high_value_scene_generator_v3.py` and `eval_high_value_scene_generator_v3.py`: train and evaluate only the v3 path.

This is intentionally the low-risk Stage B/D ablation. The pretrained autoencoder lane-type vocabulary cannot represent a new CARLA connector class without retraining it, so Stage C semantic embeddings are not falsely claimed as implemented. Stage C is deferred by the current decision to keep the Scenario Dreamer autoencoder/LDM architecture unchanged. The rolling cadence repair is implemented in `RIFT/rift/cbv/initialization/hv_online_generation.py`: a scheduled skip now advances its timestamp, so it does not emit a generation dump on every subsequent frame.

An opt-in lifecycle recorder is also implemented behind `--hv_generation_lifecycle_telemetry`. It writes `lifecycle.jsonl` beside the normal dumps and records spawned raw state, TM-control request, first natural RIFT-CBV activation, lifetime removal, force-update removal, CBV return to TM, CBV termination, and actor-unavailable transitions. It does not alter actor state or any recognition decision. CARLA does not expose a dependable TM-active query, so the event correctly says `tm_control_requested`, not TM-confirmed. Orange actors have no dedicated collision sensor before normal RIFT activation; a disappearance is therefore reported as `actor_unavailable` unless the subsequent green-CBV termination path identifies a collision.

### Stage A: instrumentation and fixed evaluation protocol

Purpose: measure the current failure funnel before changing training behavior.

Use the opt-in lifecycle recorder for every generated actor:

`model state -> spawn state -> TM control requested -> first RIFT recognition/CBV activation -> collision-qualified CBV termination / actor unavailable / lifetime removal / route end`.

Also record map diagnostics without affecting decisions:

- raw xy distance to nearest driving-lane centerline; polygon containment is deferred because current snapshots do not expose reliable local driving-lane polygons;
- lane/connector identity of the nearest map object;
- raw heading difference from local lane tangent;
- initial clearance to ego, each observed vehicle, and the paired generated actor;
- whether the actor is inside the local 64m square.

Fix the rolling cadence so every scheduled inspection advances `_last_rolling_step`, including a gate skip. With interval 50, checks must occur around 50, 100, 150 rather than every frame after the first skip.

Acceptance test: fixed 12-route cross-Town benchmark emits one lifecycle row per generated actor and rolling frames follow the configured interval. This stage changes telemetry and schedule bookkeeping only; it does not modify generated states.

### Stage B: map-topology-preserving v3 dataset

Create `build_high_value_scene_dataset_v3.py`; do not alter v2.

Retain the current target definition: two RIFT-positive target actor states from the current snapshot. Do not replace it with a new future-interaction label task.

Add to the condition and target metadata:

- lane versus lane-connector type;
- road id, lane id, graph predecessor/successor relation;
- local centerline heading and sampled lane width/polygon boundary where map API provides it;
- route-lane membership and signed relation to ego route;
- target-to-nearest-lane association, lateral offset, tangent-heading residual, and lane-polygon containment when the map extractor can supply polygons;
- traffic-context features: target/ego relative speed, target-to-ego clearance, paired-target clearance, observed-agent count.

Quality filtering must reject only demonstrably invalid source labels: no valid local lane association, invalid dimensions, source actor not live, bad map extraction, or initial physical overlap. It must not discard difficult but valid interactions merely because they are near a junction or far from the ego.

Sampling must be stratified by Town and topology bucket, not merely by route count. Required buckets: straight single lane, parallel lanes, opposing lanes, lane connector/junction, merge/split, and dense traffic. Each bucket's actual count must be reported before training.

Acceptance test: v3 dataset audit reports zero invalid target-lane associations, Town/topology distributions, and episode/route leakage checks. It must retain `fov=64` square semantics exactly.

### Stage C: topology-aware condition encoder

**Status: deferred, not implemented.** Current work must not modify the
Scenario Dreamer AutoEncoder/LDM architecture. The Stage B lane metadata is
used only for dataset audit, raw-output feasibility losses, and grouped
evaluation. It is not an additional LDM condition tensor.

Create v3 adapters/model wrappers; retain the pretrained autoencoder/LDM weights as initialization.

First feasible change: add lane semantic embeddings and directional features alongside existing resampled lane polylines. This is lower-risk than replacing the LDM representation.

- Encode lane and connector separately.
- Add centerline tangent, lane width/boundary samples, route membership, and graph-neighbor features.
- Keep ego-local coordinates and the existing route condition.
- Validate checkpoint compatibility explicitly: new embeddings are initialized separately; existing LDM and autoencoder tensors must load from v2/Phase5 without shape corruption.

Second, higher-risk change after the first ablation: generate a lane-anchor distribution plus continuous local state residuals. The decoded output remains raw x/y/yaw/speed; this is model parameterization, not online snapping. Do not start this variant until the semantic-encoder ablation has a measured benefit.

Acceptance test: a unit test confirms the same snapshot yields different condition tensors when lane type/connector/topology differs, while Phase5/v2 paths remain unchanged.

### Stage D: raw-state feasibility losses

Retain the current matched decoded position, heading, speed, size, and
physical-bound reconstruction losses as the main objective. The two targets
are already current-snapshot RIFT-positive CBVs; therefore their matched raw
state reconstruction is the direct trainable high-value objective.

The first V3 run must use only two weak feasibility terms, calculated on the
raw decoded states with no clamp or snap:

1. `lane_corridor_loss`: a low-weight hinge penalty for excessive distance from the source target's associated lane centerline/corridor.
2. `initial_clearance_loss`: a low-weight hinge penalty for initial overlap with ego, observed vehicles, or the paired generated actor. It is not a large-distance repulsion, so valid close interactions remain possible.

`topology_heading_loss` remains available only as a later single-factor
ablation and defaults to zero. It must not be enabled together with further
new losses until the two-term baseline has been evaluated. `route_context_loss`
is deferred: the current snapshots do not provide a trustworthy continuous
supervision target for it, and RIFT must remain an evaluation signal rather
than an inference-time or differentiable training input.

Do not add an auxiliary “RIFT-CBV classifier” loss now. All training targets
are already RIFT-positive, while the available label is discrete and
non-differentiable. Its value belongs in offline and closed-loop evaluation,
not a fabricated gradient target.

For the first run use `lane_corridor_loss_weight=0.10`,
`initial_clearance_loss_weight=0.10`, and
`lane_heading_loss_weight=0.0`. Add losses gradually and run ablations one at a time:

1. Baseline: matched state reconstruction only.
2. Add lane corridor only.
3. Add initial clearance to the corridor run.
4. Only if the preceding run improves held-out feasibility without a material L1 regression, test lane-heading alone as a separate ablation.

Use map-derived ground truth associations from Stage B. Calculate every new loss on the model's decoded raw output. Do not clamp output before loss calculation.

Keep `val_decoded_position_l1` as the checkpoint-selection metric, as required. Record all added validation metrics beside it; reject a variant if it lowers L1 but worsens road containment or clearance materially.

Acceptance test: on held-out v3 data, compare v2 and v3 with the same seeds. Required improvements are lower road-outside rate and collision/overlap rate without worse held-out position L1 beyond a predeclared tolerance.

### Stage E: offline evaluation before online CARLA

Create `eval_high_value_scene_generator_v3.py`. It must evaluate raw proposals, not clamped or snapped proposals.

Report per sample and per Town/topology bucket:

- position L1/L2, heading, speed, and size errors;
- lane-corridor containment rate, and lane-polygon containment only when future snapshots provide polygons;
- nearest-lane distance and heading residual;
- FOV overflow, pair overlap, ego/observed-agent overlap;
- source target versus generated topology-bucket distribution;
- results for one sample and best-of-N separately. Best-of-N is an oracle diagnostic, not a deployment score.

The offline gate for advancing to online testing should be map feasibility improvement across held-out routes, not just a lower aggregate L1 dominated by Town12/13.

### Stage F: closed-loop benchmark

Use a fixed benchmark with a no-HV baseline and v2/v3 runs, same routes, seed, ego PDM-Lite, and RIFT-Pluto configuration. Preserve raw model state in both v2 and v3 conditions.

Primary generator metrics:

- model-call count and conditional CARLA spawn success;
- generated actor road-containment and TM-survival rate;
- orange-to-green RIFT-CBV conversion rate;
- time from spawn to first RIFT-CBV activation;
- 0/1/2 generated RIFT-positive actor count at scheduled evaluations;
- generated-actor collisions, off-road events, disappearance, and lifetime removals;
- two-generated-actor closed-loop high-value success rate.

Secondary scene metrics:

- ego route completion, driving score, and collision ratio;
- CBV collision/progress metrics;
- videos cropped to the same ego 64m FOV.

Compare v3 against v2 only after both runs have identical route/seed configuration. A high ego score with few generated RIFT-CBVs is not a successful generator.

## 5. Online policy after V3

The deployment policy stays deliberately minimal:

1. If the environment already has at least two RIFT-positive actors, skip generation.
2. Otherwise call the generator on the 64m ego-local condition.
3. Spawn the raw state with terrain z only.
4. Let TM and normal RIFT recognition run naturally.
5. Record outcomes; do not use audit to alter state, delete actors, or force CBV registration.

CARLA spawn rejection remains a simulator validity event, not a correction. It must be counted as model infeasibility. No lane snap, heading replacement, speed clamp, audit rerank, or force-update is permitted in the main comparison.

## 6. Decision gates

| gate | required evidence before proceeding |
|---|---|
| A -> B | lifecycle telemetry and corrected rolling cadence verified on 12 routes |
| B -> D | topology-balanced dataset audit with valid lane associations |
| D -> E | raw offline feasibility improves without unacceptable position-L1 regression |
| E -> F | benefit holds on held-out Town/topology buckets |
| F -> deployment candidate | higher generated RIFT-CBV conversion and no worse matched ego safety metrics |

This route makes model quality, rather than online post-processing, responsible for the improvement.

## 7. V3 execution commands

These commands are separate from Phase5/v2. They use the already collected full Phase7 raw snapshots, keep the 64m square FOV, and create new paths only. `FT_B32` is the existing v2 B32 fine-tuned checkpoint used as initialization; it is not overwritten.

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export HV_ROOT=$ROOT/data/hv_init
export RAW_SNAPSHOTS=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
export V3_DATA=$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64
export RIFT_HD_MAP_DIR=$ROOT/RIFT/data/map_data/HD-Map
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export FT_B32=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt
export V3_FT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_fov64_ft_from_v2_b32
cd $ROOT/scenario-dreamer

$PY build_high_value_scene_dataset_v3.py \
  --input_dir $RAW_SNAPSHOTS \
  --output_dir $V3_DATA \
  --mode inpainting \
  --fov 64 \
  --map_source hd_map \
  --rift_root $ROOT/RIFT \
  --rift_hd_map_dir $RIFT_HD_MAP_DIR \
  --num_generate_agents 2 \
  --quality_filter_v2 \
  --min_target_speed 0.0 \
  --max_target_lane_distance 5.0 \
  --sparse_context_copies 4 \
  --seed 42

CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v3.py \
  --train_jsonl $V3_DATA/train.jsonl \
  --val_jsonl $V3_DATA/val.jsonl \
  --output_dir $V3_FT \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --resume_checkpoint $FT_B32 \
  --num_generate_agents 2 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-5 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 \
  --lane_heading_loss_weight 0.0 \
  --initial_clearance_loss_weight 0.10 \
  --best_metric val_decoded_position_l1

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v3.py \
  --jsonl $V3_DATA/test.jsonl \
  --checkpoint $V3_FT/best.ckpt \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 \
  --num_samples_per_condition 4 \
  --output_json $HV_ROOT/eval/phase8_v3_ft_offline.json \
  --output_generated_jsonl $HV_ROOT/eval/phase8_v3_ft_offline_generated.jsonl

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting \
  --jsonl $V3_DATA/test.jsonl \
  --checkpoint $FT_B32 \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 \
  --fov 64 \
  --num_samples_per_condition 4 \
  --output_json $HV_ROOT/eval/phase8_v2_ft_baseline_on_v3_test.json \
  --output_generated_jsonl $HV_ROOT/eval/phase8_v2_ft_baseline_on_v3_test_generated.jsonl
```

Compare the v3 `sampled` and `best_of_n_position_l2_oracle` blocks against the v2 all-sample/best-of-N blocks on this same test JSONL. The primary decision is lower lane-corridor, clearance, and off-FOV failure without a material position-L1/L2 regression. The oracle block is diagnostic only and must not be used as an online sampling or rejection policy.

For the pure Scenario Dreamer composition diagnostic, no CARLA or RIFT process is started. The generic base LDM creates blue background-state proposals; the v3 HV LDM conditions on those proposals and creates two orange high-value candidates. Purple/green are runtime RIFT roles and cannot truthfully be produced by an offline-only script.

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY compose_scenario_dreamer_hv_scene.py \
  --jsonl $V3_DATA/test.jsonl \
  --record_index 0 \
  --output_json $HV_ROOT/eval/phase8_v3_composed_scene_000000.json \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --hv_checkpoint $V3_FT/best.ckpt \
  --background_num_agents 6 \
  --hv_num_agents 2
```

Use this small online V3 smoke test only after the offline report is acceptable. It enables normal RIFT recognition but does not enable audit-driven registration, force-update, clamping, snapping, or actor deletion based on audit. The output `summary.json` includes the lifecycle conversion statistics.

```bash
export OUT=hvgen_v3_debug_fov64_lifecycle_route1711
export DUMP=$HV_ROOT/online_debug/$OUT
mkdir -p $DUMP $ROOT/RIFT/$OUT
chown -R riftuser:riftuser $DUMP $ROOT/RIFT/$OUT

runuser -u riftuser -- bash -lc "
export ROOT=$ROOT
export PY=$PY
export HV_ROOT=$HV_ROOT
export AE_CKPT=$AE_CKPT
export LDM_CKPT=$LDM_CKPT
export V3_FT=$V3_FT
export OUT=$OUT
export DUMP=$DUMP
export CARLA_ROOT=\$ROOT/carla
export CARLA_PYTHONAPI=\$ROOT/carla/PythonAPI/carla
export CARLA_EGG=\$(find \$ROOT/carla/PythonAPI/carla/dist -maxdepth 1 -name 'carla-*.egg' | head -n 1)
export PYTHONPATH=\$ROOT/RIFT:\$ROOT/scenario-dreamer:\$CARLA_PYTHONAPI:\$CARLA_EGG:\${PYTHONPATH:-}
cd \$ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 \$PY scripts/run.py \
  --mode eval \
  --output_dir \$OUT \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset 1711 \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume \
  -rep 1 \
  --save_bev \
  --rift_hd_map_dir \$ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir \$ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator_v3:ScenarioDreamerHighValueSceneGeneratorV3 \
  --hv_generation_checkpoint \$V3_FT/best.ckpt \
  --hv_autoencoder_checkpoint \$AE_CKPT \
  --hv_ldm_checkpoint \$LDM_CKPT \
  --hv_generation_mode sparse_inpainting \
  --hv_num_generate_agents 2 \
  --hv_sd_fov 64 \
  --hv_map_radius 64 \
  --hv_snapshot_map_coordinate_mode right_handed \
  --hv_generation_rolling \
  --hv_generation_rolling_min_step 50 \
  --hv_generation_rolling_interval_steps 50 \
  --hv_generation_lifetime_radius 45 \
  --hv_generation_lifecycle_telemetry \
  --no_hv_generation_fail_on_invalid \
  --hv_generation_dump_dir \$DUMP
\$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir \$DUMP \
  --statistics_dir \$ROOT/RIFT/\$OUT \
  --output_json \$DUMP/summary.json
"
```
