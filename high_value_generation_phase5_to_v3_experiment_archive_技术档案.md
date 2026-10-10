# High-Value Generation: Phase5 to V3 Experiment Archive

## 1. Project objective and non-negotiable semantics

The project augments RIFT/CARLA closed-loop evaluation with initial states for
surrounding vehicles. The target is not merely physically valid traffic: at
runtime, generated vehicles should have map/topology/context-consistent raw
states and should become RIFT-recognized CBVs when they truly interact with
the ego vehicle.

The four visual/control states are:

| Source | Before RIFT recognition | After normal `CBVs_recog()` recognition |
| --- | --- | --- |
| Scenario Dreamer ordinary BV | blue | purple, then RIFT/Pluto controlled |
| high-value (HV) model BV | orange | green, then RIFT/Pluto controlled |

RIFT post-spawn audit is observational only. It must not rank samples, delete
unrecognized actors, alter model x/y/yaw/speed, or directly make an actor a
CBV. `CBVs_recog()` is the only normal route to Pluto control.

The simulator must use raw model x/y/yaw/speed. Road height is queried only to
place the actor on the terrain surface. Position snapping and lane-heading
replacement are prohibited because they hide model error.

## 2. Common server environment

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export HV_ROOT=$ROOT/data/hv_init
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}
```

`CARLA_ROOT`, `CARLA_PYTHONAPI`, the CARLA egg, and `PYTHONPATH` must also be
exported inside `runuser -u riftuser -- bash -lc ...` online commands. Omission
of `$ROOT/scenario-dreamer` causes `ModuleNotFoundError: high_value_init`.

## 3. Phase5 baseline

### 3.1 Task and architecture

Phase5 fine-tuned the Scenario Dreamer LDM for a fixed two-HV-agent
inpainting task. It uses the frozen Scenario Dreamer AutoEncoder and updates
only the LDM. The target state format is:

```text
[x_local, y_local, heading_rad, speed_mps, length_m, width_m, exists]
```

The original primary checkpoint is:

```bash
export PHASE5_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt
```

The model always has a validated two-target task at training time. It does not
support an arbitrary HV target count without changing training data/task
definition. At inference, the generic graph interface can request different
counts, but such output is out of the validated distribution and is not a
claim of performance.

### 3.2 Phase5 offline result

The recorded raw-state evaluation showed substantial residual error:

| metric | raw output | clamp ablation |
| --- | ---: | ---: |
| position L2 | 15.70 | 16.29 |
| heading absolute error | 0.189 | 0.209 |
| speed L1 | 2.61 | 2.49 |
| size L1 | 1.18 | 1.21 |
| physical invalid rate | 0.205 | 0.000 |

Clamping removes reported invalid states but worsens position/heading and is
not a model-quality improvement. Raw output is the valid metric.

## 4. Phase6 online integration and corrections

Implemented RIFT online wiring:

- `high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator`
- CARLA spawning of raw generated state, using road height only.
- online dump JSON, generator/audit/spawn statistics, saved BEV videos.
- normal RIFT recognition/lifecycle telemetry and source-aware visualization.
- existing-RIFT-positive gate: do not call generation once at least two
  existing RIFT-positive actors are present.

Historical issues and their status:

| issue | status |
| --- | --- |
| dataset path `inpainting_sparse_all_recursive` did not exist | resolved by constructing/selecting the actual dataset directory |
| source tree not in `PYTHONPATH` under `runuser` | fixed in online command templates |
| CARLA started as root | invalid; run simulator as `riftuser` |
| `--hv_generation_register_cbv` misunderstood as direct registration | current code leaves candidates empty unless the explicit legacy audit flag is enabled |
| force-update removed valid actors periodically | not used for normal replenish experiments |
| large map/far camera visualization | replaced with ego-local 64m video bounds |

## 5. Phase7 V2 data collection and training

### 5.1 Collection

The recursive raw snapshot collection directory was:

```bash
export RAW_SNAPSHOTS=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
```

It contains 54,936 JSON snapshots from 220 routes, over seeds 0 through 7.
Raw files are symbolic links in some source folders; use `find -L` when
counting source JSON files.

One foreground collection worker is invoked through the forwarding script
below. The completed experiment used eight disjoint route partitions and
separate CARLA/TM ports across GPUs 0-3; every worker must have a distinct
output subdirectory and port pair.

```bash
cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/collect_high_value_init_scenes.py \
  --output_dir $COLLECT_ROOT/worker0_seed0 \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset <worker_specific_route_ids> \
  --seed 0 --repetitions 1 --no_resume \
  --base_port 32000 --base_tm_port 52000 \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --sample_interval_s 1.0 --actor_radius 80 --map_radius 64 \
  --max_frames_per_episode 200
```

Repeat with seeds 0-7 and non-overlapping worker route partitions. The
historical live launcher was `xunlian.sh`; it showed every worker stream in
the terminal and wrote one log per worker.

Merge old recursive and new collection data without basename collisions:

```bash
export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
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

The earlier `build_high_value_scene_dataset_v2.py` JSON serialization failure
was a `PosixPath` summary field; it was fixed without changing the V2 raw data
or model checkpoint.

### 5.2 V2 full dataset and best checkpoint

V2 full-data split:

```text
train: 58,168 records
val:    6,431 records
```

The comparison runs used batch size 32. Their best validation values were:

| training run | best epoch | best val decoded position L1 | best val decoded state loss |
| --- | ---: | ---: | ---: |
| scratch | 12 / 60 | 0.14995 | 0.57911 |
| fine-tune from Phase5 | 12 / 40 | 0.14638 | 0.56352 |

The selected current V2 baseline is:

```bash
export V2_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt
```

Final epochs degraded on both runs, so `best.ckpt`, not `last.ckpt`, is the
comparison checkpoint. Fine-tuning reads Phase5 with `torch.load`; it does not
write Phase5. The AutoEncoder is frozen; the LDM is optimized with a fresh
optimizer state.

### 5.3 V2 offline evaluation command

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting \
  --jsonl $DATASET_DIR/test.jsonl \
  --checkpoint $V2_BEST \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 \
  --fov 64 \
  --num_samples_per_condition 4 \
  --max_records 1000 \
  --output_json $HV_ROOT/eval/v2_test1000_raw.json \
  --output_generated_jsonl $HV_ROOT/eval/v2_test1000_raw_generated.jsonl
```

## 6. V2 online cross-Town baseline

The successful V2-only cross-Town run used routes:

```text
24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333
```

It covers Town01/02/03/04/05/06/07/10HD/11/12/13/15, with seed 0 and one
repeat. The result `hvgen_debug_cross_town_fov64_ftb32_seed0_rerun1` recorded:

| metric | result |
| --- | ---: |
| ego mean driving score | 86.2505 |
| ego mean route completion | 93.7508 |
| ego collision-route ratio | 0.25 |
| model calls | 19 |
| calls with spawned actors | 17 / 19 |
| calls with two immediate generated RIFT-positive actors | 1 / 19 |

This is not evidence that the model is high-quality: immediate two-CBV rate
was low, and a matched no-HV baseline is required for causal driving-score
comparison.

## 7. V3: map-aware data and weak feasibility losses

### 7.1 Motivation and design

V3 does not change the Scenario Dreamer AutoEncoder or LDM architecture.
Stage C topology-encoder architecture changes are intentionally deferred.
V3 adds HD Map enrichment and simple raw-state geometry losses while keeping
decoded two-HV-state reconstruction as the primary checkpoint metric.

V3 losses:

```text
diffusion loss
+ 0.4 * decoded state reconstruction loss
+ 0.10 * lane corridor loss
+ 0.10 * initial clearance loss
+ 0.00 * lane heading loss
```

Heading is logged but not optimized because junction/connector tangents are
ambiguous. `val_decoded_position_l1` remains the best-checkpoint criterion.

### 7.2 HD Map dataset construction

Old raw snapshots did not serialize local map lanes. The V3 snapshot-lane
builder therefore rejected all records. This was not a y-axis sign issue: the
stored lane data was empty. V3 now enriches each snapshot from RIFT HD Map.

Full V3 dataset command:

```bash
export V3_DATA=$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full
cd $ROOT/scenario-dreamer
$PY build_high_value_scene_dataset_v3.py \
  --input_dir $RAW_SNAPSHOTS \
  --output_dir $V3_DATA \
  --mode inpainting \
  --fov 64 \
  --map_source hd_map \
  --rift_root $ROOT/RIFT \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --num_generate_agents 2 \
  --quality_filter_v2 \
  --min_target_speed 0.0 \
  --max_target_lane_distance 5.0 \
  --sparse_context_copies 4 \
  --seed 42
```

Actual complete V3 output:

| item | value |
| --- | ---: |
| processed snapshots | 54,936 |
| HD Map errors | 0 |
| raw records | 57,580 |
| lane rejected | 1,565 |
| quality rejected | 2,125 |
| duplicate records removed | 16,702 |
| train / val / test | 30,468 / 4,558 / 3,727 |
| episode leakage | 0 across every split pair |

Topology before split: 38,185 connector/junction, 12,485 multi-lane, 6,905
opposing/parallel, and only 5 sparse/single-lane records. Town12 remains
dominant, and several towns appear in only one split; report per-town and
per-topology metrics rather than only one global average.

### 7.3 V3 training implementation audit

V3 originally had autograd inplace errors in the new decoded/raw geometry
branch. They were fixed only in V3:

- latent conditioning replacement is `torch.where`, not indexed assignment;
- raw geometry uses functional agent inverse scaling, not in-place
  `unnormalize_scene`.

The server one-batch test completed full forward and backward:

```text
V3 one-batch backward PASS
loss: 0.72046
decoded_state_loss: 1.29343
lane_corridor_loss: 0.34832
initial_clearance_loss: 0.0
```

V3 fine-tuning command:

```bash
export V3_FT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_fov64_hdmap_full_ft_from_v2b32_noninplace_20260924
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v3.py \
  --train_jsonl $V3_DATA/train.jsonl \
  --val_jsonl $V3_DATA/val.jsonl \
  --output_dir $V3_FT \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --resume_checkpoint $V2_BEST \
  --num_generate_agents 2 \
  --batch_size 32 --epochs 30 --lr 1e-5 \
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
```

V3 must be evaluated against both V3 test and the older V2 test before it can
replace V2. Its new data and losses can improve geometry but can also create
distribution shift/forgetting.

## 8. Composed ordinary-BV plus HV generation

### 8.1 Intended iterative semantics

Each generation event must do:

```text
current model-generated scene context
-> Scenario Dreamer proposes ordinary BVs
-> current context + newly proposed ordinary BVs
-> HV model proposes two HV BVs
```

At later replenishment events, previously generated ordinary and HV actors are
read from the live snapshot and remain model context. Normal replenishment is
event-driven (missing/dead/out-of-lifetime generated HV actors), not forced
periodic replacement.

### 8.2 Implementation

Entry point:

```text
high_value_init.scene_generator_composed_v2:
ScenarioDreamerComposedHighValueSceneGeneratorV2
```

It tags sources as `scenario_dreamer_background` and `high_value`, records
context counts, and is visualized blue/orange before normal RIFT recognition.

`--sd_background_model_only_traffic` is the strict mode:

- at reset it removes every non-model CARLA vehicle, not merely nearby
  `role_name=background` actors;
- after every CARLA tick it removes any non-model vehicle introduced later;
- it preserves ego and actor IDs produced by the high-value initializer;
- it is opt-in, so legacy V2 and ordinary RIFT runs are unchanged.

The composed entry has an explicit BV source parameter:

```text
--sd_background_source carla
    existing CARLA BVs are the context for two HV outputs; no ordinary
    Scenario Dreamer BV is added. This is the V2-compatible default.

--sd_background_source scenario_dreamer
    Scenario Dreamer creates ordinary BVs first, then those BVs plus prior
    model actors are the context for two HV outputs.
```

`--sd_background_model_only_traffic` is valid only with
`--sd_background_source scenario_dreamer`.

This still uses CARLA as the physics simulator and to spawn model states. It
does not use CARLA/Traffic Manager to originate ordinary traffic states.

### 8.3 Current empirical status

One-route `sd1 + hv2` diagnostic on route 1711 succeeded:

```text
1 Scenario Dreamer ordinary BV + 2 V2 HV BVs spawned
both HV BVs became normal RIFT CBVs at frame 26
both later returned to Traffic Manager at frame 198
```

The 12-route cross-Town composed run before strict model-only mode recorded:

| item | result |
| --- | ---: |
| full scene spawn success | 10 / 12 |
| ordinary BV spawned | 10 |
| HV spawned | 20 |
| blue -> purple conversion | 2 / 10 |
| orange -> green conversion | 9 / 20 |
| immediate two RIFT-positive generated BVs | 1 / 12 |
| ego driving score | 84.02 |

This run is a chain-integration diagnostic, not a valid all-model traffic
evaluation: it only replaced nearby background at initialization, and CARLA
traffic could still remain/appear later.

### 8.4 Generic background LDM is not deployable

The generic `scenario_dreamer_ldm_large_nuplan/last.ckpt` was audited with no
background-specific checkpoint over 800 one-BV samples:

| metric | value |
| --- | ---: |
| fully feasible scenes | 111 / 800 = 13.875% |
| invalid size | 71.75% |
| distance to lane > 5m | 45.25% |
| FOV invalid | 10.50% |
| lane distance mean / p90 | 6.95m / 18.11m |

Do not claim `sd6 + hv2` works with this checkpoint. Do not fix these values
by lane snapping, heading replacement, or state clamping. A separate ordinary
BV dataset and background checkpoint are required.

Offline audit command:

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_scenario_dreamer_background_generator.py \
  --jsonl $V3_DATA/test.jsonl \
  --autoencoder_checkpoint $AE_CKPT \
  --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 1 \
  --num_samples_per_condition 4 \
  --max_records 200 \
  --fov 64 --fov_margin 2 --max_lane_distance 5.0 \
  --min_length 2.0 --max_length 8.0 \
  --min_width 1.4 --max_width 3.2 \
  --min_clearance 2.5 \
  --output_json $HV_ROOT/eval/sd_background_audit/summary.json \
  --output_generated_jsonl $HV_ROOT/eval/sd_background_audit/generated.jsonl
```

### 8.5 Current count control and original Scenario Dreamer semantics

`--sd_background_num_agents 1` was used only for a spawn-debug experiment. It
is not a model or architecture limit. In composed Scenario Dreamer mode it may
be set to a positive fixed value such as `6` or `8`, subject to the graph
capacity:

```text
1 ego + observed BV context + requested ordinary BV targets + 2 HV targets <= 61
```

The two V2 HV outputs intentionally remain fixed at two. All V2 training
records were built with `num_generate_agents=2`, selecting two RIFT-positive
targets. The graph implementation can technically allocate a different number
of target nodes, but running the V2 checkpoint with a count other than two is
out of distribution and is not evidence of adaptive HV count generation.

The original NuPlan Scenario Dreamer inpainting implementation differs in two
important ways. In `utils/inpainting_helpers.py`, it first crops a 64 m square
and uses the `x <= 0` half after route/ego normalization as the conditioning
partition; the opposite half is generated. It also samples the number of
inpainted agents from a learned conditional count distribution
`P(num_after | num_lanes, num_before)` rather than taking a fixed CLI count.
For initial-scene generation it similarly samples the joint lane/agent count
from `init_prob_matrix`.

The current RIFT wrapper does not yet reproduce either behavior:

- `high_value_init/scene_generator.py` places every live BV in the complete
  ego-local square `[-32, 32] x [-32, 32]` into the inpainting condition; it
  does not retain only the rear half or generate only the forward half.
- `RouteConditionedHighValueLDM` receives an explicitly supplied
  `num_generate_agents`; it does not load or sample the original Scenario
  Dreamer count matrices.
- V2/V3 HV datasets select targets over the complete FOV. They are not
  supervised as "rear-half ordinary BV context -> forward-half two HV states".

Therefore the current composed generator can create a fixed requested number
of ordinary BVs and then two HV BVs conditioned on them, but it must not be
described as the original directional, learned-count Scenario Dreamer
inpainting pipeline.

The current rolling hook is also HV replenishment, not a general ordinary-BV
rollout manager: it inspects periodically, skips when two generated HV actors
remain, and otherwise invokes one complete composed proposal. With an ordinary
background count `K`, every such event adds `K` new ordinary proposals. Running
it indefinitely can accumulate ordinary actors and eventually violate the
61-agent graph capacity. Do not use it to claim continuous ordinary-traffic
prediction.

The correct follow-up implementation is a separate ordinary-BV controller:

1. use the original compatible rear-half partition and learned count
   distribution, or train an equivalent count predictor on the new data;
2. calculate the desired ordinary BV count inside the FOV, remove only expired
   model actors, and generate the missing count rather than a full new batch;
3. pass the resulting live ordinary BV states to the fixed-two V2/V3 HV model;
4. train a variable-target-count HV dataset before permitting adaptive HV
   output counts.

Safe initial-only smoke command for the current fixed-count composition:

```bash
cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py \
  --mode eval \
  --output_dir sd6_hv2_initial_only_route1711_ftb32 \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset 1711 \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume -rep 1 --save_bev \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator_composed_v2:ScenarioDreamerComposedHighValueSceneGeneratorV2 \
  --hv_generation_checkpoint $V2_BEST \
  --hv_autoencoder_checkpoint $AE_CKPT \
  --hv_ldm_checkpoint $LDM_CKPT \
  --hv_generation_mode sparse_inpainting \
  --hv_sd_fov 64 \
  --sd_background_source scenario_dreamer \
  --sd_background_num_agents 6 \
  --sd_background_model_only_traffic \
  --hv_num_generate_agents 2 \
  --hv_generation_existing_rift_positive_threshold 2 \
  --hv_generation_dump_dir $HV_ROOT/online_debug/sd6_hv2_initial_only_route1711_ftb32
```

In this command CARLA remains the physics/spawn executor, but CARLA-originated
ordinary traffic is removed. Both ordinary-BV and HV initial states originate
from the composed model proposal. Given the existing generic background audit,
this is an integration diagnostic only, not a performance claim.

## 9. Current risks and next experiments

1. Train/evaluate V3 HV checkpoint before online replacement. Keep V2 best
   immutable as the baseline.
2. Construct an ordinary vehicle target dataset from real CARLA background
   states; train a separate background LDM/checkpoint over variable ordinary
   target counts. This is required before large all-model traffic scenes.
3. Use strict `--sd_background_model_only_traffic` only for model-only
   composition experiments. Verify every live non-ego vehicle has a generated
   source in dumps/video.
4. For every online result report: calls, full spawn success, actor-level
   retry failures, immediate 0/1/2 RIFT-positive buckets, blue->purple,
   orange->green, lifecycle terminal events, per-route completion, and ego
   collision statistics.
5. Never interpret ego driving score as a generator-only score without a
   matched route/seed/config baseline.

## Appendix A. Complete phase5-to-v3 reproduction ledger

### A.1 Scope, source documents, and status convention

This appendix is the command-and-evidence ledger missing from the earlier
summary sections. It consolidates the historical records below. Where a
historical command failed or was superseded, that fact is retained instead of
silently presenting it as a recommended command.

| source | authoritative material imported here |
| --- | --- |
| `high_value_init_generation_experiment_status_next_steps.md` | original objective, Phase0-6 requirements, early recursive data, Phase5 training/evaluation, and early online failure analysis |
| `high_value_init_generation_phase0_phase6_audit.md` | Phase0-6 implementation audit and the intended validation funnel |
| `high_value_generation_phase7_reproduction_复现_v2.md` | Phase7 multi-seed collection, union, final B32 runs, offline evaluation, and 12-Town online run |
| `high_value_generation_v3_technical_route_技术路线和命令.md` | V3 constraints, HD-map data route, raw-state losses, decision gates, and V3 commands |
| `xunlian.sh`, `tuili.sh` | historical end-to-end launchers; do not run blindly because they combine already-completed stages with new stages |

Status labels used below:

- **completed**: a recorded output/result exists.
- **historical failed**: useful for diagnosis, but not a rerun recommendation.
- **current reproducible**: separate output paths and valid current code are required.
- **planned**: implementation or experiment not yet evidenced by a result.

### A.2 Project task and invariant technical route

The project goal is not reconstruction alone. In an active RIFT/CARLA
rollout, generate surrounding-vehicle initial raw states that are consistent
with ego, route, local map, road topology, and traffic context, and that can
naturally become RIFT CBVs during the rollout. The two fixed HV target states
use `[x, y, heading, speed, length, width, valid]` in the ego-local frame.

The evolution of the route is:

```text
RIFT-native labeled snapshots
-> Scenario Dreamer AutoEncoder/LDM initialization
-> Phase5 two-HV inpainting fine-tune
-> Phase0-6 online sparse generation / diagnostics / raw-state losses
-> Phase7 multi-seed data expansion and V2 B32 comparison
-> V3 HD-map-associated data plus weak raw feasibility losses
-> closed-loop evaluation with normal RIFT recognition and lifecycle metrics
```

Non-negotiable semantics for every current comparison:

1. RIFT labels are used for collection, target selection, the existing-two-CBV
   gate, and evaluation; they are not model inputs or online state editors.
2. Raw predicted x/y/yaw/speed are spawned without lane snapping, yaw
   replacement, speed replacement, clamp, or audit reranking. Road waypoint
   lookup supplies terrain z only.
3. Orange/blue are generated actors under Traffic Manager initially. Green and
   purple arise only after normal `CBVs_recog()` activation and use RIFT-Pluto.
4. Audit is observational. It must not delete a generated actor merely because
   it is not immediately RIFT-positive.
5. `val_decoded_position_l1` remains the checkpoint selector. Offline raw
   feasibility and closed-loop conversion are required secondary evidence.

### A.3 Common server setup and preflight

Use these exports in the parent shell. The nested `runuser` shell must repeat
them, especially `$ROOT/scenario-dreamer` in `PYTHONPATH`.

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT=$ROOT/carla
export CARLA_PYTHONAPI=$CARLA_ROOT/PythonAPI/carla
export CARLA_EGG=$(find "$CARLA_PYTHONAPI/dist" -maxdepth 1 -name 'carla-*.egg' | head -n 1)
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_PYTHONAPI:$CARLA_EGG:${PYTHONPATH:-}
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PHASE5_CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt
export V2_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt
```

```bash
cd $ROOT/RIFT
$PY - <<'PY'
from agents.navigation.local_planner import RoadOption
from rift.carla_runner import CarlaRunner
import high_value_init
print('RIFT and high_value_init import ok')
PY
```

If this import check fails, do not launch CARLA. The historical all-route runs
with `ModuleNotFoundError: high_value_init` executed the ego benchmark but did
not execute the HV generator, so their ego scores are not HV results.

### A.4 Phase5: original data, training, evaluation, and limitations

**Completed/recorded assets.** The recursive source directory was
`$HV_ROOT/rift_labeled_all_recursive_20260919_133508`, containing 70,329
symbolic-linked JSON files when counted with `find -L`. The early recursive
dataset build retained 13,706 records: train 10,939, val 1,518, test 1,249.
Its audit passed, with mean observed count 3.6345 and mean original
RIFT-positive count 2.5676.

**Historical failed invocation.** The first Phase5 command pointed to the
non-existent path `datasets/inpainting_sparse_all_recursive`; its
`FileNotFoundError` was a dataset-path problem, not a model failure. Keep this
record because it explains the later actual directory choice:

```bash
# historical failed path; do not reuse without first building it
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \
  --mode inpainting \
  --train_jsonl $HV_ROOT/datasets/inpainting_sparse_all_recursive/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/inpainting_sparse_all_recursive/val.jsonl \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5 \
  --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 --epochs 20 --batch_size 16 --lr 5e-5 \
  --decoded_state_loss_weight 0.2 \
  --decoded_position_loss_weight 1.0 \
  --decoded_heading_loss_weight 0.5 \
  --decoded_speed_loss_weight 0.7 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.2
```

**Historical completed training configuration.** The recorded early command
uses the constructed recursive dataset path. It is retained for Phase5
reproduction only; it writes the existing Phase5 output and must therefore be
redirected to a new output path for a new experiment.

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \
  --mode inpainting \
  --train_jsonl $HV_ROOT/datasets/inpainting_all_recursive/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/inpainting_all_recursive/val.jsonl \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm_all_recursive \
  --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --resume_checkpoint $HV_ROOT/checkpoints/hv_route_inpaint_ldm_combined_unique/best.ckpt \
  --num_generate_agents 2 --epochs 20 --batch_size 16 --lr 5e-5 \
  --decoded_state_loss_weight 0.1
```

Recorded Phase5 best checkpoint: `hv_route_inpaint_ldm_sparse_phase5/best.ckpt`.
The later raw evaluation on 100 conditions reported position L2 15.70,
heading absolute 0.189, speed L1 2.61, size L1 1.18, and physical-invalid
rate 0.205. Clamp reduced the reported invalid rate to zero but worsened
position L2 to 16.29 and heading to 0.209; clamp is an ablation, not an
improvement.

Phase5 raw/offline evaluation pattern:

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator.py \
  --mode inpainting \
  --jsonl $HV_ROOT/datasets/inpainting_all_recursive/test.jsonl \
  --checkpoint $PHASE5_CKPT \
  --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --output_json $HV_ROOT/eval/phase5_raw_s1.json \
  --output_generated_jsonl $HV_ROOT/eval/phase5_raw_s1_generated.jsonl \
  --num_generate_agents 2 --num_samples_per_condition 1 --max_records 100
```

### A.5 Phase0-6: implementation audit, commands, and historical findings

| phase | implemented change | result / remaining evidence |
| --- | --- | --- |
| 0/1 | `sparse_inpainting` maps online sparse context to model inpainting and no longer requires hidden actors | resolved low-density precheck blockage; strict/full mode remains separate |
| 2 | dump records generator load/call, hidden context, generation/spawn/audit failure stage and reason | permits distinguishing no-call, model failure, spawn failure, and audit outcome |
| 3 | generated-only audit accounting, optional legacy audit registration, normal RIFT registration/lifecycle reporting | later corrected: normal `CBVs_recog()` rather than audit must determine Pluto control |
| 4 | sparse-context dataset copies and low-observed-count audit | Phase7 uses four sparse copies and observed counts 0-4 |
| 5 | decoded position/heading/speed/size reconstruction loss and physical-bound loss | state error became trainable; later V3 adds weak raw geometry terms |
| 6 | static audit, offline evaluation, online smoke/multi-route command and summary funnel | all numerical claims still require actual server outputs |

Canonical static audit before server work:

```bash
cd $ROOT/RIFT
$PY scripts/audit_high_value_generation_pipeline.py \
  --root $ROOT --scenario_dreamer_root $ROOT/scenario-dreamer
$PY -m py_compile \
  scripts/run.py scripts/summarize_high_value_generation_online.py \
  rift/carla_runner.py rift/cbv/initialization/hv_online_generation.py \
  rift/gym_carla/envs/carla_env.py rift/gym_carla/visualization/bev_render.py
```

The first online route-1711 strict-inpainting smoke completed the ego route
but called no model because it required two hideable background actors and
found only one or zero. This is a task-entry mismatch, not a sampling result.
The corrected normal policy is sparse inpainting, a gate only when at least
two live RIFT-positive actors already exist, raw-state spawning, and normal
RIFT recognition afterwards.

### A.6 Phase7: completed multi-seed collection and exact data build

**Completed collection.** `rift_labeled_phase7_v2_uncovered_routes_seed0_7`
contains 54,936 snapshots from all 220 routes. Eight workers used GPUs 0-3,
disjoint XML route subsets, seeds 0-7, separate CARLA/TM ports, and live
terminal-prefixed output. The actual historical launcher is `xunlian.sh`;
running it again would repeat collection and later training, so use only its
separate blocks when reproducing a missing stage.

Collection verification:

```bash
export COLLECT_ROOT=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
python - <<'PY'
import glob, json, os, re
from collections import Counter
root=os.environ['COLLECT_ROOT']; counts=Counter()
for path in glob.glob(root + '/**/*.json', recursive=True):
    try:
        scene=json.load(open(path)).get('scene', {})
        m=re.search(r'RouteScenario_(\d+)', str(scene.get('scene_id','')) + str(scene.get('episode_key','')))
        if m: counts[m.group(1)] += 1
    except Exception: pass
print('num_files', sum(counts.values()))
print('num_routes', len(counts))
print('routes_with_lt10_snapshots', sum(v < 10 for v in counts.values()))
PY
```

Observed: `num_files=54936`, `num_routes=220`, and
`routes_with_lt10_snapshots=0`.

Union and V2 dataset construction actually used for final B32 models:

```bash
export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
export UNION_DIR=$HV_ROOT/rift_labeled_union_phase5_phase7_v2_uncovered_routes_seed0_7
export DATASET_DIR=$HV_ROOT/datasets/inpainting_sparse_phase7_v2_uncovered_routes_seed0_7
mkdir -p "$UNION_DIR"
i=0
find -L "$OLD_COMBINED" "$COLLECT_ROOT" -type f -name '*.json' -print0 |
while IFS= read -r -d '' file; do
  i=$((i + 1))
  ln -sfn "$file" "$UNION_DIR/$(printf '%09d' "$i")_$(basename "$file")"
done

cd $ROOT/scenario-dreamer
$PY build_high_value_scene_dataset_v2.py \
  --input_dir "$UNION_DIR" --output_dir "$DATASET_DIR" \
  --mode inpainting --min_rift_positive_agents 2 --num_generate_agents 2 \
  --actor_radius 80 --fov 64 --positive_target_policy route_s_diversity \
  --sparse_context_copies 4 --sparse_context_min_observed 0 \
  --sparse_context_max_observed 4 --val_ratio 0.10 --test_ratio 0.10 --seed 42 \
  --source real_positive_phase7_v2_uncovered_multiseed --quality_filter_v2 \
  --min_target_speed 0.5 --max_target_speed 25 \
  --min_target_ego_distance 3 --max_target_ego_distance 60 \
  --target_fov_margin 2 --min_target_pair_distance 2 --dedup_grid_xy 1

$PY audit_high_value_scene_dataset.py \
  --train_jsonl "$DATASET_DIR/train.jsonl" \
  --val_jsonl "$DATASET_DIR/val.jsonl" \
  --test_jsonl "$DATASET_DIR/test.jsonl" \
  --expected_mode inpainting --expected_num_generate_agents 2 --fov 64 \
  --output_json "$DATASET_DIR/audit.json"
```

Final B32 dataset evidence: train 58,168, val 6,431. Sparse-context records
were 78.6% of the dataset; low-observed (`<=2`) records were about 66.7% of
train and 67.2% of val. This distribution deliberately exposes the HV model
to sparse conditions but does not create an adaptive target-count task.

### A.7 Phase7: all recorded V2 training runs

The following commands are intentionally all listed. Every output directory
is distinct; a resume checkpoint is read-only input and is not overwritten.

**Historical B16 fine-tune from Phase5** (the first Phase7 script version):

```bash
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting --train_jsonl "$DATASET_DIR/train.jsonl" --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_uncovered_routes_seed0_7_ft_from_phase5 \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$PHASE5_CKPT" --num_generate_agents 2 \
  --epochs 30 --batch_size 16 --lr 2e-5 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

**Historical B16 scratch run** (same first Phase7 script version):

```bash
CUDA_VISIBLE_DEVICES=1 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting --train_jsonl "$DATASET_DIR/train.jsonl" --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_uncovered_routes_seed0_7_scratch \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --epochs 50 --batch_size 16 --lr 5e-5 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

**Completed B32 scratch baseline:**

```bash
export SCRATCH_B32=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_scratch_20260922
CUDA_VISIBLE_DEVICES=1 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting --train_jsonl "$DATASET_DIR/train.jsonl" --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$SCRATCH_B32" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --epochs 60 --batch_size 32 --lr 5e-5 --seed 0 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

Result: best epoch 12/60, position L1 0.1499497757, state loss 0.5791099119;
epoch 59 position L1 degraded to 0.1617999601.

**Completed B32 fine-tune from Phase5, the current V2 offline baseline:**

```bash
export FT_B32=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922
CUDA_VISIBLE_DEVICES=3 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting --train_jsonl "$DATASET_DIR/train.jsonl" --val_jsonl "$DATASET_DIR/val.jsonl" \
  --output_dir "$FT_B32" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$PHASE5_CKPT" --num_generate_agents 2 \
  --epochs 40 --batch_size 32 --lr 2e-5 --seed 0 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

Result: best epoch 12/40, position L1 0.1463771265, state loss 0.5635170419;
epoch 39 position L1 degraded to 0.1613097740. Use only
`$FT_B32/best.ckpt` for the current V2 reference. Training creates a new
checkpoint directory, keeps the AutoEncoder frozen, and does not modify
`$PHASE5_CKPT`.

### A.8 Phase7 offline evaluation and result interpretation

The raw-state V2 evaluation is the primary offline report. Clamp is only a
separate diagnostic because it edits the output after sampling.

```bash
export CURRENT_CKPT=$FT_B32/best.ckpt
export EVAL_DIR=$HV_ROOT/eval/phase7_v2_full_b32_ft_from_phase5
mkdir -p "$EVAL_DIR"
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting --jsonl "$DATASET_DIR/test.jsonl" \
  --checkpoint "$CURRENT_CKPT" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --fov 64 --boundary_margin 2 --min_pair_distance 2 \
  --num_samples_per_condition 4 --max_records 1000 \
  --output_json "$EVAL_DIR/test1000_bestof4_raw.json" \
  --output_generated_jsonl "$EVAL_DIR/test1000_bestof4_raw_generated.jsonl"

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting --jsonl "$DATASET_DIR/test.jsonl" \
  --checkpoint "$CURRENT_CKPT" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --fov 64 --boundary_margin 2 --min_pair_distance 2 \
  --num_samples_per_condition 4 --max_records 1000 --clamp_generated_states \
  --output_json "$EVAL_DIR/test1000_bestof4_clamp.json" \
  --output_generated_jsonl "$EVAL_DIR/test1000_bestof4_clamp_generated.jsonl"
```

Report reconstruction error, raw physical invalid rate, FOV overflow,
lane-distance/containment once available, pair clearance, one-sample result,
and best-of-N oracle result separately. Best-of-N must never be used as an
online sample rejection policy.

### A.9 Online experiment chronology and result validity

| run family | observed state | interpretation |
| --- | --- | --- |
| Phase6 route1711 strict smoke | hidden slots insufficient; no generator call | integration/task-definition failure, not model score |
| all-220 force/replenish pre-import-fix | every HV import failed with `ModuleNotFoundError: high_value_init` | invalid HV experiment; ego benchmark finished without HV generator |
| all-220 import-fix reruns | runs stopped/partial due runtime failures including Pluto non-finite assertion | no completed matched comparison; do not aggregate as final model result |
| V2 12-Town rerun1 | completed and generator actually invoked | valid diagnostic, but not a no-HV causal comparison |
| V2 raw-state route1711 | successful spawn/normal lifecycle in some runs | height-only placement is an instrumentation/state-contract correction; previous videos alone cannot prove all raw state components were preserved |
| composed `sd1+hv2` route1711 | one ordinary model BV plus two HV states spawned | integration proof only; generic background model quality remains inadequate |

The completed V2 cross-Town diagnostic used routes
`24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333`,
PDM-Lite ego, RIFT-Pluto CBV policy, seed 0, one repeat, and ego-local video
bounds 32 m. Its valid measurements are:

| metric | value |
| --- | ---: |
| routes | 12 |
| mean ego driving score | 86.2505 |
| mean route completion | 93.7508 |
| ego collision-route ratio | 0.25 |
| generator calls | 19 |
| calls with actor spawn | 17 / 19 |
| immediate generated RIFT-positive count 2 | 1 / 19 |
| immediate generated RIFT-positive count 1 | 8 / 19 |
| immediate generated RIFT-positive count 0 | 10 / 19 |

This proves integration but not a deployable high-value generator. The
conditional two-positive rate is 5.3%, and the route score has no matched
no-HV control.

Current reproducible V2 online command pattern (run as `riftuser`, new output
directory, no force update):

```bash
export OUT=hvgen_debug_cross_town_fov64_ftb32_seed0_newrun
export DUMP=$HV_ROOT/online_debug/$OUT
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
  --mode eval --output_dir $OUT \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset 24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333 \
  --seed 0 --ego_cfg pdm_lite.yaml --cbv_cfg rift_pluto.yaml --no_resume -rep 1 --save_bev \
  --video_render_bounds 32 --video_render_map_radius 64 --video_render_offset 0 \
  --rift_hd_map_dir \$ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir \$ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator \
  --hv_generation_checkpoint $V2_BEST \
  --hv_autoencoder_checkpoint $AE_CKPT --hv_ldm_checkpoint $LDM_CKPT \
  --hv_generation_mode sparse_inpainting --hv_num_generate_agents 2 --hv_sd_fov 64 \
  --hv_generation_max_attempts 16 --hv_generation_rolling \
  --hv_generation_rolling_min_step 50 --hv_generation_rolling_interval_steps 50 \
  --hv_generation_lifetime_radius 45 \
  --hv_generation_existing_rift_positive_threshold 2 \
  --hv_generation_lifecycle_telemetry \
  --no_hv_generation_fail_on_invalid --hv_generation_dump_dir $DUMP
"

$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir "$DUMP" --statistics_dir "$ROOT/RIFT/$OUT" --output_json "$DUMP/summary.json"
```

### A.10 V3: full data, implementation, training, evaluation, and status

V3 is separate from V2. It does not alter the base AutoEncoder/LDM topology.
It enriches snapshot map associations from the RIFT HD map and adds only weak
raw decoded lane-corridor and initial-clearance losses. The initial V3
in-place-autograd failure was fixed with functional `torch.where` conditioning
and non-inplace state inverse scaling. One server batch completed forward and
backward, but this is execution evidence only, not model-quality evidence.

V3 full build, then audit the explicit output file `quality_report.json`
(not a non-existent `dataset_summary.json`):

```bash
export RAW_SNAPSHOTS=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
export V3_DATA=$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full
cd $ROOT/scenario-dreamer
$PY build_high_value_scene_dataset_v3.py \
  --input_dir $RAW_SNAPSHOTS --output_dir $V3_DATA --mode inpainting --fov 64 \
  --map_source hd_map --rift_root $ROOT/RIFT --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --num_generate_agents 2 --quality_filter_v2 --min_target_speed 0.0 \
  --max_target_lane_distance 5.0 --sparse_context_copies 4 --seed 42

python - <<'PY'
import json, os
root=os.environ['V3_DATA']
for name in ['quality_report.json', 'audit.json']:
    path=os.path.join(root, name)
    print(name, 'exists=', os.path.exists(path))
    if os.path.exists(path): print(json.load(open(path)))
PY
```

Completed V3 data result: 54,936 snapshots processed; HD-map errors 0;
57,580 raw records; 1,565 lane rejections; 2,125 quality rejections; 16,702
duplicates removed; train/val/test 30,468/4,558/3,727; no episode overlap.
The data is not town-balanced: Town12 dominates train; Town10HD appears in
validation only and Town15 heavily in test. Per-town/per-topology evaluation
is mandatory.

V3 fine-tune from immutable V2 best:

```bash
export V3_FT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_fov64_hdmap_full_ft_from_v2b32_noninplace_20260924
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v3.py \
  --train_jsonl $V3_DATA/train.jsonl --val_jsonl $V3_DATA/val.jsonl \
  --output_dir $V3_FT --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --resume_checkpoint $V2_BEST --num_generate_agents 2 --batch_size 32 --epochs 30 --lr 1e-5 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 --lane_heading_loss_weight 0.0 \
  --initial_clearance_loss_weight 0.10 --best_metric val_decoded_position_l1
```

V3 and V2 must be evaluated on exactly the same V3 test split before online
work. Neither `V3_FT` nor `$V2_BEST` is overwritten by these commands.

```bash
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v3.py \
  --jsonl $V3_DATA/test.jsonl --checkpoint $V3_FT/best.ckpt \
  --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 --num_samples_per_condition 4 \
  --output_json $HV_ROOT/eval/phase8_v3_ft_offline.json \
  --output_generated_jsonl $HV_ROOT/eval/phase8_v3_ft_offline_generated.jsonl

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting --jsonl $V3_DATA/test.jsonl --checkpoint $V2_BEST \
  --autoencoder_checkpoint $AE_CKPT --ldm_checkpoint $LDM_CKPT \
  --num_generate_agents 2 --fov 64 --num_samples_per_condition 4 \
  --output_json $HV_ROOT/eval/phase8_v2_ft_baseline_on_v3_test.json \
  --output_generated_jsonl $HV_ROOT/eval/phase8_v2_ft_baseline_on_v3_test_generated.jsonl
```

### A.11 Ordinary Scenario Dreamer BV composition: present state and gap

The composed V2 entry supports two explicitly distinct paths:

```text
--sd_background_source carla
    live CARLA ordinary BVs are passed as HV context; only two HV states are generated.

--sd_background_source scenario_dreamer
    an explicit fixed number of model ordinary BVs is generated, appended to
    live context, then two HV states are generated.
```

`--sd_background_num_agents 1` was only a spawn-debug setting. It can be 6
or another positive number subject to the 61-agent graph capacity. However,
the current wrapper uses an externally fixed count; it does not use the
original Scenario Dreamer learned conditional count matrices. Current rolling
is HV replenishment and would append a new ordinary batch for every HV
replenishment event. It is not yet a safe continuous ordinary-traffic
predictor and must not be represented as one.

The original NuPlan Scenario Dreamer inpainting path crops a 64 m square and
conditions on its `x <= 0` partition, then generates the opposite partition;
it samples agent count from a learned conditional distribution. The present
RIFT wrapper instead conditions on all live BV states within the full square
and explicitly allocates target slots. Current V2/V3 HV training chooses two
targets anywhere in the full FOV. Thus the desired "rear-half BV context ->
front-half prediction" experiment is not yet implemented or trained.

The generic base LDM audit was poor: only 111/800 one-BV samples were fully
feasible; invalid size 71.75%, lane distance >5 m 45.25%, and FOV-invalid
10.50%. It cannot support a claim that `sd6 + hv2` is deployable. The next
correct implementation is a separate normal-BV dataset/checkpoint plus a
count/replenishment controller that generates only the missing ordinary BVs,
then invokes the fixed-two HV model.

### A.12 Final review checklist and unresolved decisions

Before claiming any replacement model is better, retain the following
evidence in one result directory:

1. Dataset counts, split leakage audit, Town/topology distribution, and HD-map
   extraction errors.
2. Training `history.json`, selected checkpoint, and proof that the source
   checkpoint was read but not overwritten.
3. Raw V2/V3 offline report on the same test JSONL, one-sample and oracle
   best-of-N reported separately.
4. Matched no-HV, V2-HV, and V3-HV online runs on identical routes/seeds,
   PDM-Lite/RIFT-Pluto configs, video bounds, and CARLA settings.
5. Online funnel: generator calls, full-scene spawn success, actor-level
   spawn retries, raw road/FOV/clearance diagnostics, immediate 0/1/2
   RIFT-positive buckets, lifecycle blue->purple and orange->green conversion,
   termination/lifetime events, completion, ego driving score, and collision
   ratio.

Open technical decisions, not completed claims:

- Stage C map-topology embeddings are deferred because they would alter the
  pretrained condition architecture.
- Adaptive ordinary BV count/replenishment and directional half-FOV
  Scenario-Dreamer-compatible conditioning require separate implementation and
  training evidence.
- Adaptive HV count requires variable-target-count HV data/training; current
  V2/V3 best checkpoints are validated only for two HV outputs.
- V3 has data construction and one-batch backward evidence but no completed
  V3 training/evaluation/closed-loop result recorded in this archive.
