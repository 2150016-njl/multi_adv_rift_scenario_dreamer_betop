#!/usr/bin/env bash

export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python

export BG_UNION_R3=$HV_ROOT/rift_labeled_background_union_20260926_r3
export BG_TRIAL_DATA=$HV_ROOT/datasets/background_directional_fov64_r3_trial_lt30k
export BG_TRIAL_DIAG=$HV_ROOT/eval/background_directional_r3_trial_lt30k_diagnostics.json

export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export BG_TRIAL_CKPT=$HV_ROOT/checkpoints/background_directional_fov64_r3_trial_lt30k_gpu1

cd "$ROOT/scenario-dreamer"

"$PY" build_scenario_dreamer_background_dataset.py \
  --input_dir "$BG_UNION_R3" \
  --output_dir "$BG_TRIAL_DATA" \
  --rift_root "$ROOT/RIFT" \
  --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
  --fov 64 \
  --rear_max_x 0.0 \
  --front_min_x 0.0 \
  --target_fov_margin 2.0 \
  --max_target_agents 8 \
  --max_num_agents 61 \
  --max_town_fraction 0.35 \
  --max_topology_fraction 0.75 \
  --max_target_lane_distance 5.0 \
  --min_accepted_records 19000 \
  --diagnostics_json "$BG_TRIAL_DIAG" \
  --source carla_rift_raw_background_directional_trial_lt30k \
  --seed 42

CUDA_VISIBLE_DEVICES=1 "$PY" train_scenario_dreamer_background.py \
  --train_jsonl "$BG_TRIAL_DATA/train.jsonl" \
  --val_jsonl "$BG_TRIAL_DATA/val.jsonl" \
  --output_dir "$BG_TRIAL_CKPT" \
  --autoencoder_checkpoint "$AE_CKPT" \
  --ldm_checkpoint "$LDM_CKPT" \
  --max_target_agents 8 \
  --batch_size 16 \
  --epochs 40 \
  --lr 2e-5 \
  --decoded_state_loss_weight 0.4 \
  --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 \
  --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 \
  --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 \
  --initial_clearance_loss_weight 0.10 \
  --forward_support_loss_weight 0.10 \
  --obb_clearance_loss_weight 0.10 \
  --heading_unit_loss_weight 0.02 \
  --speed_limit_loss_weight 0.05 \
  --min_obb_axis_clearance 0.5 \
  --min_initial_clearance 2.5 \
  --fov 64 \
  --partition_boundary_x 0.0 \
  --target_fov_margin 2.0 \
  --max_num_agents 61 \
  --best_metric val_decoded_position_l1
