#!/usr/bin/env python
"""Print reproducible Phase 6 commands for high-value generation experiments.

This script intentionally prints commands instead of running them. Phase 6
requires CARLA, the RIFT runtime, Scenario Dreamer checkpoints, and often a
specific user account; printing a reviewed command plan is safer than hiding
those assumptions behind a local launcher.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="${ROOT}")
    parser.add_argument("--py", default="${PY}")
    parser.add_argument("--carla_root", default="${CARLA_ROOT}")
    parser.add_argument("--hv_root", default="${HV_ROOT}")
    parser.add_argument("--routes", default="rift/scenario/route/bench2drive220.xml")
    parser.add_argument("--smoke_routes_subset", default="1711")
    parser.add_argument("--multi_routes_subset", default="1711,1712,1713,1714,1715")
    parser.add_argument("--checkpoint_name", default="hv_route_inpaint_ldm_sparse_phase5")
    parser.add_argument("--dataset_name", default="inpainting_sparse_all_recursive")
    parser.add_argument("--eval_name", default="phase6_sparse_s1")
    parser.add_argument("--online_smoke_name", default="phase6_online_sparse_smoke")
    parser.add_argument("--online_multi_name", default="phase6_online_sparse_multi")
    parser.add_argument("--num_samples_per_condition", type=int, default=1)
    parser.add_argument("--max_eval_records", type=int, default=100)
    parser.add_argument("--rolling_interval_steps", type=int, default=50)
    parser.add_argument("--rolling_min_step", type=int, default=50)
    parser.add_argument("--generation_max_attempts", type=int, default=16)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--decoded_state_loss_weight", type=float, default=0.2)
    parser.add_argument("--physical_bounds_loss_weight", type=float, default=0.2)
    return parser.parse_args()


def line(text=""):
    print(text)


def bash_block(title, body):
    line(f"# {title}")
    line("```bash")
    line(body.strip())
    line("```")
    line()


def main():
    args = parse_args()
    root = args.root.rstrip("/")
    hv_root = args.hv_root.rstrip("/")
    py = args.py
    checkpoint_dir = f"{hv_root}/checkpoints/{args.checkpoint_name}"
    dataset_dir = f"{hv_root}/datasets/{args.dataset_name}"
    eval_dir = f"{hv_root}/eval"
    smoke_dump = f"{hv_root}/{args.online_smoke_name}"
    multi_dump = f"{hv_root}/{args.online_multi_name}"
    ae_ckpt = f"{root}/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt"
    ldm_ckpt = f"{root}/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt"

    line("# Phase 6 High-Value Generation Experiment Commands")
    line()
    line("These commands assume ROOT, PY, CARLA_ROOT, and HV_ROOT point to the server runtime paths.")
    line("They do not use RIFT labels as model inputs; RIFT/CARLA audit remains the final pass criterion.")
    line()

    bash_block(
        "0. Environment",
        f"""
export ROOT={root}
export CARLA_ROOT={args.carla_root}
export PY={py}
export HV_ROOT={hv_root}
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH
""",
    )

    bash_block(
        "1. Static Sanity Checks",
        f"""
cd $ROOT/RIFT
$PY scripts/audit_high_value_generation_pipeline.py

cd $ROOT/scenario-dreamer
$PY audit_high_value_scene_dataset.py \\
  --train_jsonl {dataset_dir}/train.jsonl \\
  --val_jsonl {dataset_dir}/val.jsonl \\
  --test_jsonl {dataset_dir}/test.jsonl \\
  --expected_mode inpainting \\
  --expected_num_generate_agents {args.num_generate_agents} \\
  --fov 64 \\
  --output_json {dataset_dir}/audit_phase6.json
""",
    )

    bash_block(
        "2. Phase 5/6 Retraining With Sparse Context And State Constraints",
        f"""
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \\
  --mode inpainting \\
  --train_jsonl {dataset_dir}/train.jsonl \\
  --val_jsonl {dataset_dir}/val.jsonl \\
  --output_dir {checkpoint_dir} \\
  --autoencoder_checkpoint {ae_ckpt} \\
  --ldm_checkpoint {ldm_ckpt} \\
  --num_generate_agents {args.num_generate_agents} \\
  --epochs 20 \\
  --batch_size 16 \\
  --lr 5e-5 \\
  --decoded_state_loss_weight {args.decoded_state_loss_weight} \\
  --decoded_position_loss_weight 1.0 \\
  --decoded_heading_loss_weight 0.5 \\
  --decoded_speed_loss_weight 0.7 \\
  --decoded_size_loss_weight 0.5 \\
  --physical_bounds_loss_weight {args.physical_bounds_loss_weight}
""",
    )

    bash_block(
        "3. Offline s1 Evaluation",
        f"""
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator.py \\
  --mode inpainting \\
  --jsonl {dataset_dir}/test.jsonl \\
  --checkpoint {checkpoint_dir}/best.ckpt \\
  --autoencoder_checkpoint {ae_ckpt} \\
  --ldm_checkpoint {ldm_ckpt} \\
  --output_json {eval_dir}/{args.eval_name}_eval_100_s{args.num_samples_per_condition}.json \\
  --output_generated_jsonl {eval_dir}/{args.eval_name}_generated_100_s{args.num_samples_per_condition}.jsonl \\
  --num_generate_agents {args.num_generate_agents} \\
  --num_samples_per_condition {args.num_samples_per_condition} \\
  --max_records {args.max_eval_records}

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator.py \\
  --mode inpainting \\
  --jsonl {dataset_dir}/test.jsonl \\
  --checkpoint {checkpoint_dir}/best.ckpt \\
  --autoencoder_checkpoint {ae_ckpt} \\
  --ldm_checkpoint {ldm_ckpt} \\
  --output_json {eval_dir}/{args.eval_name}_eval_100_s{args.num_samples_per_condition}_clamp.json \\
  --output_generated_jsonl {eval_dir}/{args.eval_name}_generated_100_s{args.num_samples_per_condition}_clamp.jsonl \\
  --num_generate_agents {args.num_generate_agents} \\
  --num_samples_per_condition {args.num_samples_per_condition} \\
  --max_records {args.max_eval_records} \\
  --clamp_generated_states
""",
    )

    common_online = f"""
  --mode eval \\
  --routes {args.routes} \\
  --ego_cfg pdm_lite.yaml \\
  --cbv_cfg rift_pluto.yaml \\
  --no_resume \\
  -rep 1 \\
  --save_bev \\
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \\
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \\
  --high_value_init_generation \\
  --hv_generation_generator_entry high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator \\
  --hv_generation_checkpoint {checkpoint_dir}/best.ckpt \\
  --hv_autoencoder_checkpoint {ae_ckpt} \\
  --hv_ldm_checkpoint {ldm_ckpt} \\
  --hv_generation_mode sparse_inpainting \\
  --hv_num_generate_agents {args.num_generate_agents} \\
  --hv_generation_hidden_slots {args.num_generate_agents} \\
  --hv_generation_max_attempts {args.generation_max_attempts} \\
  --hv_generation_rolling \\
  --hv_generation_rolling_min_step {args.rolling_min_step} \\
  --hv_generation_rolling_interval_steps {args.rolling_interval_steps} \\
  --hv_generation_rolling_min_rift_positive {args.num_generate_agents} \\
  --hv_generation_register_cbv \\
  --hv_generation_clamp_generated_states \\
  --no_hv_generation_fail_on_invalid"""

    bash_block(
        "4. Online Smoke Route",
        f"""
cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py \\
{common_online} \\
  --routes_subset {args.smoke_routes_subset} \\
  --hv_generation_dump_dir {smoke_dump}

$PY scripts/summarize_high_value_generation_online.py \\
  --dump_dir {smoke_dump} \\
  --statistics_dir $ROOT/RIFT/log/eval \\
  --output_json {smoke_dump}_summary.json
""",
    )

    bash_block(
        "5. Multi-Route Online Evaluation",
        f"""
cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py \\
{common_online} \\
  --routes_subset {args.multi_routes_subset} \\
  --hv_generation_dump_dir {multi_dump}

$PY scripts/summarize_high_value_generation_online.py \\
  --dump_dir {multi_dump} \\
  --statistics_dir $ROOT/RIFT/log/eval \\
  --output_json {multi_dump}_summary.json
""",
    )


if __name__ == "__main__":
    main()
