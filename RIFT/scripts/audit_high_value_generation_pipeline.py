#!/usr/bin/env python
"""Static audit for the high-value scene-generation pipeline."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def check(name, passed, detail=""):
    return {"name": name, "passed": bool(passed), "detail": detail}


def main():
    checks = []
    run_py = read("RIFT/scripts/run.py")
    runner = read("RIFT/rift/carla_runner.py")
    rift_dataset = read("scenario-dreamer/high_value_init/rift_dataset.py")
    init_py = read("scenario-dreamer/high_value_init/__init__.py")
    scene_generator = read("scenario-dreamer/high_value_init/scene_generator.py")
    hv_ldm = read("scenario-dreamer/high_value_init/high_value_ldm.py")
    dataset_builder = read("scenario-dreamer/build_high_value_scene_dataset.py")
    dataset_audit = read("scenario-dreamer/audit_high_value_scene_dataset.py")
    train_script = read("scenario-dreamer/train_high_value_scene_generator.py")
    eval_script = read("scenario-dreamer/eval_high_value_scene_generator.py")
    online = read("RIFT/rift/cbv/initialization/hv_online_generation.py")
    snapshotter = read("RIFT/rift/cbv/initialization/hv_scene_snapshot.py")
    carla_env = read("RIFT/rift/gym_carla/envs/carla_env.py")
    online_summary = read("RIFT/scripts/summarize_high_value_generation_online.py")
    online_compare = read("RIFT/scripts/compare_hv_online_runs.py")
    matched_online_launcher = read("RIFT/scripts/run_hv_matched_online_eval.sh")
    phase6_commands = read("RIFT/scripts/print_high_value_generation_phase6_commands.py")
    snapshot_summary = read("RIFT/scripts/summarize_high_value_labeled_snapshots.py")
    v4_builder = read("scenario-dreamer/build_high_value_scene_dataset_v4.py")
    v4_dataset_audit = read("scenario-dreamer/audit_high_value_scene_dataset_v4.py")
    v4_train = read("scenario-dreamer/train_high_value_scene_generator_v4a.py")
    v4_eval = read("scenario-dreamer/eval_high_value_scene_generator_v4a.py")
    v4_generator = read("scenario-dreamer/high_value_init/scene_generator_v4a.py")
    v3_generator = read("scenario-dreamer/high_value_init/scene_generator_v3.py")
    v2_eval = read("scenario-dreamer/eval_high_value_scene_generator_v2.py")
    v3_eval = read("scenario-dreamer/eval_high_value_scene_generator_v3.py")
    eval_geometry = read("scenario-dreamer/high_value_init/evaluation_geometry.py")
    frozen_eval_audit = read("scenario-dreamer/audit_frozen_hv_evaluation.py")
    v4_collection_planner = read("RIFT/scripts/plan_hv_v4a_collection.py")
    v4_collection_manifest_audit = read("RIFT/scripts/audit_hv_v4a_collection_manifest.py")
    v4_collection_launch = read("RIFT/scripts/render_hv_v4a_collection_launch.py")
    v4_collection_launch_audit = read("RIFT/scripts/audit_hv_v4a_collection_launch.py")
    v4_geometry_ldm = read("scenario-dreamer/high_value_init/high_value_ldm_v4a.py")
    v4_geometry_train = read("scenario-dreamer/train_high_value_scene_generator_v4a_geometry.py")
    v4_stage_audit = read("scenario-dreamer/audit_v4a_training_stages.py")
    background_builder = read("scenario-dreamer/build_scenario_dreamer_background_dataset.py")
    background_dataset_audit = read("scenario-dreamer/audit_scenario_dreamer_background_dataset.py")
    background_train = read("scenario-dreamer/train_scenario_dreamer_background.py")
    background_eval = read("scenario-dreamer/eval_scenario_dreamer_background.py")
    raw_generation_analysis = read("scenario-dreamer/analyze_raw_generation_rows.py")
    target_geometry_audit = read("scenario-dreamer/audit_target_geometry.py")
    background_controller = read("scenario-dreamer/high_value_init/background_controller.py")
    background_generator = read("scenario-dreamer/high_value_init/scene_generator_background_v1.py")
    background_collection_planner = read("RIFT/scripts/plan_scenario_dreamer_background_collection.py")
    background_collection_audit = read("RIFT/scripts/audit_scenario_dreamer_background_collection_manifest.py")
    background_collection_launch = read("RIFT/scripts/render_background_collection_launch.py")
    background_collection_launch_audit = read("RIFT/scripts/audit_background_collection_launch.py")
    future_outcomes = read("RIFT/rift/cbv/initialization/hv_future_outcome.py")
    future_outcomes_audit = read("RIFT/scripts/audit_hv_future_outcomes.py")
    future_outcomes_collector = read("RIFT/scripts/collect_hv_future_outcomes.py")
    future_outcomes_merge = read("RIFT/scripts/merge_hv_future_outcomes.py")
    ldm = read("scenario-dreamer/nn_modules/ldm.py")
    dit = read("scenario-dreamer/nn_modules/dit.py")

    checks.append(check(
        "package_init_does_not_auto_import_legacy_mask_delta",
        "HighValueInitHead" not in init_py and "rift_generator" not in init_py,
    ))
    checks.append(check(
        "online_generation_does_not_register_cbv",
        "add_CBV" not in online and "_activate_CBV" not in online and "selected_actor_ids" not in online,
    ))
    checks.append(check(
        "online_generation_spawns_anonymous_actors",
        "request_new_actor" in online and "generated_hv" in online and "state_updates" not in online,
    ))
    checks.append(check(
        "scene_generator_does_not_read_rift_labels",
        "snapshot.get(\"rift_labels\"" not in scene_generator and "interaction_labels" not in scene_generator,
    ))
    checks.append(check(
        "scene_generator_outputs_anonymous_states",
        "self.model.generate" in scene_generator
        and "generated_agent_states" in hv_ldm
        and "selected_actor_ids" not in scene_generator
        and "cbv_actor_ids" not in scene_generator
        and "target_mask" not in scene_generator
        and "actor_scores" not in scene_generator,
    ))
    checks.append(check(
        "dataset_builder_rejects_selector_delta_fields",
        "state_generation_schema_is_clean" in dataset_builder,
    ))
    checks.append(check(
        "dataset_builder_does_not_emit_actor_identity_targets",
        "removed_positive_actor_ids_for_trace_only" not in dataset_builder
        and "target_actor_ids" not in dataset_builder
        and '"cbv_actor_ids"' not in dataset_builder
        and '"selected_actor_ids"' not in dataset_builder,
    ))
    checks.append(check(
        "dataset_builder_filters_to_scenario_dreamer_fov",
        "--fov" in dataset_builder
        and "half_fov" in dataset_builder
        and "abs(state[0]) > half_fov" in dataset_builder,
    ))
    checks.append(check(
        "dataset_audit_checks_fov_capacity_and_split_leakage",
        "target_out_of_fov" in dataset_audit
        and "scenario_dreamer_agent_capacity_exceeded" in dataset_audit
        and "episode_split_leakage" in dataset_audit,
    ))
    checks.append(check(
        "rift_dataset_lets_pyg_build_batches",
        'data["agent"].batch = torch.zeros' not in rift_dataset
        and 'data["lane"].batch = torch.zeros' not in rift_dataset
        and 'data["route"].batch = torch.zeros' not in rift_dataset,
    ))
    checks.append(check(
        "ldm_generation_uses_original_gaussian_noise_loop",
        "x_agent = torch.randn" in ldm and "mode == 'inpainting'" in ldm,
    ))
    checks.append(check(
        "high_value_ldm_calls_original_ldm_inpainting",
        "self.ldm.forward(data, mode=\"inpainting\")" in hv_ldm,
    ))
    checks.append(check(
        "high_value_ldm_reuses_original_modules_directly",
        "from nn_modules.autoencoder import AutoEncoder" in hv_ldm
        and "from nn_modules.ldm import LDM" in hv_ldm
        and "ScenarioDreamerLDM" not in hv_ldm,
    ))
    checks.append(check(
        "high_value_ldm_has_no_selector_or_delta_outputs",
        "target_mask_logits" not in hv_ldm
        and "actor_scores" not in hv_ldm
        and "state_delta" not in hv_ldm
        and "selected_actor_ids" not in hv_ldm
        and "cbv_actor_ids" not in hv_ldm,
    ))
    checks.append(check(
        "route_condition_is_injected_into_dit",
        "use_route_condition" in dit and "route_condition_encoder" in dit and "route_condition" in dit,
    ))
    checks.append(check(
        "run_exposes_generation_pipeline_separately",
        "--high_value_init_generation" in run_py
        and "collect_hv_generation" in run_py
        and "--high_value_init and --high_value_init_generation are separate pipelines" in run_py,
    ))
    checks.append(check(
        "runner_has_collection_and_online_generation_paths",
        "def collect_hv_generation" in runner
        and "high_value_init_generation" in runner
        and "High-value scene generation snapshot collection mode" in runner,
    ))
    checks.append(check(
        "new_training_does_not_implement_rift_score_aux",
        "RIFT score auxiliary loss is intentionally not implemented" in train_script,
    ))
    checks.append(check(
        "new_training_accepts_documented_freeze_autoencoder_flag",
        "--freeze_autoencoder" in train_script,
    ))
    checks.append(check(
        "new_training_logs_decoded_state_metrics",
        "--decoded_state_loss_weight" in train_script
        and "val_decoded_position_l1" in train_script
        and "train_decoded_position_l1" in train_script,
    ))
    checks.append(check(
        "phase0_sparse_online_generation_mode_is_exposed",
        "sparse_inpainting" in run_py
        and "self.sparse_inpainting" in online
        and (
            "model_mode = \"inpainting\" if mode == \"sparse_inpainting\"" in scene_generator
            or "mode in (\"sparse_inpainting\", \"full_fov_inpainting\")" in scene_generator
        ),
    ))
    checks.append(check(
        "phase1_sparse_inpainting_does_not_require_hidden_slots",
        "def _should_hide_partial_context" in online
        and (
            "not self.sparse_inpainting" in online
            or "not (self.sparse_inpainting or self.full_fov_inpainting)" in online
        )
        and "def _requires_hidden_slots" in online,
    ))
    checks.append(check(
        "phase2_failure_diagnostics_are_stage_structured",
        "failure_stage" in online
        and "hidden_context_debug" in online
        and "spawn_failures" in online
        and "failure_stage_counts" in online_summary,
    ))
    checks.append(check(
        "phase3_generated_only_audit_and_cbv_registration_exist",
        "generated_rift_positive_actor_ids" in online
        and "num_generated_rift_positive_agents" in online
        and "_register_high_value_generated_cbvs" in carla_env
        and "cbv_registration_count" in online_summary,
    ))
    checks.append(check(
        "phase4_sparse_context_dataset_and_audit_exist",
        "--sparse_context_copies" in dataset_builder
        and "sparse_random_observed" in dataset_builder
        and "low_observed_count_leq2_rate" in dataset_audit
        and "split_context_summary" in dataset_audit,
    ))
    checks.append(check(
        "phase5_state_loss_and_physical_metrics_exist",
        "--physical_bounds_loss_weight" in train_script
        and "decoded_size_l1" in train_script
        and "_physical_bounds_loss" in hv_ldm
        and "best_of_n_size_l1" in eval_script
        and "physical_invalid_rate" in eval_script,
    ))
    checks.append(check(
        "phase6_commands_cover_offline_online_and_summary",
        "Offline s1 Evaluation" in phase6_commands
        and "Online Smoke Route" in phase6_commands
        and "Multi-Route Online Evaluation" in phase6_commands
        and "summarize_high_value_generation_online.py" in phase6_commands,
    ))
    checks.append(check(
        "phase6_online_summary_reports_required_rift_statistics",
        "route_completion" in online_summary
        and "driving_score" in online_summary
        and "collision_routes_ratio" in online_summary
        and "generated_rift_positive_agent_count" in online_summary
        and "cbv_registration_count" in online_summary,
    ))
    checks.append(check(
        "snapshot_summary_reports_rift_positive_distribution",
        "positive_scene_ratio" in snapshot_summary
        and "positive_agents_per_scene" in snapshot_summary
        and "examples_missing_labels" in snapshot_summary,
    ))
    checks.append(check(
        "v2_v3_frozen_evaluation_is_seeded_raw_and_geometry_auditable",
        "--seed" in v2_eval
        and "--seed" in v3_eval
        and "raw_state_postprocessing" in v2_eval
        and "raw_state_postprocessing" in v3_eval
        and "checkpoint_sha256" in v2_eval
        and "checkpoint_sha256" in v3_eval
        and "obb_signed_axis_clearance" in eval_geometry
        and "associated_lane_distance_mean" in eval_geometry
        and "not_raw_single_sample" in frozen_eval_audit,
    ))
    checks.append(check(
        "v4a_targeted_collection_planner_preserves_forward_raw_and_town_contracts",
        "required_new_non_dominant_accepted_records" in v4_collection_planner
        and "dominant_town_policy" in v4_collection_planner
        and "min_forward_rift_positive" in v4_collection_planner
        and "_build_record" in v4_collection_planner
        and "raw_state_postprocessing" in v4_collection_planner
        and "duplicate_route_across_workers" in v4_collection_manifest_audit
        and "dominant_town_route_selected" in v4_collection_manifest_audit,
    ))
    checks.append(check(
        "v4a_planner_exposes_one_pass_quota_risk_without_claiming_collection_success",
        "quota_shortfall_against_one_historical_pass_by_town" in v4_collection_planner
        and "one_historical_pass_meets_all_town_quotas" in v4_collection_planner
        and "one_pass_quota_shortfall_by_town" in v4_collection_manifest_audit,
    ))
    checks.append(check(
        "v4a_launcher_isolates_twelve_collection_jobs_before_starting_them",
        "--processes_per_gpu" in v4_collection_launch
        and "--base_tm_port" in v4_collection_launch
        and "mkdir -p" in v4_collection_launch
        and "refusing to overwrite launch artifacts" in v4_collection_launch
        and "duplicate_route" in v4_collection_launch_audit
        and "port_collision" in v4_collection_launch_audit,
    ))
    checks.append(check(
        "future_outcome_sidecar_is_current_anchor_only_and_not_online_control",
        "future_outcome_supervision_for_current_time_anchor_only" in future_outcomes
        and "future_positions_used_as_spawn_state" in future_outcomes
        and "counterfactual_ego_completion_impact_available" in future_outcomes
        and "normal_cbv_activated" in future_outcomes
        and "route_corridor_overlap_any" in future_outcomes
        and "collect_hv_future_outcome" in runner
        and "collect_hv_future_outcome" in run_py
        and "--mode\", \"collect_hv_future_outcome\"" in future_outcomes_collector
        and "missing_current_anchor" in future_outcomes_audit,
    ))
    checks.append(check(
        "future_outcome_merger_preserves_sidecar_provenance_and_refuses_overwrite",
        "sidecar_source_path" in future_outcomes_merge
        and "duplicate_rows_skipped" in future_outcomes_merge
        and "refusing to overwrite" in future_outcomes_merge,
    ))
    checks.append(check(
        "v4a_a0_a1_a2_losses_are_isolated_without_architecture_change",
        "RouteConditionedHighValueLDMV3" in v4_geometry_ldm
        and "forward_support_loss" in v4_geometry_ldm
        and "obb_clearance_loss" in v4_geometry_ldm
        and "_unnormalize_agents_for_v4a_losses" in v4_geometry_ldm
        and "a1_forward_support" in v4_geometry_train
        and "a2_obb_clearance" in v4_geometry_train
        and "parent_checkpoint_sha256" in v4_geometry_train
        and "a1_parent_sha256_mismatch" in v4_stage_audit
        and "a2_parent_sha256_mismatch" in v4_stage_audit,
    ))
    checks.append(check(
        "v4a_dataset_has_full_fov_forward_contract_without_selector_outputs",
        "full_fov_non_target" in v4_builder
        and "forward_min_x" in v4_builder
        and "agent_capacity_rejected" in v4_builder
        and "min_accepted_records" in v4_builder
        and "accepted_record_count_below_minimum" in v4_builder
        and "state_generation_schema_is_clean" in v4_builder
        and '"target_actor_ids"' not in v4_builder,
    ))
    checks.append(check(
        "v4a_dataset_audit_checks_forward_fov_capacity_and_episode_split",
        "target_not_in_forward_sector" in v4_dataset_audit
        and "condition_fov_contract_mismatch" in v4_dataset_audit
        and "capacity_contract_mismatch" in v4_dataset_audit
        and "episode_split_leakage" in v4_dataset_audit,
    ))
    checks.append(check(
        "v4a_training_reuses_v3_architecture_and_rejects_context_truncation",
        "RouteConditionedHighValueLDMV3" in v4_train
        and "reject_observed_over_capacity" in v4_train
        and "fixed validation noise stream" in v4_train
        and "V4-A preserves the validated fixed-two-target task" in v4_train,
    ))
    checks.append(check(
        "v4a_raw_evaluation_has_no_postprocessing_or_online_oracle_policy",
        "none_raw_model_output" in v4_eval
        and "clamp_generated_states=False" in v4_eval
        and "best_of_n_position_l2_oracle_not_online_policy" in v4_eval
        and "associated_lane_geometry" in v4_eval,
    ))
    checks.append(check(
        "v4a_online_entry_requires_full_fov_raw_contract",
        "full_fov_inpainting" in v4_generator
        and "requires a checkpoint written by train_high_value_scene_generator_v4a.py" in v4_generator
        and "No context truncation is permitted" in v4_generator
        and "raw_state_postprocessing" in v4_generator
        and "register_cbv_from_audit" in v4_generator,
    ))
    checks.append(check(
        "v4a_online_mode_is_cli_visible_and_does_not_hide_context",
        "full_fov_inpainting" in run_py
        and "hv_generation_forward_min_x" in runner
        and "self.full_fov_inpainting" in online
        and "not (self.sparse_inpainting or self.full_fov_inpainting)" in online,
    ))
    checks.append(check(
        "rift_dataset_supports_opt_in_no_truncation_policy",
        "reject_observed_over_capacity" in rift_dataset
        and "refusing to truncate context" in rift_dataset,
    ))
    checks.append(check(
        "v4a_collection_and_snapshot_summary_measure_forward_candidate_coverage",
        "num_forward_rift_positive_agents" in snapshotter
        and "hv_generation_collect_min_forward_rift_positive" in runner
        and "forward_positive_scene_ratio" in snapshot_summary
        and "forward_positive_count_hist" in snapshot_summary,
    ))
    checks.append(check(
        "online_summary_reports_v4a_raw_forward_diagnostics",
        "v4a_raw_contract" in online_summary
        and "raw_generated_forward_violation_count" in online_summary
        and "state_postprocessing" in online_summary,
    ))
    checks.append(check(
        "post_spawn_fidelity_is_observation_only_and_excludes_terrain_z_from_model_error",
        "def _post_spawn_fidelity" in online
        and "post_spawn_xy_error_m" in online
        and "post_spawn_yaw_error_deg" in online
        and "it never feeds back into sampling or actor control" in online
        and "post_spawn_raw_state_fidelity" in online_summary,
    ))
    checks.append(check(
        "matched_online_comparison_is_analysis_only_and_reports_source_lifecycle",
        "never launches\nCARLA" in online_compare
        and "orange-to-green" in online_compare
        and "summarize_lifecycle" in online_compare
        and "paired_against_baseline" in online_compare,
    ))
    checks.append(check(
        "matched_online_launcher_is_foreground_raw_and_normal_rift_only",
        "run_no_hv" in matched_online_launcher
        and "ScenarioDreamerHighValueSceneGeneratorV3" in matched_online_launcher
        and "ScenarioDreamerHighValueSceneGeneratorV4A" in matched_online_launcher
        and "--hv_generation_lifecycle_telemetry" in matched_online_launcher
        and "--hv_generation_register_cbv" not in matched_online_launcher
        and "--hv_generation_rolling_force_update" not in matched_online_launcher
        and "--hv_generation_clamp_generated_states" not in matched_online_launcher
        and "compare_hv_online_runs.py" in matched_online_launcher,
    ))
    checks.append(check(
        "offline_joint_feasibility_analysis_does_not_select_or_rewrite_samples",
        "joint_feasible" in raw_generation_analysis
        and "no_clamp_no_rerank_no_rejection" in raw_generation_analysis
        and "only reads" in raw_generation_analysis,
    ))
    checks.append(check(
        "ground_truth_geometry_audit_precedes_lane_loss_tuning",
        "ground-truth targets" in target_geometry_audit
        and "associated_lane_corridor_failure_count" in target_geometry_audit
        and "no_dataset_or_state_mutation" in target_geometry_audit,
    ))
    checks.append(check(
        "v3_online_entry_rejects_non_v3_checkpoint_contracts",
        "def _require_v3_checkpoint" in v3_generator
        and "train_high_value_scene_generator_v3.py" in v3_generator
        and "fixed-two-target contract" in v3_generator
        and "hv_snapshot_map_coordinate_mode right_handed" in v3_generator,
    ))
    checks.append(check(
        "background_dataset_is_independent_directional_current_scene_task",
        "current_front_scene_completion_not_future_prediction" in background_builder
        and "def normalize_raw_snapshot" in background_builder
        and "base.normalize_snapshot" not in background_builder
        and "rear_x_lt_boundary" in background_builder
        and "front_x_gte_boundary" in background_builder
        and "max_town_fraction" in background_builder
        and "max_topology_fraction" in background_builder
        and "min_accepted_records" in background_builder
        and "accepted_record_count_below_minimum" in background_builder
        and "Collect targeted non-dominant ordinary-traffic routes" in background_builder
        and "rift_labels_used_as_input" not in background_builder
        and "train_only" in background_builder
        and "FORBIDDEN_FIELD_NAMES" in background_dataset_audit
        and '"rift_label"' in background_dataset_audit,
    ))
    checks.append(check(
        "background_model_preserves_architecture_and_raw_feasibility_audit",
        "unchanged_v3_scenario_dreamer_ldm_losses_only" in background_train
        and "obb_clearance_loss_weight" in background_train
        and "heading_unit_loss_weight" in background_train
        and "speed_limit_loss_weight" in background_train
        and "raw_state_postprocessing" in background_eval
        and "best_of_n_position_l2_oracle" in background_eval,
    ))
    checks.append(check(
        "background_controller_is_k_missing_count_only_and_online_entry_is_gated",
        "k_missing" in background_controller
        and "side_effects" in background_controller
        and "graph_capacity_rejected_no_context_truncation" in background_controller
        and "not wired into the legacy HV initializer yet" in background_generator
        and "requires a checkpoint written by train_scenario_dreamer_background.py" in background_generator
        and "raw_state_postprocessing" in background_generator,
    ))
    checks.append(check(
        "background_collection_plan_targets_tens_of_thousands_of_accepted_records_with_route_seed_diversity",
        "none_all_raw_traffic_snapshots" in background_collection_planner
        and "no_new_generic_collection" in background_collection_planner
        and "target_total_accepted_records" in background_collection_planner
        and "min_accepted_records_per_town" in background_collection_planner
        and "min_routes_per_town" in background_collection_planner
        and "recommended_seed_rounds" in background_collection_planner
        and "A rate estimate is not an acceptance guarantee" in background_collection_planner
        and "duplicate_route_across_workers" in background_collection_audit,
    ))
    checks.append(check(
        "background_launcher_uses_unfiltered_raw_traffic_and_riftuser_isolation",
        "--min_forward_rift_positive\", \"0\"" in background_collection_launch
        and "runuser -u" in background_collection_launch
        and "mkdir -p" in background_collection_launch
        and "CARLA is launched as riftuser" in background_collection_launch
        and "background_task_or_filter" in background_collection_launch_audit,
    ))

    passed = all(item["passed"] for item in checks)
    print(json.dumps({"passed": passed, "checks": checks}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
