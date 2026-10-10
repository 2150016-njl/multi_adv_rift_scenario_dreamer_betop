#!/usr/bin/env python
"""Static audit for the high-value initial-state pipeline.

This audit checks code-structure invariants only. It does not validate CARLA
runtime behavior or model quality.
"""

import json
from pathlib import Path


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check(name, condition, detail=""):
    return {
        "name": name,
        "passed": bool(condition),
        "detail": detail,
    }


def main():
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parent

    run_py = read(root / "scripts" / "run.py")
    env_py = read(root / "rift" / "gym_carla" / "envs" / "carla_env.py")
    labeler_py = read(root / "rift" / "cbv" / "initialization" / "rift_interaction_labeler.py")
    initializer_py = read(root / "rift" / "cbv" / "initialization" / "high_value_initializer.py")
    base_cbv_py = read(root / "rift" / "cbv" / "recognition" / "base_cbv.py")
    ego_init_py = read(root / "rift" / "ego" / "__init__.py")
    cbv_planning_init_py = read(root / "rift" / "cbv" / "planning" / "__init__.py")
    map_utils_py = read(root / "rift" / "cbv" / "planning" / "pluto" / "utils" / "nuplan_map_utils.py")
    pdm_route_planner_py = read(root / "rift" / "ego" / "pdm_lite" / "privileged_route_planner.py")
    carla_runner_py = read(root / "rift" / "carla_runner.py")
    video_render_py = read(root / "rift" / "gym_carla" / "visualization" / "video_render.py")
    generator_py = read(root / "rift" / "cbv" / "initialization" / "generator.py")
    dataset_builder_py = read(repo_root / "scenario-dreamer" / "build_high_value_init_dataset.py")
    train_hv_py = read(repo_root / "scenario-dreamer" / "train_high_value_init.py")
    eval_hv_py = read(repo_root / "scenario-dreamer" / "eval_high_value_init_offline.py")
    online_summary_py = read(root / "scripts" / "summarize_high_value_init_online.py")
    sd_generator_py = read(repo_root / "scenario-dreamer" / "high_value_init" / "rift_generator.py")
    sd_model_py = read(repo_root / "scenario-dreamer" / "high_value_init" / "model.py")
    sd_adapter_py = read(repo_root / "scenario-dreamer" / "high_value_init" / "scenario_dreamer_adapter.py")

    checks = [
        check(
            "high_value_init_default_disabled",
            "parser.add_argument('--high_value_init', action='store_true', default=False)" in run_py,
        ),
        check(
            "rift_env_only_activates_hv_when_enabled",
            "if env_params.get('high_value_init', False):" in env_py,
        ),
        check(
            "topk_updated_actors_top2_cbvs_are_separate",
            "hv_init_apply_top_k" in run_py
            and "'hv_init_apply_top_k': config.get('hv_init_apply_top_k', 2)" in carla_runner_py
            and "cbv_actors = self.high_value_init_result.get(" in env_py
            and '"cbv_actor_ids": cbv_actor_ids' in initializer_py,
        ),
        check(
            "rift_hd_map_dir_uses_hd_map_subdir",
            'os.environ.get("RIFT_HD_MAP_DIR", "data/map_data/HD-Map")' in map_utils_py
            and "Set RIFT_HD_MAP_DIR" in map_utils_py
            and "CarlaMap(carla_town_name=town, map_data_dir=self.rift_hd_map_dir)" in carla_runner_py,
        ),
        check(
            "rift_speed_limits_dir_uses_speed_limits_subdir",
            'os.environ.get("RIFT_SPEED_LIMITS_DIR", os.path.join("data", "speed_limits", "Speed-Limits"))'
            in pdm_route_planner_py
            and "Set RIFT_SPEED_LIMITS_DIR" in pdm_route_planner_py,
        ),
        check(
            "runner_can_set_rift_asset_dirs_from_cli",
            "parser.add_argument('--rift_hd_map_dir'" in run_py
            and "parser.add_argument('--rift_speed_limits_dir'" in run_py
            and 'os.environ["RIFT_HD_MAP_DIR"] = self.rift_hd_map_dir' in carla_runner_py
            and 'os.environ["RIFT_SPEED_LIMITS_DIR"] = self.rift_speed_limits_dir' in carla_runner_py,
        ),
        check(
            "collect_hv_init_mode_uses_rift_runner",
            "'collect_hv_init': 'rift/scenario/route/bench2drive220.xml'" in run_py
            and "parser.add_argument('--hv_init_collect_only'" in run_py
            and "def collect_hv_init(" in carla_runner_py
            and "self.collect_hv_init(data_loader)" in carla_runner_py,
        ),
        check(
            "collect_only_does_not_load_generator_or_register_cbv",
            "self.generator = None if self.collect_only else build_high_value_generator(config)" in initializer_py
            and 'if self.collect_only:' in initializer_py
            and '"collect_only": True' in initializer_py
            and "not self.high_value_init_result.get('collect_only', False)" in env_py,
        ),
        check(
            "save_bev_does_not_require_sensor_render",
            "parser.add_argument('--save_bev'" in run_py
            and "self.need_sensor_render = config['render'] or self.mode == 'collect_data'" in carla_runner_py
            and "self.need_video_render = self.need_sensor_render or config.get('save_bev', False)" in carla_runner_py,
        ),
        check(
            "bev_video_can_be_saved_without_sensor_images",
            "sensor_imgs = self.sensor_image_dict.get(index_1)" in video_render_py
            and "np.rot90(BEV_img, k=1)" in video_render_py,
        ),
        check(
            "high_value_init_saves_viz",
            "def _dump_viz(" in initializer_py and "matplotlib.use(\"Agg\")" in initializer_py,
        ),
        check(
            "rift_validation_default_is_audit",
            "hv_init_validation_mode', type=str, default='audit'" in run_py
            and 'self.validation_mode = str(config.get("hv_init_validation_mode", "audit"))' in initializer_py,
        ),
        check(
            "no_rift_fallback_after_hv_attempt",
            "if not high_value_init_attempted and not high_value_initialized:" in env_py,
        ),
        check(
            "model_scores_all_actors_without_delta_head",
            "target_mask_logits" in sd_model_py
            and "actor_scores" in sd_model_py
            and "state_delta" not in sd_model_py
            and "delta_loss" not in sd_model_py,
        ),
        check(
            "scenario_dreamer_generator_uses_ldm_for_state_generation",
            "ScenarioDreamerLDM.load_from_checkpoint" in sd_generator_py
            and "hv_model_ldm_checkpoint" in run_py
            and 'self.state_source = str(self.config.get("hv_model_state_source", "ldm_inpainting"))' in sd_generator_py
            and "_sample_lane_conditioned_scene" in sd_generator_py,
        ),
        check(
            "scenario_dreamer_default_state_generation_is_inpainting",
            "default='ldm_inpainting'" in run_py
            and "choices=['ldm_inpainting', 'ldm_lane_conditioned', 'ldm', 'identity']" in run_py
            and "'hv_model_state_source': config.get('hv_model_state_source', 'ldm_inpainting')" in carla_runner_py,
        ),
        check(
            "ldm_inpainting_conditions_on_existing_actor_and_lane_latents",
            "_sample_inpainted_scene" in sd_generator_py
            and 'ldm_data["agent"].latents = agent_latents' in sd_generator_py
            and 'ldm_data["lane"].latents = lane_latents' in sd_generator_py
            and 'ldm_data["agent"].mask = agent_mask' in sd_generator_py
            and 'ldm_data["lane"].mask = torch.ones' in sd_generator_py
            and "_sample_normalized_latents" in sd_generator_py
            and 'generated_data = self._run_ldm(ldm_data, mode="inpainting")' in sd_generator_py,
        ),
        check(
            "inpainting_keeps_selected_actor_identity",
            "_reorder_scene_conditioning_for_ldm" in sd_generator_py
            and "_scenario_dreamer_agent_order" in sd_generator_py
            and "sorted_actor_ids" in sd_generator_py
            and "actors_by_id[int(actor_id)]" in sd_generator_py
            and "_scene_with_inpainted_actor_states" in sd_generator_py,
        ),
        check(
            "topk_mask_is_selected_from_original_scene_before_generation",
            "original_logits, original_scores, actor_ids = self._score_scene(scene)" in sd_generator_py
            and "top_indices = torch.topk(original_logits, k=top_k)" in sd_generator_py
            and "selected_actor_ids = [actor_ids[idx] for idx in top_indices]" in sd_generator_py
            and "objective = float(torch.sum(logits[top_indices])" in sd_generator_py,
        ),
        check(
            "lane_conditioned_ldm_uses_scenario_dreamer_reorder_indices",
            "_reorder_lane_conditioning_for_ldm" in sd_generator_py
            and "from utils.data_helpers import reorder_indices" in sd_generator_py
            and "dataset=\"nuplan\"" in sd_generator_py,
        ),
        check(
            "rift_labels_are_not_model_inputs",
            "_build_interaction_features" not in sd_generator_py
            and "interaction_features" not in train_hv_py
            and "interaction_features" not in eval_hv_py
            and "interaction_labels" not in sd_model_py,
        ),
        check(
            "dataset_builder_does_not_perturb_or_restore",
            "rift_hv_init_actor_score_v2" in dataset_builder_py
            and "target_scores" in dataset_builder_py
            and "target_state_delta" not in dataset_builder_py
            and "perturb_" not in dataset_builder_py
            and "restore_logged_state" not in dataset_builder_py,
        ),
        check(
            "training_optimizes_mask_and_score_only",
            "score_weight" in train_hv_py
            and "target_scores" in train_hv_py
            and "delta_weight" not in train_hv_py
            and "target_state_delta" not in train_hv_py,
        ),
        check(
            "mask_loss_handles_class_imbalance",
            "mask_pos_weight" in sd_model_py
            and "negatives / positives" in sd_model_py
            and '"--mask_pos_weight"' in train_hv_py,
        ),
        check(
            "offline_eval_reports_ranking_and_score_only",
            "exact_topk" in eval_hv_py
            and "recall_at_k" in eval_hv_py
            and "score_mae" in eval_hv_py
            and "selected_delta_l1" not in eval_hv_py,
        ),
        check(
            "rift_scene_exports_map_lanes_for_scenario_dreamer_adapter",
            '"map_lanes": self._serialize_map_lanes(ego_vehicle)' in initializer_py
            and "CarlaDataProvider.get_map_api()" in initializer_py
            and "query_proximal_map_data" in initializer_py,
        ),
        check(
            "scenario_dreamer_adapter_builds_native_graph_data",
            "class RiftToScenarioDreamerAdapter" in sd_adapter_py
            and "ScenarioDreamerData()" in sd_adapter_py
            and "get_edge_index_complete_graph" in sd_adapter_py
            and "normalize_scene(" in sd_adapter_py,
        ),
        check(
            "scenario_dreamer_adapter_config_is_cli_consistent",
            "parser.add_argument('--hv_sd_fov'" in run_py
            and "'hv_sd_fov': config.get('hv_sd_fov', 64.0)" in carla_runner_py
            and '"--hv_sd_fov"' in train_hv_py
            and '"--hv_sd_fov"' in eval_hv_py
            and 'self.cfg.get("hv_sd_fov"' in sd_adapter_py,
        ),
        check(
            "rift_scene_actor_order_matches_scenario_dreamer_distance_order",
            "_ordered_actor_ids_by_ego_distance" in initializer_py
            and "ego_location.distance(actor_location)" in initializer_py
            and "labels_by_actor_id" in initializer_py,
        ),
        check(
            "scenario_dreamer_wrapper_returns_scores_masks_and_updates",
            '"target_mask": target_mask' in sd_generator_py
            and '"actor_scores": best["scores"]' in sd_generator_py
            and '"state_updates": state_updates' in sd_generator_py,
        ),
        check(
            "rift_labeler_uses_original_candidate_logic",
            "get_CBV_candidates(" in labeler_py
            and "get_rift_targetable_actors" in labeler_py
            and "CarlaDataProvider.get_actors()" in labeler_py,
        ),
        check(
            "rift_labeler_does_not_reimplement_trace_route",
            "trace_route" not in labeler_py and "is_within_distance_ahead" not in labeler_py,
        ),
        check(
            "rift_candidate_logic_only_adds_diagnostics",
            "self.last_candidate_debug" in base_cbv_py
            and "debug['rejection_counts']" in base_cbv_py
            and "return [vehicle for _, vehicle in valid_candidates]" in base_cbv_py,
        ),
        check(
            "debug_generator_is_explicit_smoke_only",
            "Smoke-test generator. Do not use it for reported experiments." in generator_py
            and "Use --hv_init_debug_generator only for pipeline smoke tests." in sd_generator_py,
        ),
        check(
            "production_interface_requires_absolute_state_updates",
            "state_deltas" not in initializer_py
            and '"state_updates": state_updates' in generator_py
            and '"state_updates": state_updates' in sd_generator_py
            and "missing absolute state update" in initializer_py,
        ),
        check(
            "rift_audit_validates_cbv_top2_not_all_topk",
            "def _validate_rift_result(self, cbv_actor_ids" in initializer_py
            and "CBV actors failed RIFT interaction validation" in initializer_py
            and "extra_ids = positive_ids - set(cbv_actor_ids)" in initializer_py,
        ),
        check(
            "invalid_model_proposal_is_dumped_not_fallbacked",
            "expected at least {self.cbv_count}" in initializer_py
            and "self._dump_attempt(" in initializer_py
            and "return last_failure_result or self._failure" in initializer_py,
        ),
        check(
            "base_cbv_exports_interaction_meta",
            "interaction_meta" in base_cbv_py and "'distance_diff': inter_distance_diff" in base_cbv_py,
        ),
        check(
            "dynamic_cbv_locked_after_success",
            "should_lock_dynamic_cbv" in env_py and "high_value_init_result" in env_py,
        ),
        check(
            "online_summary_separates_selected_and_cbv_counts",
            "selected_count" in online_summary_py
            and "cbv_count" in online_summary_py
            and "generation_sources" in online_summary_py,
        ),
        check(
            "online_summary_can_merge_rift_route_and_collision_metrics",
            "statistics_json" in online_summary_py
            and "route_completion" in online_summary_py
            and "collision_routes" in online_summary_py
            and "cbv_collision_count" in online_summary_py,
        ),
        check(
            "ego_policy_registry_is_lazy",
            "LazyPolicyRegistry" in ego_init_py
            and "from rift.ego.b2d.e2e_agent import" not in ego_init_py,
        ),
        check(
            "cbv_policy_registry_is_lazy",
            "LazyPolicyRegistry" in cbv_planning_init_py
            and "from rift.cbv.planning.fine_tuner.rlft.grpo_pluto.grpo_pluto import" not in cbv_planning_init_py,
        ),
    ]

    payload = {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
    }
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
