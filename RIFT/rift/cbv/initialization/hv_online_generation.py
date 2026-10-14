#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""Reset-time landing of generated high-value initial states in CARLA.

The generator returns anonymous local-frame vehicle states. This module spawns
those vehicles into CARLA, records what happened, and optionally audits the
result with RIFT's original interaction logic. It does not choose CBVs itself;
the environment may later register RIFT-positive generated actors as CBVs when
that experiment flag is enabled. It never rewrites existing actor states.
"""

import importlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import carla

from rift.cbv.initialization.hv_scene_snapshot import HighValueSceneSnapshotter
from rift.scenario.tools.carla_data_provider import CarlaDataProvider
from rift.scenario.tools.exception import SpawnRuntimeError


class ExternalGeneratedSceneModel:
    """Load ``module:Class`` that exposes ``generate(snapshot: dict)``."""

    def __init__(self, entry_point: str, checkpoint_path: str, device: str, config: Dict):
        if not entry_point or ":" not in entry_point:
            raise ValueError("hv_generation_generator_entry must be 'module:ClassName'")
        module_name, class_name = entry_point.split(":", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        self.impl = cls(checkpoint_path=checkpoint_path, device=device, config=config)

    def generate(self, snapshot: Dict) -> Dict:
        if hasattr(self.impl, "generate"):
            return self.impl.generate(snapshot)
        if callable(self.impl):
            return self.impl(snapshot)
        raise TypeError("loaded high-value scene generator is not callable")


class HighValueGeneratedSceneInitializer:
    """Spawn Scenario Dreamer generated anonymous agents at reset time."""

    def __init__(self, config: Dict, world, logger=None):
        self.config = config
        self.world = world
        self.logger = logger
        self.enabled = bool(config.get("high_value_init_generation", False))
        self.dump_dir = config.get("hv_generation_dump_dir") or config.get("hv_dump_dir")
        self.actor_model = config.get("hv_generation_actor_model", "vehicle.tesla.model3")
        self.actor_role = config.get("hv_generation_actor_role", "generated_hv")
        self.background_actor_role = config.get("sd_background_actor_role", "generated_sd_background")
        self.sd_background_source = str(config.get("sd_background_source", "carla"))
        self.sd_background_replace_existing = bool(config.get("sd_background_replace_existing", False))
        self.sd_background_replace_radius = float(config.get("sd_background_replace_radius", 64.0))
        self.sd_background_model_only_traffic = bool(config.get("sd_background_model_only_traffic", False))
        if self.sd_background_source not in ("carla", "scenario_dreamer"):
            raise ValueError("sd_background_source must be 'carla' or 'scenario_dreamer'")
        if self.sd_background_model_only_traffic and self.sd_background_source != "scenario_dreamer":
            raise ValueError("model-only traffic requires --sd_background_source scenario_dreamer")
        self.autopilot = bool(config.get("hv_generation_autopilot", True))
        self.fail_on_invalid = bool(config.get("hv_generation_fail_on_invalid", True))
        if self.enabled and bool(config.get("hv_generation_clamp_generated_states", False)):
            raise ValueError("strict HV forbids generated-state clipping")
        self.num_generate_agents = int(config.get("hv_num_generate_agents", 2))
        # Fail closed: do not silently accept legacy retry configurations.
        # scripts/run.py currently defaults to eight; pass 1 explicitly.
        self.requested_generation_max_attempts = int(config.get("hv_generation_max_attempts", 1))
        if self.enabled and self.requested_generation_max_attempts != 1:
            raise ValueError("strict HV requires hv_generation_max_attempts=1")
        self.generation_max_attempts = 1
        self._generation_consumed = False
        if self.enabled and self.num_generate_agents != 2:
            raise ValueError("strict HV requires hv_num_generate_agents=2")
        if self.enabled and (not isinstance(self.actor_model, str) or
                             any(c in self.actor_model for c in "*?[]") or
                             not self.actor_model.startswith("vehicle.")):
            raise ValueError("strict HV requires one exact CARLA vehicle blueprint ID")
        self.generation_mode = str(config.get("hv_generation_mode", "inpainting"))
        # V4-A conditions on all live actors in the full FOV.  Like legacy
        # sparse inpainting, it must not hide live actors before snapshotting.
        # Keep the telemetry fields distinct: full-FOV is not sparse context.
        self.sparse_inpainting = self.generation_mode == "sparse_inpainting"
        self.full_fov_inpainting = self.generation_mode == "full_fov_inpainting"
        self.partial_context = bool(config.get("hv_generation_partial_context", True))
        self.require_hidden_slots = bool(config.get("hv_generation_require_hidden_slots", True))
        hidden_slots = config.get("hv_generation_hidden_slots", None)
        self.hidden_slots = int(self.num_generate_agents if hidden_slots is None else hidden_slots)
        self.hidden_forward_min = float(config.get("hv_generation_hidden_forward_min", 0.0))
        self.hidden_forward_max = float(config.get("hv_generation_hidden_forward_max", 64.0))
        self.hidden_lateral_abs = float(config.get("hv_generation_hidden_lateral_abs", 32.0))
        self.rolling_enabled = bool(config.get("hv_generation_rolling", False))
        if self.enabled and self.rolling_enabled:
            raise ValueError("strict one-shot HV does not allow rolling generation")
        self.rolling_interval_steps = max(1, int(config.get("hv_generation_rolling_interval_steps", 50)))
        self.rolling_min_step = max(0, int(config.get("hv_generation_rolling_min_step", 50)))
        self.rolling_force_update = bool(config.get("hv_generation_rolling_force_update", False))
        rolling_min_rift_positive = config.get("hv_generation_rolling_min_rift_positive", None)
        self.rolling_min_rift_positive = int(
            self.num_generate_agents if rolling_min_rift_positive is None else rolling_min_rift_positive
        )
        self.gate_on_existing_rift_positive = bool(
            config.get("hv_generation_gate_on_existing_rift_positive", True)
        )
        self.existing_rift_positive_threshold = max(1, int(
            config.get("hv_generation_existing_rift_positive_threshold", 2)
        ))
        lifetime_radius = config.get("hv_generation_lifetime_radius", None)
        self.generated_lifetime_radius = float(
            self.hidden_forward_max + 30.0 if lifetime_radius is None else lifetime_radius
        )
        self._last_rolling_step = -self.rolling_interval_steps
        self._generated_actor_ids = set()
        self._generated_actor_sources = {}
        self._replaced_background_actor_ids = []
        self.lifecycle_telemetry = bool(config.get("hv_generation_lifecycle_telemetry", True))
        self._generated_actor_lifecycle = {}
        self._last_hidden_context_debug = {}
        # Do not remove/replace live vehicles to make the model proposal fit.
        # The full-FOV mode is the recommended configuration for this contract.
        if self.enabled and self._should_hide_partial_context():
            raise ValueError("strict one-shot HV forbids hiding context actors: "
                             "use hv_generation_mode=full_fov_inpainting "
                             "or hv_generation_partial_context=False")
        if self.enabled and (self.sd_background_model_only_traffic or
                             self.sd_background_replace_existing):
            raise ValueError("strict one-shot HV forbids removing existing traffic")
        self.snapshotter = HighValueSceneSnapshotter(
            actor_radius=float(config.get("hv_actor_radius", 80.0)),
            map_radius=float(config.get("hv_map_radius", config.get("hv_generation_map_radius", 64.0))),
            max_route_points=int(config.get("hv_max_route_points", 160)),
            max_map_lanes=int(config.get("hv_max_map_lanes", 120)),
            map_coordinate_mode=str(config.get("hv_snapshot_map_coordinate_mode", "legacy")),
            candidate_fov=float(config.get("hv_sd_fov", 64.0)),
            forward_min_x=float(config.get("hv_generation_forward_min_x", 0.0)),
            target_fov_margin=float(config.get("hv_generation_target_fov_margin", 2.0)),
            logger=logger,
        )
        self.generator = None
        self.generator_load_error = None
        if self.enabled:
            try:
                self.generator = ExternalGeneratedSceneModel(
                    entry_point=config.get(
                        "hv_generation_generator_entry",
                        "high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator",
                    ),
                    checkpoint_path=config.get("hv_generation_checkpoint"),
                    device=config.get("hv_generation_device", config.get("hv_init_device", "cuda")),
                    config=config,
                )
            except Exception as exc:
                self.generator_load_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "entry_point": config.get(
                        "hv_generation_generator_entry",
                        "high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator",
                    ),
                }

    def initialize(
            self,
            ego_vehicle: carla.Vehicle,
            route_config,
            cbv_recog_policy,
            cbv_route_planner,
            ego_rest_route_waypoints,
            existing_cbv_ids: Optional[Iterable[int]] = None) -> Dict:
        if not self.enabled:
            return {"success": False, "disabled": True}

        if self.sd_background_model_only_traffic:
            self._replaced_background_actor_ids = self.enforce_model_only_traffic(ego_vehicle, tick_after=True)
        elif self.sd_background_replace_existing:
            self._replaced_background_actor_ids = self._replace_nearby_background_actors(ego_vehicle)

        snapshot = self.snapshotter.build(
            ego_vehicle=ego_vehicle,
            route_config=route_config,
            ego_rest_route_waypoints=ego_rest_route_waypoints,
            cbv_recog_policy=cbv_recog_policy,
            cbv_route_planner=cbv_route_planner,
            existing_cbv_ids=[],
            # No pre-generation planner probe when the gate is disabled.
            # RIFT labeler writes planner interaction state, so it must not
            # serve as a hidden feedback loop for this strict one-shot run.
            include_rift_labels=bool(self.gate_on_existing_rift_positive),
            frame_id=0,
        )
        if self._has_existing_rift_positive(snapshot):
            result = self._existing_rift_positive_skip_result(snapshot, frame_id=0)
            self._dump(snapshot, None, None, result)
            return result

        hidden_actors = self._prepare_partial_context(ego_vehicle) if self._can_prepare_hidden_context() else []
        # Hiding actors changes the context, so serialize the exact context
        # consumed by the model rather than the pre-hide gate snapshot.
        if hidden_actors:
            snapshot = self.snapshotter.build(
                ego_vehicle=ego_vehicle,
                route_config=route_config,
                ego_rest_route_waypoints=ego_rest_route_waypoints,
                cbv_recog_policy=cbv_recog_policy,
                cbv_route_planner=cbv_route_planner,
                existing_cbv_ids=[],
                include_rift_labels=True,
                frame_id=0,
            )
        result, proposal, audit = self._run_generation_attempts(
            snapshot=snapshot,
            ego_vehicle=ego_vehicle,
            route_config=route_config,
            cbv_recog_policy=cbv_recog_policy,
            cbv_route_planner=cbv_route_planner,
            ego_rest_route_waypoints=ego_rest_route_waypoints,
            hidden_actors=hidden_actors,
            frame_id=0,
        )
        self._dump(snapshot, proposal, audit, result)
        if not result.get("success", False) and not result.get("skipped", False) and self.fail_on_invalid:
            raise RuntimeError(result.get("failure_reason", "high-value generation failed"))
        return result

    def should_attempt_rollout(self, frame_id: int) -> bool:
        if not self.enabled or not self.rolling_enabled:
            return False
        if frame_id < self.rolling_min_step:
            return False
        return frame_id - self._last_rolling_step >= self.rolling_interval_steps

    def maybe_generate_rollout(
            self,
            ego_vehicle: carla.Vehicle,
            route_config,
            cbv_recog_policy,
            cbv_route_planner,
            ego_rest_route_waypoints,
            frame_id: int) -> Optional[Dict]:
        if not self.should_attempt_rollout(frame_id):
            return None

        self._cleanup_generated_outside_lifetime(ego_vehicle, frame_id=frame_id)
        # In composed mode the Scenario Dreamer background actors are not a
        # substitute for the two high-value actors used by replenishment.
        # Legacy generators are recorded as ``high_value``, so their behavior
        # remains unchanged.
        active_generated = self._active_generated_nearby(ego_vehicle, source="high_value")
        if not self.rolling_force_update and len(active_generated) >= self.num_generate_agents:
            # A scheduled inspection was consumed even when no replenishment is
            # needed. Without this, this branch runs and dumps on every frame.
            self._last_rolling_step = frame_id
            return None

        probe = self.snapshotter.build(
            ego_vehicle=ego_vehicle,
            route_config=route_config,
            ego_rest_route_waypoints=ego_rest_route_waypoints,
            cbv_recog_policy=cbv_recog_policy,
            cbv_route_planner=cbv_route_planner,
            existing_cbv_ids=[],
            include_rift_labels=True,
            frame_id=frame_id,
        )
        if (
            not self.rolling_force_update
            and self._has_existing_rift_positive(probe)
        ):
            # A gate skip is still one scheduled rolling inspection.
            self._last_rolling_step = frame_id
            result = self._existing_rift_positive_skip_result(probe, frame_id=frame_id)
            result["rolling_generation"] = True
            self._dump(probe, None, None, result)
            return result

        self._last_rolling_step = frame_id
        hidden_actors = self._prepare_partial_context(ego_vehicle) if self._can_prepare_hidden_context() else []
        snapshot = self.snapshotter.build(
            ego_vehicle=ego_vehicle,
            route_config=route_config,
            ego_rest_route_waypoints=ego_rest_route_waypoints,
            cbv_recog_policy=cbv_recog_policy,
            cbv_route_planner=cbv_route_planner,
            existing_cbv_ids=[],
            include_rift_labels=True,
            frame_id=frame_id,
            extra={
                "rolling_generation": True,
                "rolling_force_update": bool(self.rolling_force_update),
            },
        )
        result, proposal, audit = self._run_generation_attempts(
            snapshot=snapshot,
            ego_vehicle=ego_vehicle,
            route_config=route_config,
            cbv_recog_policy=cbv_recog_policy,
            cbv_route_planner=cbv_route_planner,
            ego_rest_route_waypoints=ego_rest_route_waypoints,
            hidden_actors=hidden_actors,
            frame_id=frame_id,
        )
        self._dump(snapshot, proposal, audit, result)
        return result

    def _run_generation_attempts(
            self,
            snapshot: Dict,
            ego_vehicle: carla.Vehicle,
            route_config,
            cbv_recog_policy,
            cbv_route_planner,
            ego_rest_route_waypoints,
            hidden_actors: List[Dict],
            frame_id: int):
        """One generator call, two exact spawn requests, zero retries or forced ticks.

        Native CBVs_recog() owns candidate evaluation on the simulator's next
        scheduled step.  Do NOT call the RIFT labeler here: it writes planner
        interaction state and, before a physics tick, may not see all new actors.
        """
        result_base = {
            "schema_version": "hv_init_online_generation_result_v1",
            "scene_id": snapshot.get("scene_id"),
            "frame_id": int(frame_id),
            "generation_mode": self.generation_mode,
            "sparse_inpainting": bool(self.sparse_inpainting),
            "full_fov_inpainting": bool(self.full_fov_inpainting),
            "partial_context_enabled": bool(self.partial_context),
            "hidden_context_required": bool(self._requires_hidden_slots()),
            "hidden_context_actor_ids": [row["actor_id"] for row in hidden_actors],
            "hidden_context_actors": hidden_actors,
            "num_hidden_context_actors": len(hidden_actors),
            "num_requested_hidden_slots": int(self.hidden_slots),
            "num_required_hidden_slots": int(min(self.hidden_slots, self.num_generate_agents)),
            "num_requested_generated_agents": self.num_generate_agents,
            "replaced_background_actor_ids": list(self._replaced_background_actor_ids),
            "observed_agent_count_before_generation": len(snapshot.get("agents", []) or []),
            "scene_diagnostics_before_generation": dict(snapshot.get("diagnostics", {})),
            "hidden_context_debug": dict(self._last_hidden_context_debug),
            "rolling_generation": False,
            "rift_audit_is_observational": False,
            "rift_audit_status": "deferred_to_normal_cbvs_recog",
            "rift_audit_passed": None,
            "num_generated_rift_positive_agents": None,
            "generator_called": False,
            "skipped": False,
            "failure_stage": None,
        }
        result = {
            "success": False,
            **result_base,
            "spawned_actor_ids": [],
            "spawn_failures": [],
            "spawn_attempts": [],
            "audit_failed": None,
            "generation_max_attempts": 1,
            "configured_generation_max_attempts": self.requested_generation_max_attempts,
            "generation_attempts": [],
        }
        if self.generator_load_error is not None or self.generator is None:
            result.update({
                "failure_stage": "import",
                "failure_reason": (
                    f"{self.generator_load_error.get('type', 'GeneratorLoadError')}: "
                    f"{self.generator_load_error.get('message', 'generator unavailable')}"
                    if self.generator_load_error else "generator unavailable"
                ),
                "generator_load_error": self.generator_load_error,
            })
            return result, None, None
        if self._generation_consumed:
            result.update({
                "skipped": True,
                "failure_stage": "already_consumed",
                "failure_reason": "one-shot generation already consumed for this initializer",
            })
            return result, None, None
        if hidden_actors:
            result.update({
                "failure_stage": "hidden_context_forbidden",
                "failure_reason": "strict HV cannot remove existing actors before generation",
            })
            return result, None, None

        # Consume the only opportunity *before* invoking the generator.
        self._generation_consumed = True
        self._last_spawn_attempts = []
        proposal = None
        stage = "generator"
        try:
            result["generator_called"] = True
            proposal = self.generator.generate(snapshot)  # exactly once
            stage = "proposal_validation"
            if not isinstance(proposal, dict):
                raise TypeError("generator must return one dict proposal")
            states = proposal.get("generated_agent_states")
            if not isinstance(states, (list, tuple)) or len(states) != self.num_generate_agents:
                raise ValueError("generator must output exactly two vehicle states")
            for i, state in enumerate(states):
                if not isinstance(state, (list, tuple)) or len(state) < 6:
                    raise ValueError(f"vehicle {i}: expected [x,y,yaw,speed,length,width,...]")
                if any(not math.isfinite(float(value)) for value in state[:4]):
                    raise ValueError(f"vehicle {i}: non-finite x/y/yaw/speed")
                if float(state[3]) < 0.0:
                    raise ValueError(f"vehicle {i}: negative speed")
                if len(state) > 6 and float(state[6]) != 1.0:
                    raise ValueError(f"vehicle {i}: exists must be 1")
            sources = proposal.get("generated_agent_sources")
            if sources is None:
                sources = ["high_value"] * self.num_generate_agents
            if not isinstance(sources, (list, tuple)) or len(sources) != self.num_generate_agents:
                raise ValueError("generated_agent_sources must have exactly two entries")
            if any(source != "high_value" for source in sources):
                raise ValueError("strict HV mode accepts exactly two high_value sources")

            stage = "spawn"
            actors = self._spawn_generated_agents(snapshot["ego"], states, sources)
            if len(actors) != self.num_generate_agents:
                # _spawn_generated_agents normally raises and cleans up on
                # partial failure. Guard unusual providers that return fewer.
                self._cleanup_spawned(actors)
                raise SpawnRuntimeError("CARLA did not accept both generated vehicles")
            try:
                CarlaDataProvider.set_scenario_actors(ego_vehicle, actors)
                # request_new_actor() registers new actors with cached locations
                # and transforms initially set to None.  Populate those caches
                # before carla_env.reset() calls get_nearby_agents().
                # on_carla_tick() reads actor state; it does NOT tick the world
                # or change any generated/spawned x/y/yaw/speed.
                CarlaDataProvider.on_carla_tick()
            except Exception:
                self._cleanup_spawned(actors)
                raise
            # No world.tick(), on_carla_tick(), lane snapping, heading fixes,
            # RIFT-score feedback, or replacement proposal here.
            ids = [int(actor.id) for actor in actors]
            result.update({
                "success": True,
                "spawned_actor_ids": ids,
                "spawned_actor_sources": {str(actor_id): "high_value" for actor_id in ids},
                "num_spawned_generated_agents": len(ids),
                "spawn_attempts": list(self._last_spawn_attempts),
                "failure_stage": None,
            })
            self._generated_actor_ids.update(ids)
            self._record_spawn_lifecycle(result)
            result["generation_attempts"].append({
                "generation_attempt_index": 0,
                "success": True,
                "stage": "spawned_native_recognition_pending",
                "spawn_attempts": result["spawn_attempts"],
            })
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            result.update({
                "success": False,
                "failure_stage": stage,
                "failure_reason": reason,
                "spawn_attempts": list(self._last_spawn_attempts),
            })
            if stage == "spawn":
                result["spawn_failures"].append({
                    "generation_attempt_index": 0,
                    "failure_reason": reason,
                    "spawn_attempts": result["spawn_attempts"],
                })
            result["generation_attempts"].append({
                "generation_attempt_index": 0,
                "success": False,
                "failure_stage": stage,
                "failure_reason": reason,
                "spawn_attempts": result["spawn_attempts"],
            })
        return result, proposal, None

    def _should_hide_partial_context(self) -> bool:
        return bool(self.partial_context and not (self.sparse_inpainting or self.full_fov_inpainting))

    def _can_prepare_hidden_context(self) -> bool:
        return bool(self._should_hide_partial_context() and self.generator_load_error is None)

    def _requires_hidden_slots(self) -> bool:
        return bool(self.require_hidden_slots and self._should_hide_partial_context())

    def _has_existing_rift_positive(self, snapshot: Dict) -> bool:
        if not self.gate_on_existing_rift_positive:
            return False
        count = int((snapshot.get("diagnostics") or {}).get("num_rift_positive_agents", 0))
        return count >= self.existing_rift_positive_threshold

    def _existing_rift_positive_skip_result(self, snapshot: Dict, frame_id: int) -> Dict:
        existing_count = int((snapshot.get("diagnostics") or {}).get("num_rift_positive_agents", 0))
        return {
            "schema_version": "hv_init_online_generation_result_v2",
            "scene_id": snapshot.get("scene_id"),
            "frame_id": int(frame_id),
            "generation_mode": self.generation_mode,
            "sparse_inpainting": bool(self.sparse_inpainting),
            "full_fov_inpainting": bool(self.full_fov_inpainting),
            "generator_called": False,
            "success": False,
            "skipped": True,
            "skip_reason": "existing_rift_positive_actor",
            "pre_generation_rift_positive_agent_count": existing_count,
            "existing_rift_positive_threshold": self.existing_rift_positive_threshold,
            "rift_audit_is_observational": True,
            "spawned_actor_ids": [],
            "generated_rift_positive_actor_ids": [],
            "num_generated_rift_positive_agents": 0,
            "generated_rift_cbv_count_bucket": "0",
        }

    def _generated_rift_cbv_bucket(self, count: int) -> str:
        if count <= 0:
            return "0"
        if count == 1:
            return "1"
        if count == 2:
            return "2"
        return "3_plus"

    @staticmethod
    def _generated_positive_labels(audit: Optional[Dict], spawned_actor_ids: List[int]) -> List[Dict]:
        spawned_ids = {int(actor_id) for actor_id in spawned_actor_ids}
        labels = []
        for label in (audit or {}).get("rift_labels", []) or []:
            try:
                actor_id = int(label.get("actor_id"))
            except Exception:
                continue
            if actor_id in spawned_ids and label.get("is_interactive", False):
                labels.append(dict(label))
        labels.sort(key=HighValueGeneratedSceneInitializer._positive_label_rank, reverse=True)
        return labels

    @staticmethod
    def _positive_label_rank(label: Dict):
        try:
            score = float(label.get("score") or 0.0)
        except Exception:
            score = 0.0
        try:
            ego_distance = float(label.get("ego_distance") or 0.0)
        except Exception:
            ego_distance = 0.0
        return score, -ego_distance

    def _spawn_generated_agents(
            self,
            ego_state: Dict,
            generated_states: List[List[float]],
            generated_sources: Optional[List[str]] = None) -> List[carla.Actor]:
        spawned = []
        try:
            # Provider.create_blueprint() silently falls back when an ID is
            # unknown. Refuse that behavior under the fixed-blueprint contract.
            if hasattr(self.world, "get_blueprint_library"):
                bp = self.world.get_blueprint_library().find(self.actor_model)
                if bp.id != self.actor_model:
                    raise ValueError("configured blueprint is unavailable")
            for index, state in enumerate(generated_states):
                source = str((generated_sources or ["high_value"] * len(generated_states))[index])
                transform, velocity, attempt = self._state_to_carla(ego_state, state, index)
                attempt["generator_source"] = source
                self._last_spawn_attempts.append(attempt)
                try:
                    actor = CarlaDataProvider.request_new_actor(
                        self.actor_model,
                        transform,
                        rolename=self._actor_role_for_source(source),
                        autopilot=self.autopilot,
                        safe_blueprint=True,
                        tick=False,
                        max_try=1,
                    )
                except Exception as exc:
                    # Keep the raw proposal and map diagnostics intact. This
                    # records why CARLA rejected it without moving the state.
                    self._last_spawn_attempts[-1]["spawn_request_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise
                if actor.type_id != self.actor_model:
                    # No substitution: a fallback vehicle has a different box.
                    try:
                        CarlaDataProvider.remove_actor_by_id(actor.id)
                    except Exception:
                        actor.destroy()
                    raise SpawnRuntimeError(
                        f"CARLA blueprint fallback {actor.type_id} != {self.actor_model}"
                    )
                if velocity is not None and actor.type_id.startswith("vehicle."):
                    actor.set_target_velocity(velocity)
                self._last_spawn_attempts[-1]["spawned_actor_id"] = int(actor.id)
                # This is deliberately sampled before the explicit tick used
                # for the observational RIFT audit below.  It distinguishes
                # CARLA's acceptance of the raw requested transform from
                # normal movement during that audit tick.
                self._last_spawn_attempts[-1].update(
                    self._spawn_acceptance_fidelity(actor, self._last_spawn_attempts[-1])
                )
                self._generated_actor_sources[int(actor.id)] = source
                spawned.append(actor)
        except SpawnRuntimeError:
            self._cleanup_spawned(spawned)
            raise
        except Exception as exc:
            self._cleanup_spawned(spawned)
            if self.logger:
                self.logger.log(f">> HV generation spawn failed after {len(spawned)} spawned actors: {exc}", color="red")
            raise
        return spawned

    def _actor_role_for_source(self, source: str) -> str:
        if source == "scenario_dreamer_background":
            return self.background_actor_role
        return self.actor_role

    def _cleanup_spawned(self, actors: List[carla.Actor]) -> None:
        for actor in actors:
            try:
                CarlaDataProvider.remove_actor_by_id(actor.id)
            except Exception:
                try:
                    actor.destroy()
                except Exception:
                    pass
            finally:
                # A failed atomic two-car placement must leave no stale IDs.
                self._generated_actor_ids.discard(int(actor.id))
                self._generated_actor_sources.pop(int(actor.id), None)

    def _state_to_carla(self, ego_state: Dict, state: List[float], index: int):
        if len(state) < 6:
            raise ValueError("generated state must be [x, y, heading, speed, length, width, ...]")
        local_x, local_y, heading, speed = [float(v) for v in state[:4]]
        ego_yaw = math.radians(float(ego_state.get("yaw_deg", 0.0)))
        cos_yaw = math.cos(ego_yaw)
        sin_yaw = math.sin(ego_yaw)
        world_x = float(ego_state["x"]) + cos_yaw * local_x - sin_yaw * local_y
        world_y = float(ego_state["y"]) + sin_yaw * local_x + cos_yaw * local_y
        world_yaw = math.degrees(ego_yaw + heading)
        # The generator owns x/y, heading and speed.  The state schema has no
        # z component, so the nearest driving waypoint supplies terrain height
        # only; the spawned transform deliberately retains raw x/y/yaw.
        waypoint = self._road_waypoint(world_x, world_y, float(ego_state.get("z", 0.0)))
        transform = carla.Transform(
            carla.Location(
                x=float(world_x),
                y=float(world_y),
                z=float(waypoint.transform.location.z) + 0.3,
            ),
            carla.Rotation(yaw=float(world_yaw)),
        )
        spawn_yaw = math.radians(float(world_yaw))
        velocity = carla.Vector3D(
            x=float(speed) * math.cos(spawn_yaw),
            y=float(speed) * math.sin(spawn_yaw),
            z=0.0,
        )
        attempt = {
            "index": int(index),
            "generated_local_state": [float(v) for v in state],
            "raw_world_transform": {
                "x": float(world_x),
                "y": float(world_y),
                "z_query": float(ego_state.get("z", 0.0)),
                "yaw_deg": float(world_yaw),
            },
            "spawn_transform": {
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw_deg": float(transform.rotation.yaw),
            },
            "road_waypoint": {
                "x": float(waypoint.transform.location.x),
                "y": float(waypoint.transform.location.y),
                "z": float(waypoint.transform.location.z),
                "yaw_deg": float(waypoint.transform.rotation.yaw),
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
            },
            "road_projection_distance_m": float(math.hypot(
                float(waypoint.transform.location.x) - float(world_x),
                float(waypoint.transform.location.y) - float(world_y),
            )),
            "nearest_existing_vehicles": self._nearest_existing_vehicles(transform.location),
        }
        return transform, velocity, attempt

    def _road_waypoint(self, world_x: float, world_y: float, query_z: float):
        carla_map = CarlaDataProvider.get_map(self.world)
        waypoint = carla_map.get_waypoint(
            carla.Location(x=float(world_x), y=float(world_y), z=float(query_z)),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            raise SpawnRuntimeError(
                f"generated state is not projectable to a driving lane at x={world_x:.3f}, y={world_y:.3f}"
            )
        return waypoint

    def _nearest_existing_vehicles(self, location: carla.Location, limit: int = 8) -> List[Dict]:
        vehicles = []
        try:
            actors = CarlaDataProvider.get_all_actors().filter("vehicle.*")
        except Exception:
            return vehicles
        for actor in actors:
            try:
                actor_loc = actor.get_location()
                distance = float(location.distance(actor_loc))
                vehicles.append({
                    "actor_id": int(actor.id),
                    "role_name": actor.attributes.get("role_name", ""),
                    "distance": distance,
                    "x": float(actor_loc.x),
                    "y": float(actor_loc.y),
                    "z": float(actor_loc.z),
                })
            except Exception:
                continue
        vehicles.sort(key=lambda row: row["distance"])
        return vehicles[:limit]

    def _prepare_partial_context(self, ego_vehicle: carla.Vehicle) -> List[Dict]:
        candidates = self._background_candidates_for_hiding(ego_vehicle)
        debug = dict(self._last_hidden_context_debug)
        debug.update({
            "requested_hidden_slots": int(self.hidden_slots),
            "candidate_count": len(candidates),
            "remove_failures": [],
            "destroy_failures": [],
        })
        hidden = []
        for row in candidates[:max(0, self.hidden_slots)]:
            actor = row.pop("_actor")
            try:
                self._remove_from_scenario_bvs(ego_vehicle, actor.id)
                CarlaDataProvider.remove_actor_by_id(actor.id)
                hidden.append(row)
            except Exception:
                debug["remove_failures"].append({
                    "actor_id": int(actor.id),
                    "role_name": actor.attributes.get("role_name", ""),
                })
                try:
                    actor.destroy()
                    hidden.append(row)
                except Exception as exc:
                    debug["destroy_failures"].append({
                        "actor_id": int(actor.id),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    })
                    continue
        if hidden and self.world:
            self.world.tick()
            CarlaDataProvider.on_carla_tick()
        debug["hidden_count"] = len(hidden)
        self._last_hidden_context_debug = debug
        return hidden

    def _background_candidates_for_hiding(self, ego_vehicle: carla.Vehicle) -> List[Dict]:
        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = math.radians(float(ego_transform.rotation.yaw))
        cos_yaw = math.cos(-ego_yaw)
        sin_yaw = math.sin(-ego_yaw)
        rows = []
        debug = {
            "total_vehicle_actors": 0,
            "skipped_ego": 0,
            "skipped_non_background_role": 0,
            "skipped_forward_range": 0,
            "skipped_lateral_range": 0,
            "actor_state_errors": 0,
            "get_all_actors_error": None,
            "candidate_count": 0,
        }
        try:
            actors = CarlaDataProvider.get_all_actors().filter("vehicle.*")
        except Exception as exc:
            debug["get_all_actors_error"] = f"{type(exc).__name__}: {exc}"
            self._last_hidden_context_debug = debug
            return rows
        for actor in actors:
            debug["total_vehicle_actors"] += 1
            if actor.id == ego_vehicle.id:
                debug["skipped_ego"] += 1
                continue
            if actor.attributes.get("role_name", "") != "background":
                debug["skipped_non_background_role"] += 1
                continue
            try:
                loc = actor.get_location()
                dx = float(loc.x - ego_loc.x)
                dy = float(loc.y - ego_loc.y)
                local_x = cos_yaw * dx - sin_yaw * dy
                local_y = sin_yaw * dx + cos_yaw * dy
                if local_x < self.hidden_forward_min or local_x > self.hidden_forward_max:
                    debug["skipped_forward_range"] += 1
                    continue
                if abs(local_y) > self.hidden_lateral_abs:
                    debug["skipped_lateral_range"] += 1
                    continue
                distance = math.hypot(local_x, local_y)
                rows.append({
                    "_actor": actor,
                    "actor_id": int(actor.id),
                    "role_name": actor.attributes.get("role_name", ""),
                    "local_x": float(local_x),
                    "local_y": float(local_y),
                    "distance": float(distance),
                    "x": float(loc.x),
                    "y": float(loc.y),
                    "z": float(loc.z),
                })
            except Exception:
                debug["actor_state_errors"] += 1
                continue
        rows.sort(key=lambda item: (abs(item["local_y"]), item["local_x"]))
        debug["candidate_count"] = len(rows)
        self._last_hidden_context_debug = debug
        return rows

    def _remove_from_scenario_bvs(self, ego_vehicle: carla.Vehicle, actor_id: int) -> None:
        try:
            scenario_actors = CarlaDataProvider.get_scenario_actors_by_ego(ego_vehicle.id)
            scenario_actors.get("BVs", {}).pop(actor_id, None)
        except Exception:
            return

    def _replace_nearby_background_actors(self, ego_vehicle: carla.Vehicle) -> List[int]:
        """Remove only pre-existing CARLA background actors before composed generation."""
        removed = []
        ego_location = ego_vehicle.get_location()
        try:
            actors = CarlaDataProvider.get_all_actors().filter("vehicle.*")
        except Exception:
            return removed
        for actor in actors:
            if actor.id == ego_vehicle.id or actor.attributes.get("role_name", "") != "background":
                continue
            try:
                if ego_location.distance(actor.get_location()) > self.sd_background_replace_radius:
                    continue
                self._remove_from_scenario_bvs(ego_vehicle, actor.id)
                CarlaDataProvider.remove_actor_by_id(actor.id)
                removed.append(int(actor.id))
            except Exception:
                continue
        if removed:
            self.world.tick()
            CarlaDataProvider.on_carla_tick()
        return removed

    def enforce_model_only_traffic(self, ego_vehicle: carla.Vehicle, tick_after: bool = False) -> List[int]:
        """Remove every non-model vehicle so model proposals are the only BVs.

        This is deliberately separate from the legacy nearby-background
        replacement mode: it removes vehicles regardless of CARLA role name
        and preserves only ego plus actors spawned by this initializer.
        """
        removed = []
        try:
            actors = CarlaDataProvider.get_all_actors().filter("vehicle.*")
        except Exception:
            return removed
        protected_ids = {int(ego_vehicle.id)} | {int(actor_id) for actor_id in self._generated_actor_ids}
        for actor in actors:
            if int(actor.id) in protected_ids:
                continue
            try:
                self._remove_from_scenario_bvs(ego_vehicle, actor.id)
                CarlaDataProvider.remove_actor_by_id(actor.id)
                removed.append(int(actor.id))
            except Exception:
                continue
        if removed and tick_after:
            self.world.tick()
            CarlaDataProvider.on_carla_tick()
        return removed

    def _active_generated_nearby(self, ego_vehicle: carla.Vehicle, source: Optional[str] = None) -> List[int]:
        ego_loc = ego_vehicle.get_location()
        active = []
        for actor_id in list(self._generated_actor_ids):
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor is None or not actor.is_alive:
                self._generated_actor_ids.discard(actor_id)
                self._generated_actor_sources.pop(actor_id, None)
                continue
            if source is not None and self._generated_actor_sources.get(actor_id, "high_value") != source:
                continue
            if ego_loc.distance(actor.get_location()) <= self.generated_lifetime_radius:
                active.append(actor_id)
        return active

    def active_generated_actor_ids(
            self,
            ego_vehicle: Optional[carla.Vehicle] = None,
            source: Optional[str] = None) -> List[int]:
        if ego_vehicle is not None:
            return [int(actor_id) for actor_id in self._active_generated_nearby(ego_vehicle, source=source)]

        active = []
        for actor_id in list(self._generated_actor_ids):
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor is None or not actor.is_alive:
                self._generated_actor_ids.discard(actor_id)
                self._generated_actor_sources.pop(actor_id, None)
                continue
            if source is not None and self._generated_actor_sources.get(actor_id, "high_value") != source:
                continue
            active.append(int(actor_id))
        return active

    def active_generated_actor_ids_by_source(self, ego_vehicle: Optional[carla.Vehicle] = None) -> Dict[str, List[int]]:
        actor_ids = self.active_generated_actor_ids(ego_vehicle)
        groups = {"scenario_dreamer_background": [], "high_value": []}
        for actor_id in actor_ids:
            groups.setdefault(self._generated_actor_sources.get(actor_id, "high_value"), []).append(int(actor_id))
        return groups

    def _cleanup_generated_outside_lifetime(
            self,
            ego_vehicle: carla.Vehicle,
            frame_id: Optional[int] = None) -> None:
        ego_loc = ego_vehicle.get_location()
        for actor_id in list(self._generated_actor_ids):
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor is None or not actor.is_alive:
                self._generated_actor_ids.discard(actor_id)
                self._generated_actor_sources.pop(actor_id, None)
                continue
            if ego_loc.distance(actor.get_location()) > self.generated_lifetime_radius:
                try:
                    self.mark_generated_actor_event(
                        actor_id,
                        frame_id=frame_id,
                        event="lifetime_radius_removal",
                        terminal=True,
                    )
                    CarlaDataProvider.remove_actor_by_id(actor_id)
                finally:
                    self._generated_actor_ids.discard(actor_id)
                    self._generated_actor_sources.pop(actor_id, None)

    def _record_spawn_lifecycle(self, result: Dict) -> None:
        if not self.lifecycle_telemetry:
            return
        attempts = {
            int(item["spawned_actor_id"]): item
            for item in result.get("spawn_attempts", [])
            if item.get("spawned_actor_id") is not None
        }
        for actor_id in result.get("spawned_actor_ids", []):
            actor_id = int(actor_id)
            attempt = attempts.get(actor_id, {})
            fidelity = self._post_spawn_fidelity(actor_id, attempt)
            self._generated_actor_lifecycle[actor_id] = {
                "scene_id": result.get("scene_id"),
                "spawn_frame": int(result.get("frame_id", 0)),
                "terminal": False,
                "rift_cbv_activated": False,
                "rift_cbv_active": False,
                "green_active_frame_count": 0,
                "green_period_start_frame": None,
                "green_last_active_frame": None,
                "generator_source": attempt.get("generator_source", self._generated_actor_sources.get(actor_id, "high_value")),
            }
            self._append_lifecycle_event({
                "event": "spawned",
                "actor_id": actor_id,
                "scene_id": result.get("scene_id"),
                "frame_id": int(result.get("frame_id", 0)),
                "generator_source": self._generated_actor_lifecycle[actor_id]["generator_source"],
                "tm_control_requested": bool(self.autopilot),
                "generated_local_state": attempt.get("generated_local_state"),
                "raw_world_transform": attempt.get("raw_world_transform"),
                "spawn_transform": attempt.get("spawn_transform"),
                "road_projection_distance_m": attempt.get("road_projection_distance_m"),
                # This is an immediate pre-tick observation; no forced physics
                # advancement or actor-state correction is performed.
                **fidelity,
            })

    @staticmethod
    def _wrapped_yaw_error_deg(observed_yaw: float, requested_yaw: float) -> float:
        """Return the signed shortest yaw difference in degrees."""
        return (float(observed_yaw) - float(requested_yaw) + 180.0) % 360.0 - 180.0

    def _spawn_acceptance_fidelity(self, actor: carla.Actor, attempt: Dict) -> Dict:
        """Observe the actor immediately after CARLA accepts the request.

        This is telemetry only.  In particular, no transform, velocity, or
        controller setting is changed after sampling the generated state.
        """
        raw = attempt.get("raw_world_transform") or {}
        state = attempt.get("generated_local_state") or []
        try:
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            actual_speed = math.sqrt(
                float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
            )
            actual = {
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw_deg": float(transform.rotation.yaw),
                "speed_mps": float(actual_speed),
            }
            requested_x = float(raw["x"])
            requested_y = float(raw["y"])
            requested_yaw = float(raw["yaw_deg"])
            payload = {
                "spawn_acceptance_observation_available": True,
                # A newly spawned CARLA actor can report a stale state before
                # a simulator frame has advanced.  This is NOT an acceptance
                # fidelity measurement and cannot be used as a hard gate.
                "spawn_acceptance_observation_phase": "pre_tick_unconfirmed",
                "spawn_acceptance_pre_tick_state": actual,
                "spawn_acceptance_xy_error_m": float(math.hypot(
                    actual["x"] - requested_x,
                    actual["y"] - requested_y,
                )),
                "spawn_acceptance_yaw_error_deg": abs(self._wrapped_yaw_error_deg(
                    actual["yaw_deg"], requested_yaw,
                )),
            }
            if len(state) >= 4 and math.isfinite(float(state[3])):
                payload["spawn_acceptance_requested_speed_mps"] = float(state[3])
                payload["spawn_acceptance_speed_error_mps"] = abs(actual_speed - float(state[3]))
            return payload
        except Exception as exc:
            return {
                "spawn_acceptance_observation_available": False,
                "spawn_acceptance_observation_phase": "pre_tick_unconfirmed",
                "spawn_acceptance_observation_error": f"{type(exc).__name__}: {exc}",
            }

    def _post_spawn_fidelity(self, actor_id: int, attempt: Dict) -> Dict:
        """Observe, but never correct, the accepted actor before any forced tick.

        CARLA needs road height to instantiate a vehicle, while the generated
        x/y/yaw/speed remain model-owned.  An immediate sample makes that
        contract auditable and deliberately excludes z from the error fields.
        """
        raw = attempt.get("raw_world_transform") or {}
        state = attempt.get("generated_local_state") or []
        try:
            actor = CarlaDataProvider.get_actor_by_id(int(actor_id))
            if actor is None or not getattr(actor, "is_alive", False):
                return {"post_spawn_observation_available": False}
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            actual_speed = math.sqrt(
                float(velocity.x) ** 2 + float(velocity.y) ** 2 + float(velocity.z) ** 2
            )
            actual = {
                "x": float(transform.location.x),
                "y": float(transform.location.y),
                "z": float(transform.location.z),
                "yaw_deg": float(transform.rotation.yaw),
                "speed_mps": float(actual_speed),
            }
            requested_x = float(raw["x"])
            requested_y = float(raw["y"])
            requested_yaw = float(raw["yaw_deg"])
            requested_speed = float(state[3]) if len(state) >= 4 else None
            payload = {
                "post_spawn_observation_available": True,
                "actual_post_spawn_state": actual,
                "post_spawn_xy_error_m": float(math.hypot(
                    actual["x"] - requested_x,
                    actual["y"] - requested_y,
                )),
                "post_spawn_yaw_error_deg": abs(self._wrapped_yaw_error_deg(
                    actual["yaw_deg"], requested_yaw,
                )),
            }
            if requested_speed is not None and math.isfinite(requested_speed):
                payload["requested_speed_mps"] = requested_speed
                payload["post_spawn_speed_error_mps"] = abs(actual_speed - requested_speed)
            return payload
        except Exception as exc:
            return {
                "post_spawn_observation_available": False,
                "post_spawn_observation_error": f"{type(exc).__name__}: {exc}",
            }

    def record_normal_recognition_step(self, frame_id, gate_reason, policy, selected):
        """Record the actual normal pipeline call; never invoke a recognizer."""
        if not self.lifecycle_telemetry:
            return
        debug=getattr(policy,'last_candidate_debug',{}) if gate_reason is None else {}
        attention=getattr(getattr(policy,'AttnModel',None),'last_selection_debug',None) if gate_reason is None else None
        candidate_ids=set(debug.get('candidate_actor_ids',[]))
        selected_ids={int(a.id) for a in selected}
        for actor_id,state in list(self._generated_actor_lifecycle.items()):
            if state.get('terminal',False):continue
            self._append_lifecycle_event({
                'event':'normal_recognition_step','scene_id':state.get('scene_id'),
                'actor_id':actor_id,'frame_id':frame_id,'phase':'evolved_online_state',
                'gate_reason':gate_reason,
                'native_candidate':actor_id in candidate_ids if gate_reason is None else None,
                'selected_by_normal_recognizer':actor_id in selected_ids if gate_reason is None else None,
                'candidate_debug':debug,'attention':attention,
            })

    def observe_generated_actor_lifecycle(
            self,
            frame_id: int,
            ego_vehicle: carla.Vehicle,
            active_cbv_ids: Iterable[int]) -> None:
        """Record state transitions only; this method never controls actors."""
        if not self.lifecycle_telemetry:
            return
        active_cbv_ids = {int(actor_id) for actor_id in active_cbv_ids}
        ego_location = ego_vehicle.get_location()
        for actor_id, state in list(self._generated_actor_lifecycle.items()):
            if state.get("terminal", False):
                continue
            actor = CarlaDataProvider.get_actor_by_id(actor_id)
            if actor is None or not getattr(actor, "is_alive", False):
                self._generated_actor_ids.discard(actor_id)
                self.mark_generated_actor_event(
                    actor_id,
                    frame_id=frame_id,
                    event="actor_unavailable",
                    terminal=True,
                )
                continue
            if actor_id in active_cbv_ids:
                if not state.get("rift_cbv_activated", False):
                    state["rift_cbv_activated"] = True
                    self._append_lifecycle_event({
                        "event": "rift_cbv_activated",
                        "actor_id": int(actor_id),
                        "scene_id": state.get("scene_id"),
                        "frame_id": int(frame_id),
                        "generator_source": state.get("generator_source"),
                        "frames_from_spawn": int(frame_id) - int(state.get("spawn_frame", frame_id)),
                    })
                if not state.get("rift_cbv_active", False):
                    state["rift_cbv_active"] = True
                    state["green_period_start_frame"] = int(frame_id)
                    if state.get("rift_cbv_activated", False):
                        self._append_lifecycle_event({
                            "event": "rift_cbv_green_period_started",
                            "actor_id": int(actor_id),
                            "scene_id": state.get("scene_id"),
                            "frame_id": int(frame_id),
                            "generator_source": state.get("generator_source"),
                        })
                state["green_active_frame_count"] = int(state.get("green_active_frame_count", 0)) + 1
                state["green_last_active_frame"] = int(frame_id)
            elif state.get("rift_cbv_active", False):
                self._close_green_period(state, actor_id, frame_id, "rift_cbv_no_longer_active")
            # This is a diagnostic sample, not a continuous trajectory log.
            if int(frame_id) == int(state.get("spawn_frame", -1)):
                self._append_lifecycle_event({
                    "event": "post_spawn_alive",
                    "actor_id": int(actor_id),
                    "scene_id": state.get("scene_id"),
                    "frame_id": int(frame_id),
                    "generator_source": state.get("generator_source"),
                    "distance_to_ego_m": float(actor.get_location().distance(ego_location)),
                    "is_rift_cbv": actor_id in active_cbv_ids,
                })

    def finalize_generated_actor_lifecycle(self, frame_id: Optional[int]) -> None:
        """Flush observation-only green-duration telemetry at route cleanup."""
        if not self.lifecycle_telemetry:
            return
        for actor_id, state in list(self._generated_actor_lifecycle.items()):
            if state.get("terminal", False):
                continue
            if state.get("rift_cbv_active", False):
                self._close_green_period(state, int(actor_id), frame_id, "environment_cleanup")
            self._append_lifecycle_event({
                "event": "lifecycle_observation_ended",
                "actor_id": int(actor_id),
                "scene_id": state.get("scene_id"),
                "frame_id": None if frame_id is None else int(frame_id),
                "generator_source": state.get("generator_source"),
                "green_active_frame_count": int(state.get("green_active_frame_count", 0)),
                "terminal": False,
            })

    def mark_generated_actor_event(
            self,
            actor_id: int,
            frame_id: Optional[int],
            event: str,
            terminal: bool = False) -> None:
        if not self.lifecycle_telemetry:
            return
        actor_id = int(actor_id)
        state = self._generated_actor_lifecycle.get(actor_id)
        if state is None:
            return
        if state.get("terminal", False):
            return
        if event == "rift_cbv_returned_to_tm" and state.get("rift_cbv_active", False):
            self._close_green_period(state, actor_id, frame_id, event)
        if terminal:
            if state.get("rift_cbv_active", False):
                self._close_green_period(state, actor_id, frame_id, event)
            state["terminal"] = True
        payload = {
            "event": str(event),
            "actor_id": actor_id,
            "scene_id": state.get("scene_id"),
            "frame_id": None if frame_id is None else int(frame_id),
            "generator_source": state.get("generator_source"),
            "terminal": bool(terminal),
        }
        if frame_id is not None:
            payload["frames_from_spawn"] = int(frame_id) - int(state.get("spawn_frame", frame_id))
        payload["green_active_frame_count"] = int(state.get("green_active_frame_count", 0))
        self._append_lifecycle_event(payload)

    def _close_green_period(self, state: Dict, actor_id: int, frame_id: Optional[int], event: str) -> None:
        """Close an observed RIFT-CBV interval without changing the actor."""
        start = state.get("green_period_start_frame")
        last = state.get("green_last_active_frame")
        if start is None or last is None:
            state["rift_cbv_active"] = False
            state["green_period_start_frame"] = None
            return
        end = int(last)
        duration = max(0, end - int(start) + 1)
        self._append_lifecycle_event({
            "event": "rift_cbv_green_period_ended",
            "actor_id": int(actor_id),
            "scene_id": state.get("scene_id"),
            "frame_id": None if frame_id is None else int(frame_id),
            "generator_source": state.get("generator_source"),
            "green_period_start_frame": int(start),
            "green_period_end_frame": end,
            "green_period_frames": duration,
            "green_active_frame_count": int(state.get("green_active_frame_count", 0)),
            "reason": str(event),
        })
        state["rift_cbv_active"] = False
        state["green_period_start_frame"] = None

    def _append_lifecycle_event(self, payload: Dict) -> None:
        if not self.dump_dir:
            return
        try:
            path = Path(self.dump_dir)
            path.mkdir(parents=True, exist_ok=True)
            with (path / "lifecycle.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception as exc:
            if self.logger:
                self.logger.log(
                    f">> HV lifecycle telemetry write failed: {type(exc).__name__}: {exc}",
                    color="yellow",
                )

    def _write_lifecycle_metadata(self) -> None:
        """Persist telemetry configuration even when no actor is spawned."""
        if not self.dump_dir:
            return
        path = Path(self.dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "hv_generation_lifecycle_metadata_v1",
            "lifecycle_telemetry_enabled": bool(self.lifecycle_telemetry),
            "generation_mode": self.generation_mode,
            "num_high_value_agents": int(self.num_generate_agents),
            "generator_entry": self.config.get("hv_generation_generator_entry"),
            "sd_background_model_only_traffic": bool(self.sd_background_model_only_traffic),
            "sd_background_source": self.sd_background_source,
        }
        with (path / "lifecycle_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _dump(self, snapshot: Dict, proposal: Optional[Dict], audit: Optional[Dict], result: Dict) -> None:
        if not self.dump_dir:
            return
        Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
        self._write_lifecycle_metadata()
        scene_id = str(snapshot.get("scene_id", "scene")).replace("/", "_").replace(":", "_")
        frame_id = result.get("frame_id", snapshot.get("frame_id", "none"))
        path = Path(self.dump_dir) / f"{scene_id}_frame{frame_id}_online_generation.json"
        payload = {
            "schema_version": "hv_init_online_generation_dump_v1",
            "scene": snapshot,
            "proposal": proposal,
            "audit_after_generation": audit,
            "result": result,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def update_dump_result(self, result: Dict) -> None:
        if not self.dump_dir or not isinstance(result, dict):
            return
        scene_id = str(result.get("scene_id", "scene")).replace("/", "_").replace(":", "_")
        frame_id = result.get("frame_id", "none")
        path = Path(self.dump_dir) / f"{scene_id}_frame{frame_id}_online_generation.json"
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["result"] = result
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
