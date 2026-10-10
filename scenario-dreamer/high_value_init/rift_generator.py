import copy
import math
import os
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from high_value_init.model import HighValueInitHead
from high_value_init.scenario_dreamer_adapter import RiftToScenarioDreamerAdapter


class ScenarioDreamerHighValueInitGenerator:
    """Scenario Dreamer generator consumed by RIFT.

    RIFT provides the current route/map/actor scene and, during data collection,
    RIFT interaction labels supervise a per-actor high-value scorer. At online
    inference time those labels are not used as input. The scorer first ranks
    all existing actor slots. Scenario Dreamer LDM can then inpaint only the
    selected actor latents while ego, lanes, and non-selected actors stay
    conditioned on the original RIFT scene. The first ``hv_init_target_count``
    selected actors are the adversarial CBVs.
    """

    def __init__(self, checkpoint_path=None, device="cuda", config=None):
        self.config = config or {}
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.cbv_count = int(self.config.get("hv_init_target_count", 2))
        self.apply_top_k = int(self.config.get("hv_init_apply_top_k", self.cbv_count))
        self.apply_top_k = max(self.apply_top_k, self.cbv_count)
        self.num_generation_candidates = max(
            1, int(self.config.get("hv_model_num_generation_candidates", 8))
        )
        self.state_source = str(self.config.get("hv_model_state_source", "ldm_inpainting"))
        self.backbone_name = str(self.config.get("hv_model_backbone", "autoencoder"))
        self.actor_context_dim = int(self.config.get("hv_model_actor_context_dim", 12))

        self.adapter = RiftToScenarioDreamerAdapter(self.config)
        self.autoencoder = None
        self.autoencoder_cfg = None
        self.ldm = None
        self.ldm_cfg = None

        if self.backbone_name == "autoencoder":
            self.autoencoder = self._load_autoencoder()
            actor_feature_dim = int(self.config.get(
                "hv_model_agent_latent_dim",
                getattr(self.autoencoder_cfg.model, "agent_latent_dim", 8),
            ))
        elif self.backbone_name == "adapter_only":
            actor_feature_dim = int(self.config.get("hv_model_actor_feature_dim", 12))
        else:
            raise ValueError(f"Unsupported hv_model_backbone: {self.backbone_name}")

        self.model = HighValueInitHead(
            actor_feature_dim=actor_feature_dim,
            actor_context_dim=self.actor_context_dim,
            hidden_dim=int(self.config.get("hv_model_hidden_dim", 256)),
            num_layers=int(self.config.get("hv_model_num_layers", 3)),
            num_heads=int(self.config.get("hv_model_num_heads", 8)),
            dropout=float(self.config.get("hv_model_dropout", 0.1)),
        ).to(self.device)

        if checkpoint_path is None:
            if not bool(self.config.get("hv_init_allow_untrained_generator", False)):
                raise ValueError(
                    "hv_init_checkpoint is required for the high-value actor scorer. "
                    "Use --hv_init_debug_generator only for pipeline smoke tests."
                )
        else:
            self._load_head_checkpoint(checkpoint_path)

        if self.state_source in {"ldm_inpainting", "ldm", "ldm_lane_conditioned"}:
            self.ldm = self._load_ldm()
        elif self.state_source == "identity":
            self.ldm = None
        else:
            raise ValueError(
                "Unsupported hv_model_state_source: "
                f"{self.state_source}. Expected ldm_inpainting, ldm_lane_conditioned, ldm, or identity."
            )

        self.model.eval()
        self._freeze_eval(self.autoencoder)
        self._freeze_eval(self.ldm)

    @torch.no_grad()
    def generate(self, scene: Dict) -> Dict:
        actor_count = len(scene.get("actors", []))
        if actor_count < self.cbv_count:
            raise ValueError("not enough actors for high-value actor generation")

        original_logits, original_scores, actor_ids = self._score_scene(scene)
        if len(actor_ids) != actor_count:
            raise ValueError(
                f"original actor count mismatch: {len(actor_ids)} vs {actor_count}"
            )
        top_k = min(self.apply_top_k, len(actor_ids))
        if top_k < self.cbv_count:
            raise ValueError("not enough actors to select requested CBVs")

        top_indices = torch.topk(original_logits, k=top_k).indices.detach().cpu().tolist()
        selected_actor_ids = [actor_ids[idx] for idx in top_indices]
        cbv_actor_ids = selected_actor_ids[:self.cbv_count]

        candidates = self._generate_candidate_scenes(scene, selected_actor_ids)
        best = None
        for candidate_index, candidate_scene in enumerate(candidates):
            logits, scores, candidate_actor_ids = self._score_scene(candidate_scene)
            if candidate_actor_ids != actor_ids:
                raise ValueError(
                    "candidate actor identity/order changed after Scenario Dreamer generation"
                )

            objective = float(torch.sum(logits[top_indices]).detach().cpu())
            if best is None or objective > best["objective"]:
                best = {
                    "candidate_index": candidate_index,
                    "scene": candidate_scene,
                    "logits": logits.detach().cpu().tolist(),
                    "scores": scores.detach().cpu().tolist(),
                    "actor_ids": actor_ids,
                    "top_indices": top_indices,
                    "objective": objective,
                }

        target_mask = [actor_id in set(selected_actor_ids) for actor_id in best["actor_ids"]]
        cbv_mask = [actor_id in set(cbv_actor_ids) for actor_id in best["actor_ids"]]
        state_updates = self._state_updates_from_candidate(scene, best["scene"], selected_actor_ids)

        return {
            "target_mask": target_mask,
            "cbv_mask": cbv_mask,
            "selected_actor_ids": selected_actor_ids,
            "cbv_actor_ids": cbv_actor_ids,
            "target_mask_logits": best["logits"],
            "actor_scores": best["scores"],
            "original_target_mask_logits": original_logits.detach().cpu().tolist(),
            "original_actor_scores": original_scores.detach().cpu().tolist(),
            "state_updates": state_updates,
            "state_generation_source": self.state_source,
            "num_generation_candidates": len(candidates),
            "chosen_candidate_index": best["candidate_index"],
            "apply_top_k": self.apply_top_k,
            "cbv_count": self.cbv_count,
        }

    def _generate_candidate_scenes(self, scene: Dict, selected_actor_ids: List[int]) -> List[Dict]:
        if self.state_source == "identity":
            return [copy.deepcopy(scene)]

        candidates = []
        for _candidate_index in range(self.num_generation_candidates):
            if self.state_source == "ldm_inpainting":
                candidates.append(self._sample_inpainted_scene(scene, selected_actor_ids))
            elif self.state_source == "ldm_lane_conditioned":
                candidates.append(self._sample_lane_conditioned_scene(scene))
            else:
                candidates.append(self._sample_initial_scene(scene))
        return candidates

    def _sample_inpainted_scene(self, scene: Dict, selected_actor_ids: List[int]) -> Dict:
        ae_data, actor_ids = self.adapter.build(scene)
        ae_data = ae_data.to(self.device)
        agent_mu, lane_mu, agent_log_var, lane_log_var = self.autoencoder.model.forward_encoder(
            ae_data,
            return_stats=True,
        )

        ldm_data, _ldm_actor_ids = self.adapter.build_ldm_seed(
            scene,
            agent_latent_dim=int(getattr(self.ldm_cfg.model, "agent_latent_dim", 8)),
            lane_latent_dim=int(getattr(self.ldm_cfg.model, "lane_latent_dim", 8)),
        )
        self._validate_ldm_cardinality(ldm_data)

        (
            agent_mu,
            agent_log_var,
            lane_mu,
            lane_log_var,
            l2l_edge_index,
            sorted_actor_ids,
        ) = self._reorder_scene_conditioning_for_ldm(
            ae_data,
            actor_ids,
            agent_mu.detach(),
            agent_log_var.detach(),
            lane_mu.detach(),
            lane_log_var.detach(),
            ldm_data,
        )

        agent_latents, lane_latents = self._sample_normalized_latents(
            agent_mu,
            agent_log_var,
            lane_mu,
            lane_log_var,
        )
        selected_set = {int(actor_id) for actor_id in selected_actor_ids}
        agent_mask = torch.ones(agent_latents.shape[0], dtype=torch.bool, device=self.device)
        for slot_index, actor_id in enumerate(sorted_actor_ids, start=1):
            if int(actor_id) in selected_set:
                agent_mask[slot_index] = False

        ldm_data["agent"].latents = agent_latents
        ldm_data["lane"].latents = lane_latents
        ldm_data["agent"].mask = agent_mask
        ldm_data["lane"].mask = torch.ones(lane_latents.shape[0], dtype=torch.bool, device=self.device)
        ldm_data["lane", "to", "lane"].edge_index = l2l_edge_index.to(self.device)
        generated_data = self._run_ldm(ldm_data, mode="inpainting")
        return self._scene_with_inpainted_actor_states(
            scene,
            generated_data,
            sorted_actor_ids,
            selected_set,
        )

    def _sample_lane_conditioned_scene(self, scene: Dict) -> Dict:
        ae_data, _actor_ids = self.adapter.build(scene)
        ae_data = ae_data.to(self.device)
        agent_mu, lane_mu, _agent_log_var, _lane_log_var = self.autoencoder.model.forward_encoder(
            ae_data,
            return_stats=True,
        )
        del agent_mu

        ldm_data, actor_ids = self.adapter.build_ldm_seed(
            scene,
            agent_latent_dim=int(getattr(self.ldm_cfg.model, "agent_latent_dim", 8)),
            lane_latent_dim=int(getattr(self.ldm_cfg.model, "lane_latent_dim", 8)),
        )
        self._validate_ldm_cardinality(ldm_data)
        lane_mu, l2l_edge_index = self._reorder_lane_conditioning_for_ldm(ae_data, lane_mu, ldm_data)
        ldm_data["lane", "to", "lane"].edge_index = l2l_edge_index.to(self.device)
        lane_latents = self._normalize_lane_latents(lane_mu.detach())
        ldm_data["lane"].latents = lane_latents
        generated_data = self._run_ldm(ldm_data, mode="lane_conditioned")
        return self._scene_with_generated_actor_states(scene, generated_data, actor_ids)

    def _sample_initial_scene(self, scene: Dict) -> Dict:
        ldm_data, actor_ids = self.adapter.build_ldm_seed(
            scene,
            agent_latent_dim=int(getattr(self.ldm_cfg.model, "agent_latent_dim", 8)),
            lane_latent_dim=int(getattr(self.ldm_cfg.model, "lane_latent_dim", 8)),
        )
        self._validate_ldm_cardinality(ldm_data)
        generated_data = self._run_ldm(ldm_data, mode="initial_scene")
        return self._scene_with_generated_actor_states(scene, generated_data, actor_ids)

    def _validate_ldm_cardinality(self, ldm_data):
        max_agents = int(getattr(self.ldm_cfg.dataset, "max_num_agents", 0))
        max_lanes = int(getattr(self.ldm_cfg.dataset, "max_num_lanes", 0))
        num_agents = int(ldm_data["num_agents"][0].item())
        num_lanes = int(ldm_data["num_lanes"][0].item())
        if max_agents and num_agents > max_agents:
            raise ValueError(
                f"RIFT scene has {num_agents} Scenario Dreamer agents including ego, "
                f"but LDM max_num_agents is {max_agents}"
            )
        if max_lanes and num_lanes > max_lanes:
            raise ValueError(
                f"RIFT scene has {num_lanes} Scenario Dreamer lanes, "
                f"but LDM max_num_lanes is {max_lanes}"
            )

    def _run_ldm(self, ldm_data, mode: str):
        ldm_data = ldm_data.to(self.device)
        ema_context = self.ldm.ema.average_parameters() if hasattr(self.ldm, "ema") else nullcontext()
        with ema_context:
            generated_data, _images = self.ldm.forward(
                ldm_data,
                mode=mode,
                batch_idx=0,
                visualize=False,
            )
        return generated_data

    def _score_scene(self, scene: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        actor_features, context_features, actor_ids = self._build_features(scene)
        outputs = self.model(
            actor_features=actor_features.to(self.device),
            actor_context_features=context_features.to(self.device),
            valid_mask=torch.ones((1, len(actor_ids)), dtype=torch.bool, device=self.device),
        )
        return outputs["target_mask_logits"][0], outputs["actor_scores"][0], actor_ids

    def _build_features(self, scene: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if self.backbone_name == "autoencoder":
            return self._build_autoencoder_features(scene)
        return self._build_adapter_only_features(scene)

    def _build_autoencoder_features(self, scene: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        sd_data, actor_ids = self.adapter.build(scene)
        sd_data = sd_data.to(self.device)
        agent_mu, _lane_mu, _agent_log_var, _lane_log_var = self.autoencoder.model.forward_encoder(
            sd_data,
            return_stats=True,
        )
        if agent_mu.shape[0] != len(actor_ids) + 1:
            raise ValueError(
                "Scenario Dreamer adapter/autoencoder actor count mismatch: "
                f"got {agent_mu.shape[0]} embeddings for {len(actor_ids)} RIFT actors plus ego"
            )
        actor_features = agent_mu[1:].unsqueeze(0)
        context_features, context_actor_ids = self.adapter.actor_context_features(scene)
        if context_actor_ids != actor_ids:
            raise ValueError("adapter actor feature order changed between AE and context branches")
        return actor_features, context_features.to(self.device), actor_ids

    def _build_adapter_only_features(self, scene: Dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        context_features, actor_ids = self.adapter.actor_context_features(scene)
        return context_features, context_features.clone(), actor_ids

    def _scene_with_generated_actor_states(self, scene: Dict, generated_data, actor_ids: List[int]) -> Dict:
        generated_states = generated_data["agent"].x
        if generated_states.shape[0] < len(actor_ids) + 1:
            raise ValueError(
                "Scenario Dreamer generated fewer agent states than requested actor slots: "
                f"{generated_states.shape[0]} vs {len(actor_ids) + 1}"
            )

        local_states = generated_states[1:len(actor_ids) + 1].detach().cpu().numpy()
        assigned_states = self._assign_generated_states(scene, local_states)
        new_scene = copy.deepcopy(scene)
        ego = new_scene["ego"]
        for actor, local_state in zip(new_scene["actors"], assigned_states):
            self._apply_local_state_to_actor(ego, actor, local_state)
        return new_scene

    def _assign_generated_states(self, scene: Dict, generated_local_states: np.ndarray) -> np.ndarray:
        original_states, _actor_ids = self.adapter.local_actor_states(scene)
        if generated_local_states.shape[0] != original_states.shape[0]:
            raise ValueError("generated state count does not match RIFT actor count")

        remaining = set(range(generated_local_states.shape[0]))
        assigned = []
        for original in original_states:
            if not remaining:
                break
            best_idx = min(
                remaining,
                key=lambda idx: float(np.linalg.norm(generated_local_states[idx, :2] - original[:2])),
            )
            remaining.remove(best_idx)
            assigned.append(generated_local_states[best_idx])
        return np.asarray(assigned, dtype=np.float32)

    def _scene_with_inpainted_actor_states(
            self,
            scene: Dict,
            generated_data,
            sorted_actor_ids: List[int],
            selected_actor_ids: set) -> Dict:
        generated_states = generated_data["agent"].x
        if generated_states.shape[0] < len(sorted_actor_ids) + 1:
            raise ValueError(
                "Scenario Dreamer generated fewer inpainted states than requested actor slots: "
                f"{generated_states.shape[0]} vs {len(sorted_actor_ids) + 1}"
            )

        new_scene = copy.deepcopy(scene)
        ego = new_scene["ego"]
        actors_by_id = {int(actor["actor_id"]): actor for actor in new_scene["actors"]}
        for slot_index, actor_id in enumerate(sorted_actor_ids, start=1):
            if int(actor_id) not in selected_actor_ids:
                continue
            self._apply_local_state_to_actor(
                ego,
                actors_by_id[int(actor_id)],
                generated_states[slot_index].detach().cpu().numpy(),
            )
        return new_scene

    def _apply_local_state_to_actor(self, ego: Dict, actor: Dict, local_state: np.ndarray):
        x_local = float(local_state[0])
        y_local = float(local_state[1])
        speed = max(0.0, float(local_state[2]))
        cos_heading = float(local_state[3])
        sin_heading = float(local_state[4])
        rel_yaw = math.atan2(sin_heading, cos_heading)

        world_x, world_y = self._local_to_world(ego, x_local, y_local)
        yaw_deg = float(ego.get("yaw_deg", 0.0)) + math.degrees(rel_yaw)
        yaw_rad = math.radians(yaw_deg)

        actor["x"] = world_x
        actor["y"] = world_y
        actor["yaw_deg"] = yaw_deg
        actor["speed_mps"] = speed
        actor["vx"] = speed * math.cos(yaw_rad)
        actor["vy"] = speed * math.sin(yaw_rad)
        actor["vz"] = 0.0
        actor["length"] = max(0.1, float(local_state[5]))
        actor["width"] = max(0.1, float(local_state[6]))

    def _state_updates_from_candidate(
            self,
            original_scene: Dict,
            candidate_scene: Dict,
            selected_actor_ids: List[int]) -> Dict[str, Dict]:
        candidate_by_id = {
            int(actor["actor_id"]): actor
            for actor in candidate_scene.get("actors", [])
        }
        updates = {}
        for actor_id in selected_actor_ids:
            actor = candidate_by_id[int(actor_id)]
            updates[str(actor_id)] = {
                "x": float(actor["x"]),
                "y": float(actor["y"]),
                "z": float(actor.get("z", 0.0)),
                "yaw_deg": float(actor.get("yaw_deg", 0.0)),
                "speed_mps": float(actor.get("speed_mps", 0.0)),
                "vx": float(actor.get("vx", 0.0)),
                "vy": float(actor.get("vy", 0.0)),
                "vz": float(actor.get("vz", 0.0)),
                "roll_deg": float(actor.get("roll_deg", 0.0)),
                "pitch_deg": float(actor.get("pitch_deg", 0.0)),
            }
        return updates

    def _normalize_lane_latents(self, lane_latents: torch.Tensor) -> torch.Tensor:
        mean = getattr(self.ldm_cfg.dataset, "lane_latents_mean", None)
        std = getattr(self.ldm_cfg.dataset, "lane_latents_std", None)
        if mean is None or std is None:
            return lane_latents
        mean = self._as_device_tensor(mean, lane_latents)
        std = self._as_device_tensor(std, lane_latents).clamp_min(1e-6)
        return (lane_latents - mean) / std

    def _normalize_agent_latents(self, agent_latents: torch.Tensor) -> torch.Tensor:
        mean = getattr(self.ldm_cfg.dataset, "agent_latents_mean", None)
        std = getattr(self.ldm_cfg.dataset, "agent_latents_std", None)
        if mean is None or std is None:
            return agent_latents
        mean = self._as_device_tensor(mean, agent_latents)
        std = self._as_device_tensor(std, agent_latents).clamp_min(1e-6)
        return (agent_latents - mean) / std

    def _sample_normalized_latents(
            self,
            agent_mu: torch.Tensor,
            agent_log_var: torch.Tensor,
            lane_mu: torch.Tensor,
            lane_log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_latents = agent_mu + torch.randn_like(agent_mu) * torch.exp(0.5 * agent_log_var)
        lane_latents = lane_mu + torch.randn_like(lane_mu) * torch.exp(0.5 * lane_log_var)
        return (
            self._normalize_agent_latents(agent_latents),
            self._normalize_lane_latents(lane_latents),
        )

    def _reorder_scene_conditioning_for_ldm(
            self,
            ae_data,
            actor_ids: List[int],
            agent_mu: torch.Tensor,
            agent_log_var: torch.Tensor,
            lane_mu: torch.Tensor,
            lane_log_var: torch.Tensor,
            ldm_data):
        """Match Scenario Dreamer's latent cache ordering and keep actor ids aligned."""
        from utils.data_helpers import reorder_indices

        agent_states = ae_data["agent"].x.detach().cpu().numpy()
        road_points = ae_data["lane"].x.detach().cpu().numpy()
        edge_index_lane_to_lane = ldm_data["lane", "to", "lane"].edge_index.detach().cpu().numpy()
        agent_order = self._scenario_dreamer_agent_order(
            agent_states,
            road_points,
            edge_index_lane_to_lane,
            int(ldm_data["lg_type"][0].item()),
        )

        agent_mu_np, agent_log_var_np, lane_mu_np, lane_log_var_np, edge_index_sorted, _agent_mask, _lane_mask = reorder_indices(
            agent_mu.detach().cpu().numpy(),
            agent_log_var.detach().cpu().numpy(),
            lane_mu.detach().cpu().numpy(),
            lane_log_var.detach().cpu().numpy(),
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            int(ldm_data["lg_type"][0].item()),
            dataset="nuplan",
        )
        sorted_actor_ids = [
            int(actor_ids[int(slot_index) - 1])
            for slot_index in agent_order[1:]
        ]
        return (
            torch.from_numpy(agent_mu_np).to(device=self.device, dtype=agent_mu.dtype),
            torch.from_numpy(agent_log_var_np).to(device=self.device, dtype=agent_log_var.dtype),
            torch.from_numpy(lane_mu_np).to(device=self.device, dtype=lane_mu.dtype),
            torch.from_numpy(lane_log_var_np).to(device=self.device, dtype=lane_log_var.dtype),
            torch.from_numpy(edge_index_sorted).long(),
            sorted_actor_ids,
        )

    def _scenario_dreamer_agent_order(
            self,
            agent_states: np.ndarray,
            road_points: np.ndarray,
            edge_index_lane_to_lane: np.ndarray,
            lg_type: int) -> List[int]:
        from utils.data_helpers import reorder_indices

        identity = np.eye(agent_states.shape[0], dtype=np.float32)
        dummy_lane = np.zeros((road_points.shape[0], 1), dtype=np.float32)
        dummy_lane_log_var = np.zeros_like(dummy_lane)
        sorted_identity, _agent_log_var, _lane_mu, _lane_log_var, _edge_index, _agent_mask, _lane_mask = reorder_indices(
            identity,
            identity.copy(),
            dummy_lane,
            dummy_lane_log_var,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            lg_type,
            dataset="nuplan",
        )
        return [int(np.argmax(row)) for row in sorted_identity]

    def _reorder_lane_conditioning_for_ldm(self, ae_data, lane_mu: torch.Tensor, ldm_data):
        """Apply Scenario Dreamer's LDM cache ordering to online lane latents."""
        from utils.data_helpers import reorder_indices

        agent_states = ae_data["agent"].x.detach().cpu().numpy()
        road_points = ae_data["lane"].x.detach().cpu().numpy()
        edge_index_lane_to_lane = ldm_data["lane", "to", "lane"].edge_index.detach().cpu().numpy()
        lane_mu_np = lane_mu.detach().cpu().numpy()

        agent_latent_dim = int(getattr(self.ldm_cfg.model, "agent_latent_dim", 8))
        dummy_agent_mu = np.zeros((agent_states.shape[0], agent_latent_dim), dtype=np.float32)
        dummy_agent_log_var = np.zeros_like(dummy_agent_mu)
        dummy_lane_log_var = np.zeros_like(lane_mu_np)

        _agent_mu, _agent_log_var, lane_mu_sorted, _lane_log_var, edge_index_sorted, _agent_mask, _lane_mask = reorder_indices(
            dummy_agent_mu,
            dummy_agent_log_var,
            lane_mu_np,
            dummy_lane_log_var,
            edge_index_lane_to_lane,
            agent_states,
            road_points,
            int(ldm_data["lg_type"][0].item()),
            dataset="nuplan",
        )
        lane_mu_sorted = torch.from_numpy(lane_mu_sorted).to(device=lane_mu.device, dtype=lane_mu.dtype)
        edge_index_sorted = torch.from_numpy(edge_index_sorted).long()
        return lane_mu_sorted, edge_index_sorted

    def _as_device_tensor(self, value, like_tensor: torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        return tensor.to(device=like_tensor.device, dtype=like_tensor.dtype)

    def _local_to_world(self, ego: Dict, x_local: float, y_local: float) -> Tuple[float, float]:
        yaw = math.radians(float(ego.get("yaw_deg", 0.0)))
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (
            float(ego["x"]) + cos_yaw * x_local - sin_yaw * y_local,
            float(ego["y"]) + sin_yaw * x_local + cos_yaw * y_local,
        )

    def _load_autoencoder(self):
        checkpoint_path = self.config.get("hv_model_autoencoder_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                "hv_model_backbone=autoencoder requires --hv_model_autoencoder_checkpoint. "
                "This is the Scenario Dreamer AE checkpoint."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        from models.scenario_dreamer_autoencoder import ScenarioDreamerAutoEncoder

        cfg = self._load_scenario_dreamer_cfg(checkpoint_path, "hv_model_cfg_path", "ae")
        self.autoencoder_cfg = cfg
        model = ScenarioDreamerAutoEncoder.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            map_location=self.device,
        ).to(self.device)
        return model

    def _load_ldm(self):
        checkpoint_path = self.config.get("hv_model_ldm_checkpoint")
        if not checkpoint_path:
            raise ValueError(
                f"hv_model_state_source={self.state_source} requires "
                "--hv_model_ldm_checkpoint. This is the Scenario Dreamer LDM checkpoint."
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        if self.autoencoder_cfg is None:
            raise ValueError("Scenario Dreamer LDM generation requires an autoencoder cfg/model")

        from models.scenario_dreamer_ldm import ScenarioDreamerLDM

        cfg = self._load_scenario_dreamer_cfg(checkpoint_path, "hv_model_ldm_cfg_path", "ldm")
        self._assign_autoencoder_path(cfg, self.config.get("hv_model_autoencoder_checkpoint"))
        self.ldm_cfg = cfg
        model = ScenarioDreamerLDM.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            cfg_ae=self.autoencoder_cfg,
            map_location=self.device,
        ).to(self.device)
        if hasattr(model, "ema"):
            model.ema.to(self.device)
        return model

    def _load_scenario_dreamer_cfg(self, checkpoint_path: str, cfg_key: str, section: str):
        cfg_path = self.config.get(cfg_key)
        if cfg_path:
            from omegaconf import OmegaConf
            return self._normalize_cfg(OmegaConf.load(cfg_path), section)

        checkpoint = self._torch_load(checkpoint_path, map_location="cpu")
        cfg = checkpoint.get("hyper_parameters", {}).get("cfg") if isinstance(checkpoint, dict) else None
        if cfg is None:
            raise ValueError(
                f"Cannot recover Scenario Dreamer {section} cfg from checkpoint. "
                f"Pass --{cfg_key} pointing to a resolved Scenario Dreamer config."
            )
        return self._normalize_cfg(cfg, section)

    def _normalize_cfg(self, cfg, section: str):
        if isinstance(cfg, dict):
            from omegaconf import OmegaConf
            cfg = OmegaConf.create(cfg)
        if section == "ae" and hasattr(cfg, "ae") and not hasattr(cfg, "model"):
            dataset_name = self._read_dataset_name(cfg)
            cfg = cfg.ae
            if dataset_name is not None and not hasattr(cfg, "dataset_name"):
                self._set_cfg_value(cfg, "dataset_name", dataset_name)
        elif section == "ldm" and hasattr(cfg, "ldm") and not hasattr(cfg, "model"):
            dataset_name = self._read_dataset_name(cfg)
            cfg = cfg.ldm
            if dataset_name is not None and not hasattr(cfg, "dataset_name"):
                self._set_cfg_value(cfg, "dataset_name", dataset_name)
        return cfg

    def _read_dataset_name(self, cfg):
        if not hasattr(cfg, "dataset_name"):
            return None
        dataset_node = getattr(cfg, "dataset_name")
        return getattr(dataset_node, "name", dataset_node)

    def _assign_autoencoder_path(self, cfg, checkpoint_path: str):
        if checkpoint_path and hasattr(cfg, "model"):
            self._set_cfg_value(cfg.model, "autoencoder_path", checkpoint_path)

    def _set_cfg_value(self, cfg, key: str, value):
        try:
            from omegaconf import OmegaConf
            OmegaConf.set_struct(cfg, False)
            setattr(cfg, key, value)
        except Exception:
            setattr(cfg, key, value)

    def _load_head_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        checkpoint = self._torch_load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            state_dict = (
                checkpoint.get("hv_head_state_dict")
                or checkpoint.get("head_state_dict")
                or checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            state_dict = checkpoint
        cleaned_state_dict = {}
        expected_keys = self.model.state_dict()
        for key, value in state_dict.items():
            if key.startswith("model."):
                key = key[len("model."):]
            if key.startswith("hv_head."):
                key = key[len("hv_head."):]
            if key.startswith("head."):
                key = key[len("head."):]
            if key not in expected_keys:
                continue
            if not hasattr(value, "shape") or tuple(value.shape) != tuple(expected_keys[key].shape):
                continue
            cleaned_state_dict[key] = value
        if not cleaned_state_dict:
            raise ValueError(f"No high-value head weights found in {checkpoint_path}")
        self.model.load_state_dict(cleaned_state_dict, strict=True)

    def _torch_load(self, checkpoint_path, map_location):
        try:
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=map_location)

    def _freeze_eval(self, model):
        if model is None:
            return
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
