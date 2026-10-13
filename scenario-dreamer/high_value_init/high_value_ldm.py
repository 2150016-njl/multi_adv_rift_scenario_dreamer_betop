"""Route-conditioned Scenario Dreamer LDM for high-value initial states.

This module reuses Scenario Dreamer's original AutoEncoder and LDM modules.
RIFT labels are only used upstream to construct positive training records and
downstream to audit generated scenes. They are not model inputs here.
"""

import itertools
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from cfgs.config import BEFORE_PARTITION
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

from high_value_init.rift_dataset import (
    HighValueSceneGraphBuilder,
    NUPLAN_STATS,
    decoded_sd_agents_to_hv_states,
    state_generation_schema_is_clean,
)
from nn_modules.autoencoder import AutoEncoder
from nn_modules.ldm import LDM
from utils.data_helpers import normalize_latents, unnormalize_latents, unnormalize_scene


SCHEMA_VERSION = "hv_route_conditioned_ldm_v1"


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _to_plain(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=False)
    return value


def _read_yaml(path: Path) -> Dict:
    if not path.exists():
        return {}
    cfg = OmegaConf.load(path)
    data = OmegaConf.to_container(cfg, resolve=False)
    if isinstance(data, dict):
        data.pop("defaults", None)
        return data
    return {}


def _merge_dicts(*items: Dict) -> Dict:
    merged: Dict = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
    return merged


def _default_cfgs(autoencoder_checkpoint: Optional[str], ldm_checkpoint: Optional[str]) -> Tuple[DictConfig, DictConfig]:
    root = Path(__file__).resolve().parents[1]
    cfg_dir = root / "cfgs"

    dataset = _merge_dicts(
        _read_yaml(cfg_dir / "dataset" / "nuplan_base.yaml"),
        _read_yaml(cfg_dir / "dataset" / "nuplan_ldm.yaml"),
    )
    ae_dataset = _merge_dicts(
        _read_yaml(cfg_dir / "dataset" / "nuplan_base.yaml"),
        _read_yaml(cfg_dir / "dataset" / "nuplan_autoencoder.yaml"),
    )
    train = _merge_dicts(
        _read_yaml(cfg_dir / "train" / "base.yaml"),
        _read_yaml(cfg_dir / "train" / "nuplan_ldm.yaml"),
    )
    eval_cfg = _merge_dicts(
        _read_yaml(cfg_dir / "eval" / "base.yaml"),
        _read_yaml(cfg_dir / "eval" / "nuplan_ldm.yaml"),
    )
    ae_model = _merge_dicts(
        _read_yaml(cfg_dir / "model" / "autoencoder.yaml"),
        _read_yaml(cfg_dir / "model" / "nuplan_autoencoder.yaml"),
    )
    ldm_model = _merge_dicts(
        _read_yaml(cfg_dir / "model" / "ldm.yaml"),
        _read_yaml(cfg_dir / "model" / "nuplan_ldm.yaml"),
    )

    ae_model.update({
        "lane_conn_attr": int(ae_dataset["num_lane_connection_types"]),
        "num_lane_types": int(ae_dataset["num_lane_types"]),
        "num_agent_types": int(ae_dataset["num_agent_types"]),
        "num_points_per_lane": int(ae_dataset["num_points_per_lane"]),
        "max_num_lanes": int(ae_dataset["max_num_lanes"]),
    })
    ldm_model["autoencoder_path"] = autoencoder_checkpoint or ldm_model.get("autoencoder_path")

    cfg_ae = OmegaConf.create({
        "dataset_name": "nuplan",
        "dataset": ae_dataset,
        "train": _merge_dicts(_read_yaml(cfg_dir / "train" / "base.yaml"), _read_yaml(cfg_dir / "train" / "nuplan_autoencoder.yaml")),
        "eval": _merge_dicts(_read_yaml(cfg_dir / "eval" / "base.yaml"), _read_yaml(cfg_dir / "eval" / "nuplan_autoencoder.yaml")),
        "model": ae_model,
        "datamodule": {},
    })
    cfg_ldm = OmegaConf.create({
        "dataset_name": "nuplan",
        "dataset": dataset,
        "train": train,
        "eval": eval_cfg,
        "model": ldm_model,
        "datamodule": {},
    })
    if ldm_checkpoint:
        cfg_ldm.model.base_checkpoint = ldm_checkpoint
    return cfg_ldm, cfg_ae


def _extract_hparam_cfg(payload: Dict, keys: Sequence[str]) -> Optional[DictConfig]:
    hparams = payload.get("hyper_parameters", {}) if isinstance(payload, dict) else {}
    for key in keys:
        if key in payload:
            return OmegaConf.create(_to_plain(payload[key]))
        if key in hparams:
            return OmegaConf.create(_to_plain(hparams[key]))
    return None


def _set_cfg_attr(cfg: DictConfig, dotted_key: str, value) -> None:
    OmegaConf.set_struct(cfg, False)
    cursor = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or cursor[part] is None:
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    OmegaConf.set_struct(cfg, True)


def _state_dict_with_prefix(payload, prefix: str) -> Dict[str, torch.Tensor]:
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise TypeError("checkpoint does not contain a state dict")
    stripped = {
        key[len(prefix):]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix) and torch.is_tensor(value)
    }
    if stripped:
        return stripped
    raw = {key: value for key, value in state.items() if isinstance(key, str) and torch.is_tensor(value)}
    if raw:
        return raw
    raise ValueError(f"checkpoint has no tensor keys matching prefix {prefix!r}")


def _load_module_state(module: torch.nn.Module, payload: Dict, prefix: str, strict: bool = False) -> Dict:
    state = _state_dict_with_prefix(payload, prefix)
    missing, unexpected = module.load_state_dict(state, strict=strict)
    return {"missing": list(missing), "unexpected": list(unexpected)}


def _ensure_float(value, name: str) -> float:
    if value is None:
        raise ValueError(f"missing latent statistic: {name}")
    if isinstance(value, str) and "${" in value:
        raise ValueError(f"unresolved latent statistic interpolation: {name}={value}")
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0])
    return float(value)


def _resolve_latent_stats(cfg_ldm: DictConfig, checkpoint_dir: Optional[Path]) -> Dict[str, float]:
    dataset = cfg_ldm.dataset
    names = [
        "agent_latents_mean",
        "agent_latents_std",
        "lane_latents_mean",
        "lane_latents_std",
    ]
    if all(getattr(dataset, name, None) is not None and "${" not in str(getattr(dataset, name)) for name in names):
        return {name: _ensure_float(getattr(dataset, name), name) for name in names}

    stat_path = getattr(dataset, "latent_stats_path", None)
    candidate_paths = []
    if stat_path and "${" not in str(stat_path):
        candidate_paths.append(Path(os.path.expandvars(str(stat_path))))
    if checkpoint_dir is not None:
        candidate_paths.extend([
            checkpoint_dir / "latent_stats.pkl",
            checkpoint_dir.parent.parent / "metadata" / "latent_stats" / "scenario_dreamer_autoencoder_nuplan.pkl",
        ])
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidate_paths.append(Path(project_root) / "metadata" / "latent_stats" / "scenario_dreamer_autoencoder_nuplan.pkl")

    for path in candidate_paths:
        if path.exists():
            with path.open("rb") as f:
                stats = pickle.load(f)
            for name in names:
                _set_cfg_attr(cfg_ldm, f"dataset.{name}", _ensure_float(stats.get(name), name))
            return {name: _ensure_float(getattr(cfg_ldm.dataset, name), name) for name in names}

    raise FileNotFoundError(
        "Scenario Dreamer latent statistics were not found. Use an LDM checkpoint whose "
        "hyper_parameters.cfg.dataset contains agent/lane latent mean/std, or place "
        "scenario_dreamer_autoencoder_nuplan.pkl under metadata/latent_stats."
    )


def _prepare_cfgs(
        autoencoder_checkpoint: str,
        ldm_checkpoint: Optional[str],
        checkpoint_path: Optional[str],
        use_route_condition: bool,
        route_feature_dim: int,
        route_dropout: float) -> Tuple[DictConfig, DictConfig, Dict]:
    source_path = checkpoint_path or ldm_checkpoint
    source_payload = _torch_load(source_path) if source_path and Path(source_path).exists() else {}
    base_payload = _torch_load(ldm_checkpoint) if ldm_checkpoint and Path(ldm_checkpoint).exists() else {}
    ae_payload = _torch_load(autoencoder_checkpoint) if autoencoder_checkpoint and Path(autoencoder_checkpoint).exists() else {}

    cfg_ldm = (
        _extract_hparam_cfg(source_payload, ["cfg_ldm", "cfg"]) or
        _extract_hparam_cfg(base_payload, ["cfg_ldm", "cfg"]) or
        _default_cfgs(autoencoder_checkpoint, ldm_checkpoint)[0]
    )
    cfg_ae = (
        _extract_hparam_cfg(source_payload, ["cfg_ae"]) or
        _extract_hparam_cfg(base_payload, ["cfg_ae"]) or
        _extract_hparam_cfg(ae_payload, ["cfg"]) or
        _default_cfgs(autoencoder_checkpoint, ldm_checkpoint)[1]
    )

    for cfg in (cfg_ldm, cfg_ae):
        OmegaConf.set_struct(cfg, False)
    cfg_ldm.dataset_name = "nuplan"
    cfg_ae.dataset_name = "nuplan"
    cfg_ldm.model.autoencoder_path = autoencoder_checkpoint
    cfg_ldm.model.use_route_condition = bool(use_route_condition)
    cfg_ldm.model.route_feature_dim = int(route_feature_dim)
    cfg_ldm.model.route_dropout = float(route_dropout)
    cfg_ldm.dataset.num_agent_types = int(getattr(cfg_ldm.dataset, "num_agent_types", 3))
    cfg_ldm.dataset.num_lane_types = int(getattr(cfg_ldm.dataset, "num_lane_types", 3))
    cfg_ae.model.lane_conn_attr = int(getattr(cfg_ae.dataset, "num_lane_connection_types", 4))
    cfg_ae.model.num_lane_types = int(getattr(cfg_ae.dataset, "num_lane_types", 3))
    cfg_ae.model.num_agent_types = int(getattr(cfg_ae.dataset, "num_agent_types", 3))
    cfg_ae.model.num_points_per_lane = int(getattr(cfg_ae.dataset, "num_points_per_lane", 20))
    cfg_ae.model.max_num_lanes = int(getattr(cfg_ae.dataset, "max_num_lanes", 100))
    for cfg in (cfg_ldm, cfg_ae):
        OmegaConf.set_struct(cfg, True)

    stats = _resolve_latent_stats(cfg_ldm, Path(source_path).parent if source_path else None)
    return cfg_ldm, cfg_ae, {
        "source_payload": source_payload,
        "base_payload": base_payload,
        "ae_payload": ae_payload,
        "latent_stats": stats,
    }


class RouteConditionedHighValueLDM:
    """Fine-tune and sample Scenario Dreamer LDM for HV initial-state generation."""

    def __init__(
            self,
            checkpoint_path: Optional[str],
            autoencoder_checkpoint: str,
            ldm_checkpoint: Optional[str] = None,
            device: str = "cuda",
            mode: str = "inpainting",
            num_generate_agents: int = 2,
            route_num_points: int = 80,
            use_route_condition: bool = True,
            route_dropout: float = 0.0,
            decoded_state_loss_weight: float = 0.0,
            decoded_position_loss_weight: float = 1.0,
            decoded_heading_loss_weight: float = 0.5,
            decoded_speed_loss_weight: float = 0.5,
            decoded_size_loss_weight: float = 0.2,
            physical_bounds_loss_weight: float = 0.0,
            min_generated_speed: float = 0.0,
            max_generated_speed: float = 30.0,
            min_generated_length: float = 2.0,
            max_generated_length: float = 8.0,
            min_generated_width: float = 1.4,
            max_generated_width: float = 3.2,
            clamp_generated_states: bool = False,
            use_encoder_mean: bool = True,
            builder_cfg: Optional[Dict] = None):
        if mode not in ("inpainting", "full_generation"):
            raise ValueError(f"unsupported mode: {mode}")
        if not autoencoder_checkpoint:
            raise ValueError("autoencoder_checkpoint is required")
        if not Path(autoencoder_checkpoint).exists():
            raise FileNotFoundError(autoencoder_checkpoint)
        if checkpoint_path and not Path(checkpoint_path).exists():
            raise FileNotFoundError(checkpoint_path)
        if ldm_checkpoint and not Path(ldm_checkpoint).exists():
            raise FileNotFoundError(ldm_checkpoint)
        if float(min_generated_speed) > float(max_generated_speed):
            raise ValueError("min_generated_speed must be <= max_generated_speed")
        if float(min_generated_length) > float(max_generated_length):
            raise ValueError("min_generated_length must be <= max_generated_length")
        if float(min_generated_width) > float(max_generated_width):
            raise ValueError("min_generated_width must be <= max_generated_width")

        self.device = torch.device(device)
        self.mode = mode
        self.num_generate_agents = int(num_generate_agents)
        self.use_encoder_mean = bool(use_encoder_mean)
        self.decoded_state_loss_weight = float(decoded_state_loss_weight)
        self.decoded_position_loss_weight = float(decoded_position_loss_weight)
        self.decoded_heading_loss_weight = float(decoded_heading_loss_weight)
        self.decoded_speed_loss_weight = float(decoded_speed_loss_weight)
        self.decoded_size_loss_weight = float(decoded_size_loss_weight)
        self.physical_bounds_loss_weight = float(physical_bounds_loss_weight)
        self.generated_state_bounds = {
            "speed": (float(min_generated_speed), float(max_generated_speed)),
            "length": (float(min_generated_length), float(max_generated_length)),
            "width": (float(min_generated_width), float(max_generated_width)),
        }
        self.clamp_generated_states = bool(clamp_generated_states)
        self.cfg_ldm, self.cfg_ae, payloads = _prepare_cfgs(
            autoencoder_checkpoint=autoencoder_checkpoint,
            ldm_checkpoint=ldm_checkpoint,
            checkpoint_path=checkpoint_path,
            use_route_condition=use_route_condition,
            route_feature_dim=5,
            route_dropout=route_dropout,
        )
        self.agent_latent_dim = int(self.cfg_ldm.model.agent_latent_dim)
        self.lane_latent_dim = int(self.cfg_ldm.model.lane_latent_dim)
        graph_cfg = {
            "fov": float(getattr(self.cfg_ldm.dataset, "fov", NUPLAN_STATS["fov"])),
            "num_points_per_lane": int(getattr(self.cfg_ldm.dataset, "num_points_per_lane", 20)),
            "max_num_lanes": int(getattr(self.cfg_ldm.dataset, "max_num_lanes", 100)),
            "max_num_agents": int(getattr(self.cfg_ldm.dataset, "max_num_agents", 61)),
            "route_num_points": int(route_num_points),
        }
        if builder_cfg:
            graph_cfg.update(builder_cfg)
        self.graph_builder = HighValueSceneGraphBuilder(graph_cfg)

        self.autoencoder = AutoEncoder(self.cfg_ae.model).to(self.device)
        ae_payload = payloads["ae_payload"] or _torch_load(autoencoder_checkpoint)
        self.ae_load_info = _load_module_state(self.autoencoder, ae_payload, prefix="model.", strict=False)
        self.autoencoder.eval()
        for param in self.autoencoder.parameters():
            param.requires_grad_(False)

        self.ldm = LDM(self.cfg_ldm).to(self.device)
        load_payload = {}
        if payloads["source_payload"].get("ldm_state_dict"):
            load_payload = payloads["source_payload"]["ldm_state_dict"]
        elif payloads["source_payload"]:
            load_payload = payloads["source_payload"]
        elif payloads["base_payload"]:
            load_payload = payloads["base_payload"]
        if not load_payload:
            raise ValueError("ldm checkpoint is required to initialize RouteConditionedHighValueLDM")
        if isinstance(load_payload, dict) and "ldm_state_dict" in load_payload:
            load_payload = load_payload["ldm_state_dict"]
        self.ldm_load_info = _load_module_state(self.ldm, load_payload, prefix="diff_model.", strict=False)

    def train(self):
        self.ldm.train()
        self.autoencoder.eval()
        return self

    def eval(self):
        self.ldm.eval()
        self.autoencoder.eval()
        return self

    def parameters(self):
        return self.ldm.parameters()

    def training_loss(self, records: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.build_training_batch(records)
        loss_dict = self.ldm.loss(batch)
        if self.decoded_state_loss_weight > 0.0:
            decoded = self._decoded_state_loss(batch)
            loss_dict["loss"] = loss_dict["loss"] + self.decoded_state_loss_weight * decoded["decoded_state_loss"]
            loss_dict.update(decoded)
        else:
            zero = loss_dict["loss"].detach() * 0.0
            loss_dict.update({
                "decoded_state_loss": zero,
                "decoded_position_l1": zero,
                "decoded_heading_l1": zero,
                "decoded_speed_l1": zero,
                "decoded_size_l1": zero,
                "decoded_physical_bounds_loss": zero,
            })
        return loss_dict

    @torch.no_grad()
    def build_training_batch(self, records: Sequence[Dict]) -> Batch:
        self._validate_records(records, require_targets=True)
        full_data_list = []
        cond_data_list = []
        metas = []
        for record in records:
            full_data, meta = self.graph_builder.build_ae_data(record, include_targets=True)
            cond_data, _ = self.graph_builder.build_ae_data(record, include_targets=False)
            if int(meta["target_agent_count"]) <= 0:
                raise ValueError(f"record {record.get('scene_id')} has no target generated agents")
            full_data_list.append(full_data)
            cond_data_list.append(cond_data)
            metas.append(meta)

        full_batch = Batch.from_data_list(full_data_list).to(self.device)
        cond_batch = Batch.from_data_list(cond_data_list).to(self.device)
        full_agent_latents, _ = self._encode_normalized(full_batch)
        cond_agent_latents, cond_lane_latents = self._encode_normalized(cond_batch)

        ldm_data_list = []
        for batch_idx, record in enumerate(records):
            seed, meta = self.graph_builder.build_ldm_seed(record, self.agent_latent_dim, self.lane_latent_dim)
            current_full_data = full_data_list[batch_idx]
            context_count = int(meta["context_agent_count"])
            target_count = int(meta["target_agent_count"])
            full_agent_i = full_agent_latents[full_batch["agent"].batch == batch_idx].detach().cpu()
            cond_agent_i = cond_agent_latents[cond_batch["agent"].batch == batch_idx].detach().cpu()
            cond_lane_i = cond_lane_latents[cond_batch["lane"].batch == batch_idx].detach().cpu()
            seed["agent"].latents = torch.zeros_like(seed["agent"].x)
            seed["lane"].latents = torch.zeros_like(seed["lane"].x)
            seed["agent"].target_state = torch.zeros(
                (seed["agent"].x.shape[0], current_full_data["agent"].x.shape[-1]),
                dtype=torch.float32,
            )
            seed["agent"].target_state_mask = torch.zeros(seed["agent"].x.shape[0], dtype=torch.bool)
            seed["agent"].latents[:context_count] = cond_agent_i[:context_count]
            seed["agent"].latents[context_count:context_count + target_count] = full_agent_i[context_count:context_count + target_count]
            seed["lane"].latents[:cond_lane_i.shape[0]] = cond_lane_i
            seed["agent"].target_state[context_count:context_count + target_count] = (
                current_full_data["agent"].x[context_count:context_count + target_count]
            )
            seed["agent"].target_state_mask[context_count:context_count + target_count] = True
            ldm_data_list.append(seed)

        return Batch.from_data_list(ldm_data_list).to(self.device)

    @torch.no_grad()
    def generate(self, record: Dict, num_generate_agents: Optional[int] = None) -> Dict:
        self.eval()
        record = self._with_generation_count(record, num_generate_agents)
        self._validate_records([record], require_targets=False)
        cond_data, _ = self.graph_builder.build_ae_data(record, include_targets=False)
        cond_batch = Batch.from_data_list([cond_data]).to(self.device)
        cond_agent_latents, cond_lane_latents = self._encode_normalized(cond_batch)

        seed, meta = self.graph_builder.build_ldm_seed(record, self.agent_latent_dim, self.lane_latent_dim)
        context_count = int(meta["context_agent_count"])
        target_count = int(meta["target_agent_count"])
        seed["agent"].latents = torch.zeros_like(seed["agent"].x)
        seed["lane"].latents = torch.zeros_like(seed["lane"].x)
        seed["agent"].latents[:context_count] = cond_agent_latents.detach().cpu()[:context_count]
        seed["lane"].latents[:cond_lane_latents.shape[0]] = cond_lane_latents.detach().cpu()
        data = Batch.from_data_list([seed]).to(self.device)

        agent_latents, lane_latents = self.ldm.forward(data, mode="inpainting")
        agent_latents, lane_latents = self._unnormalize_latents(agent_latents, lane_latents)
        agent_samples, lane_samples, agent_types, lane_types, lane_conn = self.autoencoder.forward_decoder(
            agent_latents,
            lane_latents,
            data,
        )
        # Strict raw S1 deployment: inverse normalization is mathematical, not
        # clipping. Any off-range x/y/speed remains visible and can fail in CARLA.
        agent_samples = self._unnormalize_agents_raw_for_spawn(agent_samples)
        generated = decoded_sd_agents_to_hv_states(agent_samples, context_count, target_count)
        if self.clamp_generated_states:
            raise ValueError("raw-state generation forbids post-sample clamping")
        return {
            "schema_version": "hv_init_generated_scene_v1",
            "generation_mode": self.mode,
            "num_generated_agents": int(target_count),
            "generated_agent_states": generated.detach().cpu().tolist(),
            "generated_agent_types": [1 for _ in range(int(target_count))],
            "model": SCHEMA_VERSION,
        }

    @torch.no_grad()
    def reconstruct_targets(self, record: Dict) -> Dict:
        """Generate once and compare against record targets without using RIFT labels."""
        generated = self.generate(record)
        target = torch.tensor(self.graph_builder._target_states(record), dtype=torch.float32)
        pred = torch.tensor(generated["generated_agent_states"], dtype=torch.float32)
        metrics = match_state_metrics(pred, target, bounds=self.generated_state_bounds)
        return {"generated": generated, "metrics": metrics}

    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "ldm_state_dict": self.ldm.state_dict(),
            "cfg_ldm": OmegaConf.to_container(self.cfg_ldm, resolve=False),
            "cfg_ae": OmegaConf.to_container(self.cfg_ae, resolve=False),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def _encode_normalized(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        agent_mu, lane_mu, agent_log_var, lane_log_var = self.autoencoder.forward_encoder(data, return_stats=True)
        if self.use_encoder_mean:
            agent_latents, lane_latents = agent_mu, lane_mu
        else:
            agent_latents = self._reparameterize(agent_mu, agent_log_var)
            lane_latents = self._reparameterize(lane_mu, lane_log_var)
        return normalize_latents(
            agent_latents,
            lane_latents,
            self.cfg_ldm.dataset.agent_latents_mean,
            self.cfg_ldm.dataset.agent_latents_std,
            self.cfg_ldm.dataset.lane_latents_mean,
            self.cfg_ldm.dataset.lane_latents_std,
        )

    def _unnormalize_latents(self, agent_latents: torch.Tensor, lane_latents: torch.Tensor):
        return unnormalize_latents(
            agent_latents,
            lane_latents,
            self.cfg_ldm.dataset.agent_latents_mean,
            self.cfg_ldm.dataset.agent_latents_std,
            self.cfg_ldm.dataset.lane_latents_mean,
            self.cfg_ldm.dataset.lane_latents_std,
        )

    def _unnormalize_agents_raw_for_spawn(self, values: torch.Tensor) -> torch.Tensor:
        """Inverse-transform AE agent states without clipping any predicted field.

        AE format is [x,y,speed,cos_yaw,sin_yaw,length,width].  Fixed
        blueprint dimensions are handled by CARLA, not this decoder.
        """
        cfg = self.cfg_ldm.dataset
        fov = float(cfg.fov)
        def inv(v, lo, hi):
            return (v + 1.0) * (float(hi) - float(lo)) / 2.0 + float(lo)
        return torch.stack([
            inv(values[:, 0], -fov / 2.0, fov / 2.0),
            inv(values[:, 1], -fov / 2.0, fov / 2.0),
            inv(values[:, 2], cfg.min_speed, cfg.max_speed),
            values[:, 3],
            values[:, 4],
            inv(values[:, 5], cfg.min_length, cfg.max_length),
            inv(values[:, 6], cfg.min_width, cfg.max_width),
        ], dim=-1)

    def _unnormalize_scene(self, agent_samples: torch.Tensor, lane_samples: torch.Tensor):
        return unnormalize_scene(
            agent_samples,
            lane_samples,
            fov=float(self.cfg_ldm.dataset.fov),
            min_speed=float(self.cfg_ldm.dataset.min_speed),
            max_speed=float(self.cfg_ldm.dataset.max_speed),
            min_length=float(self.cfg_ldm.dataset.min_length),
            max_length=float(self.cfg_ldm.dataset.max_length),
            min_width=float(self.cfg_ldm.dataset.min_width),
            max_width=float(self.cfg_ldm.dataset.max_width),
            min_lane_x=float(self.cfg_ldm.dataset.min_lane_x),
            min_lane_y=float(self.cfg_ldm.dataset.min_lane_y),
            max_lane_x=float(self.cfg_ldm.dataset.max_lane_x),
            max_lane_y=float(self.cfg_ldm.dataset.max_lane_y),
        )

    def _decoded_state_loss(self, data: Batch) -> Dict[str, torch.Tensor]:
        x_agent = data["agent"].latents.unsqueeze(1)
        x_lane = data["lane"].latents.unsqueeze(1)
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        timesteps = torch.randint(0, self.ldm.n_timesteps, (data.batch_size,), device=x_agent.device).long()
        t_agent = timesteps[agent_batch]
        t_lane = timesteps[lane_batch]

        agent_noise = torch.randn_like(x_agent)
        lane_noise = torch.randn_like(x_lane)
        x_agent_noisy = self.ldm.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)
        x_lane_noisy = self.ldm.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)

        agent_condition_mask = data["agent"].partition_mask == BEFORE_PARTITION
        lane_condition_mask = data["lane"].partition_mask == BEFORE_PARTITION
        x_agent_noisy[agent_condition_mask] = x_agent[agent_condition_mask]
        x_lane_noisy[lane_condition_mask] = x_lane[lane_condition_mask]

        agent_noise_pred, lane_noise_pred = self.ldm.model(x_agent_noisy, x_lane_noisy, data, t_agent, t_lane)
        pred_agent_latents = self.ldm.predict_start_from_noise(x_agent_noisy, t=t_agent, noise=agent_noise_pred)[:, 0].clone()
        pred_lane_latents = self.ldm.predict_start_from_noise(x_lane_noisy, t=t_lane, noise=lane_noise_pred)[:, 0].clone()
        pred_agent_latents[agent_condition_mask] = data["agent"].latents[agent_condition_mask]
        pred_lane_latents[lane_condition_mask] = data["lane"].latents[lane_condition_mask]

        agent_latents, lane_latents = self._unnormalize_latents(pred_agent_latents, pred_lane_latents)
        agent_samples, _, _, _, _ = self.autoencoder.forward_decoder(agent_latents, lane_latents, data)
        return self._set_matched_state_loss(agent_samples, data)

    def _set_matched_state_loss(self, agent_samples: torch.Tensor, data: Batch) -> Dict[str, torch.Tensor]:
        state_target_mask = data["agent"].target_state_mask
        if not bool(state_target_mask.any()):
            zero = agent_samples.sum() * 0.0
            return {
                "decoded_state_loss": zero,
                "decoded_position_l1": zero.detach(),
                "decoded_heading_l1": zero.detach(),
                "decoded_speed_l1": zero.detach(),
                "decoded_size_l1": zero.detach(),
                "decoded_physical_bounds_loss": zero.detach(),
            }

        losses = []
        pos_losses = []
        heading_losses = []
        speed_losses = []
        size_losses = []
        physical_bounds_losses = []
        batch = data["agent"].batch
        for scene_idx in range(int(data.batch_size)):
            scene_target_mask = state_target_mask & (batch == scene_idx)
            pred = agent_samples[scene_target_mask]
            target = data["agent"].target_state[scene_target_mask]
            if pred.numel() == 0:
                continue
            pred = self._match_prediction_order(pred, target)
            pos = torch.abs(pred[:, :2] - target[:, :2]).mean()
            speed = torch.abs(pred[:, 2:3] - target[:, 2:3]).mean()
            heading = torch.abs(pred[:, 3:5] - target[:, 3:5]).mean()
            size = torch.abs(pred[:, 5:7] - target[:, 5:7]).mean()
            physical = self._physical_bounds_loss(pred)
            losses.append(
                self.decoded_position_loss_weight * pos +
                self.decoded_speed_loss_weight * speed +
                self.decoded_heading_loss_weight * heading +
                self.decoded_size_loss_weight * size +
                self.physical_bounds_loss_weight * physical
            )
            pos_losses.append(pos.detach())
            heading_losses.append(heading.detach())
            speed_losses.append(speed.detach())
            size_losses.append(size.detach())
            physical_bounds_losses.append(physical.detach())

        if not losses:
            zero = agent_samples.sum() * 0.0
            return {
                "decoded_state_loss": zero,
                "decoded_position_l1": zero.detach(),
                "decoded_heading_l1": zero.detach(),
                "decoded_speed_l1": zero.detach(),
                "decoded_size_l1": zero.detach(),
                "decoded_physical_bounds_loss": zero.detach(),
            }
        return {
            "decoded_state_loss": torch.stack(losses).mean(),
            "decoded_position_l1": torch.stack(pos_losses).mean(),
            "decoded_heading_l1": torch.stack(heading_losses).mean(),
            "decoded_speed_l1": torch.stack(speed_losses).mean(),
            "decoded_size_l1": torch.stack(size_losses).mean(),
            "decoded_physical_bounds_loss": torch.stack(physical_bounds_losses).mean(),
        }

    def _physical_bounds_loss(self, pred: torch.Tensor) -> torch.Tensor:
        if pred.numel() == 0:
            return pred.sum() * 0.0
        speed_min, speed_max = self.generated_state_bounds["speed"]
        length_min, length_max = self.generated_state_bounds["length"]
        width_min, width_max = self.generated_state_bounds["width"]
        speed = self._unnormalize_scalar(
            pred[:, 2],
            float(self.cfg_ldm.dataset.min_speed),
            float(self.cfg_ldm.dataset.max_speed),
        )
        length = self._unnormalize_scalar(
            pred[:, 5],
            float(self.cfg_ldm.dataset.min_length),
            float(self.cfg_ldm.dataset.max_length),
        )
        width = self._unnormalize_scalar(
            pred[:, 6],
            float(self.cfg_ldm.dataset.min_width),
            float(self.cfg_ldm.dataset.max_width),
        )
        violations = [
            torch.relu(float(speed_min) - speed),
            torch.relu(speed - float(speed_max)),
            torch.relu(float(length_min) - length),
            torch.relu(length - float(length_max)),
            torch.relu(float(width_min) - width),
            torch.relu(width - float(width_max)),
        ]
        return torch.stack([item.mean() for item in violations]).mean()

    @staticmethod
    def _unnormalize_scalar(value: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
        return ((torch.clamp(value, -1.0, 1.0) + 1.0) / 2.0) * (maximum - minimum) + minimum

    def _clamp_hv_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.numel() == 0:
            return states
        states = states.clone()
        speed_min, speed_max = self.generated_state_bounds["speed"]
        length_min, length_max = self.generated_state_bounds["length"]
        width_min, width_max = self.generated_state_bounds["width"]
        states[:, 3] = torch.clamp(states[:, 3], float(speed_min), float(speed_max))
        states[:, 4] = torch.clamp(states[:, 4], float(length_min), float(length_max))
        states[:, 5] = torch.clamp(states[:, 5], float(width_min), float(width_max))
        return states

    @staticmethod
    def _match_prediction_order(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = int(min(pred.shape[0], target.shape[0]))
        pred = pred[:n]
        target = target[:n]
        if n <= 1:
            return pred
        with torch.no_grad():
            if n <= 8:
                best_perm = tuple(range(n))
                best_cost = None
                for perm in itertools.permutations(range(n)):
                    cost = torch.linalg.norm(pred[list(perm), :2] - target[:, :2], dim=-1).mean()
                    value = float(cost.detach().cpu())
                    if best_cost is None or value < best_cost:
                        best_cost = value
                        best_perm = perm
                order = list(best_perm)
            else:
                remaining = list(range(n))
                order = []
                for idx in range(n):
                    dist = torch.linalg.norm(pred[remaining, :2] - target[idx:idx + 1, :2], dim=-1)
                    local = int(torch.argmin(dist).detach().cpu())
                    order.append(remaining.pop(local))
        return pred[order]

    def _with_generation_count(self, record: Dict, count: Optional[int]) -> Dict:
        if count is None:
            count = self.num_generate_agents
        record = dict(record)
        condition = dict(record.get("condition", {}))
        condition["num_generate_agents"] = int(count)
        record["condition"] = condition
        return record

    def _validate_records(self, records: Sequence[Dict], require_targets: bool) -> None:
        for record in records:
            if not state_generation_schema_is_clean(record):
                raise ValueError(f"selector/delta field leaked into record {record.get('scene_id')}")
            if "condition" not in record:
                raise ValueError(f"record {record.get('scene_id')} has no condition")
            if require_targets and not self.graph_builder._target_states(record):
                raise ValueError(f"record {record.get('scene_id')} has no target agent states")

    @staticmethod
    def _reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std


def match_state_metrics(pred: torch.Tensor, target: torch.Tensor, bounds: Optional[Dict] = None) -> Dict[str, float]:
    if pred.numel() == 0 or target.numel() == 0:
        return {
            "position_l2": float("nan"),
            "heading_abs": float("nan"),
            "speed_l1": float("nan"),
            "size_l1": float("nan"),
            "length_l1": float("nan"),
            "width_l1": float("nan"),
            "physical_invalid_count": 0,
            "physical_invalid_rate": float("nan"),
            "fov_boundary_saturation_rate": float("nan"),
        }
    n = min(pred.shape[0], target.shape[0])
    pred = pred[:n]
    target = target[:n]
    if n <= 8:
        best_perm = None
        best_cost = None
        for perm in itertools.permutations(range(n)):
            cost = torch.linalg.norm(pred[list(perm), :2] - target[:n, :2], dim=-1).mean()
            if best_cost is None or float(cost) < best_cost:
                best_cost = float(cost)
                best_perm = perm
        pred = pred[list(best_perm)]
    else:
        remaining = list(range(n))
        order = []
        for idx in range(n):
            d = torch.linalg.norm(pred[remaining, :2] - target[idx:idx + 1, :2], dim=-1)
            local = int(torch.argmin(d).item())
            order.append(remaining.pop(local))
        pred = pred[order]
    heading = torch.atan2(torch.sin(pred[:, 2] - target[:, 2]), torch.cos(pred[:, 2] - target[:, 2])).abs()
    bounds = bounds or {
        "speed": (0.0, 30.0),
        "length": (2.0, 8.0),
        "width": (1.4, 3.2),
    }
    speed_min, speed_max = bounds["speed"]
    length_min, length_max = bounds["length"]
    width_min, width_max = bounds["width"]
    speed = pred[:, 3]
    length = pred[:, 4]
    width = pred[:, 5]
    invalid = (
        (speed < float(speed_min)) | (speed > float(speed_max)) |
        (length < float(length_min)) | (length > float(length_max)) |
        (width < float(width_min)) | (width > float(width_max))
    )
    near_boundary = (pred[:, :2].abs() >= 31.0).any(dim=-1)
    return {
        "position_l2": float(torch.linalg.norm(pred[:, :2] - target[:, :2], dim=-1).mean().item()),
        "heading_abs": float(heading.mean().item()),
        "speed_l1": float(torch.abs(pred[:, 3] - target[:, 3]).mean().item()),
        "size_l1": float(torch.abs(pred[:, 4:6] - target[:, 4:6]).mean().item()),
        "length_l1": float(torch.abs(pred[:, 4] - target[:, 4]).mean().item()),
        "width_l1": float(torch.abs(pred[:, 5] - target[:, 5]).mean().item()),
        "physical_invalid_count": int(invalid.sum().item()),
        "physical_invalid_rate": float(invalid.float().mean().item()),
        "fov_boundary_saturation_rate": float(near_boundary.float().mean().item()),
    }
