#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Generator loading utilities for high-value initial-state generation.
"""

import importlib
from typing import Dict, Optional


class HighValueInitGeneratorError(RuntimeError):
    pass


class ExternalHighValueInitGenerator:
    """Load a generator implementation from ``module:Class``.

    The loaded class must expose ``generate(scene: dict) -> dict``. The returned
    dict is expected to contain actor-aligned scores/masks and absolute
    ``state_updates`` for the selected existing actors.
    """

    def __init__(
            self,
            entry_point: str,
            checkpoint_path: Optional[str] = None,
            device: str = "cuda",
            config: Optional[Dict] = None):
        if not entry_point:
            raise HighValueInitGeneratorError(
                "high_value_init is enabled but hv_init_generator_entry is empty. "
                "Pass a Scenario Dreamer generator entry such as "
                "'high_value_init.rift_generator:ScenarioDreamerHighValueInitGenerator'."
            )
        if ":" not in entry_point:
            raise HighValueInitGeneratorError(
                "hv_init_generator_entry must be in 'module:ClassName' format."
            )

        module_name, class_name = entry_point.split(":", 1)
        module = importlib.import_module(module_name)
        generator_cls = getattr(module, class_name)
        self.impl = generator_cls(
            checkpoint_path=checkpoint_path,
            device=device,
            config=config or {}
        )

    def generate(self, scene: Dict) -> Dict:
        if hasattr(self.impl, "generate"):
            return self.impl.generate(scene)
        if callable(self.impl):
            return self.impl(scene)
        raise HighValueInitGeneratorError("The loaded high-value generator is not callable.")


class DebugTopInteractionGenerator:
    """Smoke-test generator. Do not use it for reported experiments."""

    def __init__(self, *_args, **_kwargs):
        pass

    def generate(self, scene: Dict) -> Dict:
        labels = scene["interaction_labels"]
        ranked = sorted(
            enumerate(labels),
            key=lambda item: (item[1].get("is_interactive", False), item[1].get("score", 0.0)),
            reverse=True
        )
        target_mask = [False for _ in scene["actors"]]
        for idx, label in ranked[:scene.get("target_count", 2)]:
            if not label.get("is_interactive", False):
                break
            target_mask[idx] = True

        selected_actor_ids = []
        state_updates = {}
        for idx, selected in enumerate(target_mask):
            if selected:
                actor = scene["actors"][idx]
                actor_id = int(actor["actor_id"])
                selected_actor_ids.append(actor_id)
                state_updates[str(actor_id)] = {
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
        return {
            "target_mask": target_mask,
            "selected_actor_ids": selected_actor_ids,
            "cbv_actor_ids": selected_actor_ids[:scene.get("target_count", 2)],
            "state_updates": state_updates,
            "debug": True,
        }


def build_high_value_generator(config: Dict):
    if config.get("hv_init_debug_generator", False):
        return DebugTopInteractionGenerator()
    return ExternalHighValueInitGenerator(
        entry_point=config.get("hv_init_generator_entry"),
        checkpoint_path=config.get("hv_init_checkpoint"),
        device=config.get("hv_init_device", "cuda"),
        config=config,
    )
