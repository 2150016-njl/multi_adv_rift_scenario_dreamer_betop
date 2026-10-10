"""Compose a pure Scenario Dreamer background scene with HV-generated actors.

This script does not import CARLA or RIFT. It is an offline composition tool,
not a substitute for closed-loop RIFT-CBV evaluation.
"""

import argparse
import copy
import json
from pathlib import Path

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--record_index", type=int, default=0)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", required=True)
    parser.add_argument("--background_checkpoint", default=None,
                        help="Optional Scenario Dreamer-compatible background fine-tune. Omit to use the base LDM.")
    parser.add_argument("--hv_checkpoint", required=True)
    parser.add_argument("--background_num_agents", type=int, default=6)
    parser.add_argument("--hv_num_agents", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def model(checkpoint, args, count, use_route_condition):
    return RouteConditionedHighValueLDM(
        checkpoint_path=checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode="inpainting",
        num_generate_agents=count,
        route_num_points=80,
        use_route_condition=use_route_condition,
        clamp_generated_states=False,
    )


def base_record(record, count, observed):
    item = copy.deepcopy(record)
    item["target"] = {}
    item["condition"]["num_generate_agents"] = int(count)
    item["condition"]["observed_agent_states"] = observed
    item["condition"]["observed_agent_types"] = [1 for _ in observed]
    item["schema_version"] = "hv_init_inpainting_composition_v1"
    return item


def main():
    args = parse_args()
    if args.background_num_agents < 0 or args.hv_num_agents <= 0:
        raise ValueError("background count must be >=0 and HV count must be >0")
    if args.hv_num_agents != 2:
        raise ValueError("the HV checkpoint is trained and validated for exactly two high-value actors")
    records = read_jsonl(args.jsonl)
    if args.record_index < 0 or args.record_index >= len(records):
        raise IndexError("--record_index is outside the JSONL range")
    record = records[args.record_index]
    # The base Scenario Dreamer LDM was not trained with this project's added
    # route conditioner. Keep it disabled unless a compatible background
    # fine-tune is explicitly supplied.
    background_model = model(
        args.background_checkpoint,
        args,
        args.background_num_agents,
        use_route_condition=bool(args.background_checkpoint),
    )
    background_record = base_record(record, args.background_num_agents, [])
    background = background_model.generate(background_record, args.background_num_agents)
    background_states = background["generated_agent_states"]
    hv_model = model(args.hv_checkpoint, args, args.hv_num_agents, use_route_condition=True)
    hv_record = base_record(record, args.hv_num_agents, background_states)
    high_value = hv_model.generate(hv_record, args.hv_num_agents)
    output = {
        "schema_version": "scenario_dreamer_hv_composed_scene_v1",
        "source_record": {"scene_id": record.get("scene_id"), "town": record.get("town"), "record_index": args.record_index},
        "coordinate_frame": "ego_local_rh_x_forward",
        "condition": {"ego_state": record["condition"]["ego_state"], "ego_route": record["condition"]["ego_route"], "map": record["condition"].get("map", {})},
        "background_agents": [{"role": "scenario_dreamer_background", "render_color": "blue", "state": state} for state in background_states],
        "high_value_agents": [{"role": "high_value_candidate", "render_color": "orange", "state": state} for state in high_value["generated_agent_states"]],
        "limitations": [
            "No CARLA actor is spawned.",
            "Blue/orange are offline visualization groups, not CARLA BV/CBV registrations.",
            "Purple/green RIFT-CBV status requires the CARLA/RIFT closed-loop recognizer.",
        ],
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_json": args.output_json,
        "background_agents": len(background_states),
        "high_value_agents": len(high_value["generated_agent_states"]),
    }, indent=2))


if __name__ == "__main__":
    main()
