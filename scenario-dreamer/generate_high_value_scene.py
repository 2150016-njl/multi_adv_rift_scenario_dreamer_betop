#!/usr/bin/env python
"""Generate high-value initial-state samples from JSONL conditions."""

import argparse
import json
from pathlib import Path

from high_value_init.high_value_ldm import RouteConditionedHighValueLDM
from high_value_init.rift_dataset import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inpainting", "full_generation"], default="inpainting")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--autoencoder_checkpoint", required=True)
    parser.add_argument("--ldm_checkpoint", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--route_num_points", type=int, default=80)
    parser.add_argument("--num_samples_per_condition", type=int, default=8)
    parser.add_argument("--max_records", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def safe_name(value) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def main():
    args = parse_args()
    records = read_jsonl(args.jsonl)
    if args.max_records and args.max_records > 0:
        records = records[:args.max_records]
    if not records:
        raise ValueError(f"no records in {args.jsonl}")

    model = RouteConditionedHighValueLDM(
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        ldm_checkpoint=args.ldm_checkpoint,
        device=args.device,
        mode=args.mode,
        num_generate_agents=args.num_generate_agents,
        route_num_points=args.route_num_points,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as index_file:
        for idx, record in enumerate(records):
            scene_id = safe_name(record.get("scene_id", idx))
            for sample_idx in range(args.num_samples_per_condition):
                generated = model.generate(record, num_generate_agents=args.num_generate_agents)
                payload = {
                    "schema_version": "hv_init_generated_offline_sample_v1",
                    "scene_id": record.get("scene_id"),
                    "sample_idx": sample_idx,
                    "condition": record.get("condition"),
                    "generated": generated,
                    "source_record": {
                        "idx": record.get("idx", idx),
                        "episode_key": record.get("episode_key"),
                        "town": record.get("town"),
                    },
                }
                path = output_dir / f"{idx:06d}_{scene_id}_sample{sample_idx:02d}.json"
                with path.open("w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                index_file.write(json.dumps({"path": str(path), "scene_id": record.get("scene_id"), "sample_idx": sample_idx}) + "\n")
    print(json.dumps({
        "output_dir": str(output_dir),
        "index_jsonl": str(index_path),
        "num_conditions": len(records),
        "num_samples": len(records) * args.num_samples_per_condition,
    }, indent=2))


if __name__ == "__main__":
    main()

