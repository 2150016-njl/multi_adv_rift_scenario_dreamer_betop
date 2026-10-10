"""Post-build audit for the V4B balanced split and pair-feasibility contract."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import audit_high_value_scene_dataset_v4 as v4a


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--val_jsonl", type=Path, required=True)
    parser.add_argument("--test_jsonl", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--forward_min_x", type=float, default=0.0)
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--max_town_fraction", type=float, default=0.35)
    parser.add_argument("--max_topology_fraction", type=float, default=0.75)
    parser.add_argument("--max_split_town_fraction", type=float, default=0.45)
    parser.add_argument("--max_split_topology_fraction", type=float, default=0.80)
    parser.add_argument("--min_target_obb_axis_clearance", type=float, default=0.5)
    parser.add_argument(
        "--expected_target_pair_policy",
        choices=("high_value_pair", "route_diversity"),
        default="high_value_pair",
        help="The V4B build-only target-pair policy expected in every record.",
    )
    parser.add_argument("--min_records_per_town_per_split", type=int, default=0)
    parser.add_argument("--min_records_per_topology_per_split", type=int, default=0)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def _counts(rows, field):
    if field == "town":
        return Counter(str(row.get("town") or "unknown") for row in rows)
    if field == "topology":
        return Counter(str((row.get("topology_v4") or {}).get("bucket") or "unknown") for row in rows)
    return Counter(str((row.get("v4b_contract") or {}).get("target_region_pair") or "unknown") for row in rows)


def main():
    args = parse_args()
    base_args = argparse.Namespace(
        fov=args.fov, forward_min_x=args.forward_min_x,
        target_fov_margin=args.target_fov_margin, max_num_agents=args.max_num_agents,
        max_target_lane_distance=args.max_target_lane_distance,
    )
    split_rows = [("train", v4a.read_jsonl(args.train_jsonl)),
                  ("val", v4a.read_jsonl(args.val_jsonl)),
                  ("test", v4a.read_jsonl(args.test_jsonl))]
    all_rows = [row for _, rows in split_rows for row in rows]
    errors, examples, episodes = Counter(), [], defaultdict(set)
    for split, rows in split_rows:
        for index, row in enumerate(rows):
            row_errors = list(v4a.audit_record(row, split, base_args))
            contract = row.get("v4b_contract") or {}
            if row.get("dataset_version") != "v4b_group_stratified":
                row_errors.append("dataset_version_mismatch")
            if "_v4b_build_only_target_interaction" in row:
                row_errors.append("build_only_interaction_label_leaked")
            if contract.get("target_pair_selection") != args.expected_target_pair_policy:
                row_errors.append("v4b_pair_selection_contract_missing")
            try:
                clearance = float(contract.get("target_pair_obb_axis_clearance"))
            except (TypeError, ValueError):
                clearance = float("-inf")
            if clearance < args.min_target_obb_axis_clearance:
                row_errors.append("target_pair_obb_clearance_below_contract")
            if not contract.get("target_region_pair"):
                row_errors.append("target_region_pair_missing")
            for error in set(row_errors):
                errors[error] += 1
            if row_errors and len(examples) < 30:
                examples.append({"split": split, "index": index, "scene_id": row.get("scene_id"),
                                 "errors": sorted(set(row_errors))})
            episodes[str(row.get("episode_key"))].add(split)

    for episode, memberships in episodes.items():
        if len(memberships) > 1:
            errors["episode_split_leakage"] += 1
            if len(examples) < 30:
                examples.append({"episode_key": episode, "splits": sorted(memberships),
                                 "errors": ["episode_split_leakage"]})

    totals = {field: _counts(all_rows, field) for field in ("town", "topology")}
    split_summary = {}
    for split, rows in split_rows:
        summary = {"records": len(rows), "episodes": len({str(row.get("episode_key")) for row in rows})}
        if not rows:
            errors["empty_split:%s" % split] += 1
        for field in ("town", "topology", "target_region_pair"):
            counts = _counts(rows, field)
            summary[field + "s"] = dict(sorted(counts.items()))
            if rows:
                maximum = args.max_split_town_fraction if field == "town" else args.max_split_topology_fraction if field == "topology" else 1.0
                for key, count in counts.items():
                    if count / len(rows) > maximum + 1e-12:
                        errors["split_%s_fraction_exceeded:%s:%s" % (field, split, key)] += 1
            if field in totals:
                minimum = args.min_records_per_town_per_split if field == "town" else args.min_records_per_topology_per_split
                if minimum:
                    for key in totals[field]:
                        if counts[key] < minimum:
                            errors["%s_split_coverage:%s:%s" % (field, split, key)] += 1
        split_summary[split] = summary

    overall = {field: dict(sorted(_counts(all_rows, field).items())) for field in ("town", "topology", "target_region_pair")}
    for field, maximum in (("town", args.max_town_fraction), ("topology", args.max_topology_fraction)):
        for key, count in _counts(all_rows, field).items():
            if all_rows and count / len(all_rows) > maximum + 1e-12:
                errors["%s_fraction_exceeded:%s" % (field, key)] += 1

    report = {
        "schema_version": "hv_scene_dataset_v4b_audit_v1", "passed": not errors,
        "num_records": len(all_rows), "episode_count": len(episodes),
        "overall": overall, "split_summary": split_summary,
        "errors": dict(sorted(errors.items())), "examples": examples, "args": vars(args),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, default=str)
    args.output_json.write_text(rendered, encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
