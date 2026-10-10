"""Build the V4-A full-FOV-context, forward-HV-target dataset.

V4-A deliberately keeps the V3 model contract intact: it changes data
selection only.  Every non-target actor in the ego-local FOV is conditioning
context, while the two supervised high-value targets must lie in front of the
ego.  RIFT labels are used only to select offline targets and never appear in
the model condition.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import build_high_value_scene_dataset as base
import build_high_value_scene_dataset_v2 as v2
import build_high_value_scene_dataset_v3 as v3
from high_value_init.rift_dataset import state_generation_schema_is_clean, write_jsonl


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["inpainting"], default="inpainting")
    parser.add_argument("--min_rift_positive_agents", type=int, default=2)
    parser.add_argument("--num_generate_agents", type=int, default=2)
    parser.add_argument("--actor_radius", type=float, default=80.0)
    parser.add_argument("--fov", type=float, default=64.0)
    parser.add_argument("--positive_target_policy", choices=["route_s_diversity", "top_score"], default="route_s_diversity")
    parser.add_argument("--forward_min_x", type=float, default=0.0,
                        help="Target x in ego-local coordinates must be at least this value.")
    parser.add_argument("--target_fov_margin", type=float, default=2.0)
    parser.add_argument("--max_target_lane_distance", type=float, default=5.0)
    parser.add_argument("--default_lane_half_width", type=float, default=1.75)
    parser.add_argument("--quality_filter_v2", action="store_true", default=False)
    parser.add_argument("--min_target_speed", type=float, default=0.0)
    parser.add_argument("--max_target_speed", type=float, default=25.0)
    parser.add_argument("--min_target_ego_distance", type=float, default=3.0)
    parser.add_argument("--max_target_ego_distance", type=float, default=60.0)
    parser.add_argument("--min_target_pair_distance", type=float, default=2.0)
    parser.add_argument("--max_observed_agents", type=int, default=-1)
    parser.add_argument("--dedup_grid_xy", type=float, default=1.0)
    parser.add_argument("--base_training_weight", type=float, default=1.0)
    parser.add_argument("--sparse_context_weight_bonus", type=float, default=0.0)
    parser.add_argument("--low_observed_weight_bonus", type=float, default=0.0)
    parser.add_argument("--high_speed_weight_bonus", type=float, default=0.15)
    parser.add_argument("--weight_key", default="training_weight")
    # Kept only because V2 quality helpers validate these common fields.  V4-A
    # rejects any sparse context setting below.
    parser.add_argument("--sparse_context_copies", type=int, default=0)
    parser.add_argument("--sparse_context_min_observed", type=int, default=0)
    parser.add_argument("--sparse_context_max_observed", type=int, default=0)
    parser.add_argument("--sparse_context_replace_base", action="store_true", default=False)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="real_positive_v4a_full_fov_forward")
    parser.add_argument("--map_source", choices=["snapshot", "hd_map", "hd_map_fallback"], default="hd_map")
    parser.add_argument("--rift_root", type=Path, default=None)
    parser.add_argument("--rift_hd_map_dir", type=Path, default=None)
    parser.add_argument("--max_input_files", type=int, default=-1)
    parser.add_argument("--max_num_agents", type=int, default=61)
    parser.add_argument("--max_town_fraction", type=float, default=0.35)
    parser.add_argument("--max_topology_fraction", type=float, default=0.75)
    parser.add_argument("--min_accepted_records", type=int, default=30000,
                        help="Minimum post-filter V4-A records; a balance-only forward-pair patch is not trainable at scale.")
    parser.add_argument("--min_records_per_town_per_split", type=int, default=1)
    parser.add_argument("--min_records_per_topology_per_split", type=int, default=1)
    parser.add_argument("--diagnostics_only", action="store_true", default=False,
                        help="Scan and report V4-A candidate coverage without creating a dataset directory.")
    parser.add_argument("--diagnostics_json", type=Path, default=None,
                        help="Optional explicit path for a coverage report, including failed builds.")
    return parser


def parse_args():
    """Parse the unchanged V4-A command-line contract.

    ``build_high_value_scene_dataset_v4b.py`` reuses this parser and adds its
    own split-only options.  Keeping this wrapper preserves every existing
    V4-A invocation and default verbatim.
    """
    return build_parser().parse_args()


def validate_args(args) -> None:
    if args.num_generate_agents != 2:
        raise ValueError("V4-A preserves the validated fixed-two-target task")
    # The frozen Scenario Dreamer normalization and the V3 checkpoint both
    # operate on a 64 m square.  Accepting another value here would create a
    # dataset whose coordinate scale cannot be consumed faithfully by V4-A.
    if abs(float(args.fov) - 64.0) > 1e-6:
        raise ValueError("V4-A requires --fov 64 to match the frozen V3 model normalization")
    if args.sparse_context_copies or args.sparse_context_replace_base:
        raise ValueError("V4-A requires full-FOV context; sparse context copies are prohibited")
    if args.sparse_context_min_observed or args.sparse_context_max_observed:
        raise ValueError("V4-A requires full-FOV context; sparse context bounds must both be zero")
    if not 0.0 <= args.forward_min_x < args.fov / 2.0:
        raise ValueError("--forward_min_x must be within [0, fov/2)")
    if not 0.0 <= args.target_fov_margin < args.fov / 2.0:
        raise ValueError("--target_fov_margin must be within [0, fov/2)")
    if args.forward_min_x > args.fov / 2.0 - args.target_fov_margin:
        raise ValueError("forward target interval is empty after applying --target_fov_margin")
    if args.max_num_agents != 61:
        raise ValueError("V4-A requires --max_num_agents 61, the frozen Scenario Dreamer graph capacity")
    if not 0.0 < args.max_town_fraction <= 1.0:
        raise ValueError("--max_town_fraction must be in (0, 1]")
    if not 0.0 < args.max_topology_fraction <= 1.0:
        raise ValueError("--max_topology_fraction must be in (0, 1]")
    if args.min_accepted_records < 1:
        raise ValueError("--min_accepted_records must be positive")
    if args.min_records_per_town_per_split < 0 or args.min_records_per_topology_per_split < 0:
        raise ValueError("minimum split coverage counts must be non-negative")
    v2.validate_args(args)


def _forward_positive_agents(norm: Dict, args) -> List[Dict]:
    limit = float(args.fov) / 2.0 - float(args.target_fov_margin)
    return [
        row for row in norm["positive_agents"]
        if float(row["state"][0]) >= float(args.forward_min_x)
        and float(row["state"][0]) <= limit
        and abs(float(row["state"][1])) <= limit
    ]


def _is_rift_labeled_snapshot(scene: Dict) -> bool:
    """Reject arbitrary JSON files before HD-map extraction can obscure the real input error."""
    if not isinstance(scene, dict) or not isinstance(scene.get("ego"), dict):
        return False
    actors = scene.get("agents", scene.get("actors"))
    return isinstance(actors, list) and isinstance(scene.get("rift_labels"), list)


def _build_record(scene: Dict, lanes: List[Dict], args, norm=None) -> Tuple[Dict, str]:
    """Return ``(record, rejection_reason)`` without exposing labels to inputs."""
    if norm is None:
        norm = base.normalize_snapshot(scene, actor_radius=float(args.actor_radius), fov=float(args.fov))
    if len(norm["positive_agents"]) < args.min_rift_positive_agents:
        return {}, "insufficient_rift_positive_in_full_fov"
    forward_positive = _forward_positive_agents(norm, args)
    if len(forward_positive) < args.num_generate_agents:
        return {}, "insufficient_forward_rift_positive"
    targets = base.choose_targets(forward_positive, args.num_generate_agents, args.positive_target_policy)
    if len(targets) != args.num_generate_agents:
        return {}, "forward_target_selection_failed"
    target_ids = {int(row["actor_id"]) for row in targets}
    observed = [row for row in norm["agents"] if int(row["actor_id"]) not in target_ids]
    full_fov_count = len(norm["agents"])
    if len(observed) + len(targets) != full_fov_count:
        return {}, "full_fov_accounting_failed"
    total_agents = 1 + len(observed) + len(targets)
    if total_agents > args.max_num_agents:
        return {}, "agent_capacity_rejected"

    states = [row["state"] for row in targets]
    associations = [v3.associate(state, lanes, args.default_lane_half_width) for state in states]
    if any(item["distance"] is None or item["distance"] > args.max_target_lane_distance for item in associations):
        return {}, "target_lane_rejected"

    map_payload = dict(norm["map"])
    # V3 aliases keep V4-A trainable by the V3 trainer without modifying it.
    map_payload["lanes_v3"] = lanes
    map_payload["lanes_v4"] = lanes
    record = {
        "schema_version": "hv_init_inpainting_v4a_full_fov_forward",
        "scene_id": norm["scene_id"],
        "episode_key": norm["episode_key"],
        "town": norm["town"],
        "condition": {
            "ego_state": norm["ego_state"],
            "fov": float(args.fov),
            "ego_route": norm["ego_route"],
            "map": map_payload,
            "observed_agent_states": [row["state"] for row in observed],
            "observed_agent_types": [1 for _ in observed],
            "num_generate_agents": int(args.num_generate_agents),
        },
        "target": {
            "generated_agent_states": states,
            "generated_agent_types": [1 for _ in states],
            "map_associations_v3": associations,
            "map_associations_v4": associations,
        },
        # This metadata is audit-only.  It has no RIFT scores, IDs, labels, or
        # futures, and is not consumed by the graph builder.
        "v4_contract": {
            "condition_region": "full_fov_non_target",
            "target_sector": "forward",
            "forward_min_x": float(args.forward_min_x),
            "target_fov_margin": float(args.target_fov_margin),
            "full_fov_actor_count": int(full_fov_count),
            "observed_actor_count": int(len(observed)),
            "target_actor_count": int(len(targets)),
            "agent_capacity_policy": "reject_record_no_truncation",
            "max_num_agents": int(args.max_num_agents),
        },
        "topology_v3": {"bucket": v3.topology_bucket(lanes), "lane_count": len(lanes)},
        "topology_v4": {"bucket": v3.topology_bucket(lanes), "lane_count": len(lanes)},
        "context_sampling": {"policy": "full_fov_non_target", "sparse_context_copy_index": None},
        "source": args.source,
    }
    if not state_generation_schema_is_clean(record):
        raise ValueError(f"forbidden selector/delta field leaked into {record['scene_id']}")
    return record, ""


def _stratified_episode_split(records: List[Dict], args) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split complete episodes within towns; never split an episode across sets.

    A town with too few independent episodes cannot populate every split
    without leakage.  Keep its episodes in the available splits (one episode
    stays in train) and let the explicit per-split coverage contract decide
    whether that is acceptable.  In particular, a caller that sets minimum
    town coverage to zero has explicitly allowed rare towns to be absent from
    validation and test.
    """
    by_town_episode = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_town_episode[str(record.get("town") or "unknown")][str(record["episode_key"])].append(record)

    train, val, test = [], [], []
    import random
    for town, by_episode in sorted(by_town_episode.items()):
        keys = sorted(by_episode)
        rng = random.Random(f"v4a:{args.seed}:{town}")
        rng.shuffle(keys)
        n_test = max(1, round(len(keys) * args.test_ratio)) if args.test_ratio > 0 else 0
        n_val = max(1, round(len(keys) * args.val_ratio)) if args.val_ratio > 0 else 0
        while n_test + n_val >= len(keys):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                break
        test_keys = set(keys[:n_test])
        val_keys = set(keys[n_test:n_test + n_val])
        for key, rows in by_episode.items():
            destination = test if key in test_keys else val if key in val_keys else train
            destination.extend(rows)

    index = 0
    for split, rows in (("train", train), ("val", val), ("test", test)):
        for record in rows:
            record["idx"] = int(index)
            record["split"] = split
            index += 1
    return train, val, test


def _counts(records: Iterable[Dict], field: str) -> Counter:
    if field == "town":
        return Counter(str(record.get("town") or "unknown") for record in records)
    return Counter(str((record.get("topology_v4") or {}).get("bucket") or "unknown") for record in records)


def _strict_distribution_check(train: List[Dict], val: List[Dict], test: List[Dict], args) -> Dict:
    all_records = train + val + test
    town_counts = _counts(all_records, "town")
    topology_counts = _counts(all_records, "topology")
    total = len(all_records)
    failures = []
    if total < int(args.min_accepted_records):
        failures.append(f"accepted_record_count_below_minimum:{total}<{int(args.min_accepted_records)}")
    for label, counts, maximum in (
            ("town", town_counts, args.max_town_fraction),
            ("topology", topology_counts, args.max_topology_fraction)):
        for key, count in counts.items():
            if total and count / total > maximum + 1e-12:
                failures.append(f"{label}_fraction_exceeded:{key}:{count / total:.6f}>{maximum:.6f}")
    split_town_counts, split_topology_counts = {}, {}
    for split_name, rows in (("train", train), ("val", val), ("test", test)):
        towns = _counts(rows, "town")
        topologies = _counts(rows, "topology")
        split_town_counts[split_name] = dict(sorted(towns.items()))
        split_topology_counts[split_name] = dict(sorted(topologies.items()))
        if args.min_records_per_town_per_split:
            for town in town_counts:
                if towns[town] < args.min_records_per_town_per_split:
                    failures.append(f"town_split_coverage:{split_name}:{town}:{towns[town]}")
        if args.min_records_per_topology_per_split:
            for topology in topology_counts:
                if topologies[topology] < args.min_records_per_topology_per_split:
                    failures.append(f"topology_split_coverage:{split_name}:{topology}:{topologies[topology]}")
    return {
        "town_counts": dict(sorted(town_counts.items())),
        "topology_bucket_counts": dict(sorted(topology_counts.items())),
        "split_town_counts": split_town_counts,
        "split_topology_bucket_counts": split_topology_counts,
        "failures": failures,
    }


def _count_bin(value: int, last_explicit: int = 4) -> str:
    value = int(value)
    return str(value) if value <= last_explicit else f"{last_explicit + 1}_plus"


def _update_candidate_coverage(coverage: Dict, norm: Dict, lanes: List[Dict], args) -> None:
    """Record evidence for why a forward-target dataset is or is not possible."""
    agents = norm["agents"]
    positives = norm["positive_agents"]
    forward = _forward_positive_agents(norm, args)
    coverage["full_fov_actor_count_histogram"][_count_bin(len(agents), 10)] += 1
    coverage["full_fov_rift_positive_count_histogram"][_count_bin(len(positives), 6)] += 1
    coverage["forward_rift_positive_count_histogram"][_count_bin(len(forward), 6)] += 1
    coverage["local_lane_count_histogram"][_count_bin(len(lanes), 10)] += 1
    limit = float(args.fov) / 2.0 - float(args.target_fov_margin)
    for row in positives:
        x, y = float(row["state"][0]), float(row["state"][1])
        if x >= args.forward_min_x and x <= limit and abs(y) <= limit:
            region = "forward_target_sector"
        elif x < 0.0:
            region = "rear_half"
        else:
            region = "forward_or_side_excluded_by_margin_or_min_x"
        coverage["rift_positive_region_counts"][region] += 1


def _coverage_json(coverage: Dict) -> Dict:
    return {key: dict(sorted(value.items())) for key, value in coverage.items()}


def _build_summary(
        args,
        processed: int,
        raw_records: List[Dict],
        records: List[Dict],
        rejection_counts: Counter,
        quality_rejections: Counter,
        quality_rejected: int,
        duplicates: int,
        map_source_counts: Counter,
        map_errors: Counter,
        coverage: Dict,
        record_build_error_examples=None,
        valid_labeled_snapshot_count: int = 0,
        invalid_input_examples=None,
        train=None,
        val=None,
        test=None,
        distribution=None) -> Dict:
    train, val, test = train or [], val or [], test or []
    return {
        "schema_version": "hv_scene_dataset_v4a_full_fov_forward",
        "processed_files": processed,
        "valid_rift_labeled_snapshot_count": int(valid_labeled_snapshot_count),
        "invalid_input_examples": list(invalid_input_examples or []),
        "num_raw_records_before_v2_quality_and_dedup": len(raw_records),
        "num_records_after_v2_quality_and_dedup": len(records),
        "num_quality_rejected": quality_rejected,
        "num_duplicates": duplicates,
        "num_train_records": len(train),
        "num_val_records": len(val),
        "num_test_records": len(test),
        "selection_rejections": dict(sorted(rejection_counts.items())),
        "quality_reject_reasons": dict(sorted(quality_rejections.items())),
        "map_source_counts": dict(sorted(map_source_counts.items())),
        "hd_map_errors": dict(sorted(map_errors.items())),
        "record_build_error_examples": list(record_build_error_examples or []),
        "candidate_coverage_before_target_selection": _coverage_json(coverage),
        # Available even in --diagnostics_only mode.  This is deliberately
        # pre-split: capacity planners need accepted Town/topology evidence
        # before there are enough episodes to construct a valid split.
        "accepted_record_distribution_pre_split": {
            "town_counts": dict(sorted(Counter(str(record.get("town") or "unknown") for record in records).items())),
            "topology_bucket_counts": dict(sorted(Counter(str((record.get("topology_v4") or {}).get("bucket") or "unknown") for record in records).items())),
        },
        "distribution": distribution or {},
        "args": v2.json_safe(vars(args)),
    }


def _write_diagnostics(path: Path, summary: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    validate_args(args)
    extractor = None
    if args.map_source != "snapshot":
        extractor = v3.HDMapLaneExtractor(args.rift_root, args.rift_hd_map_dir)

    raw_records = []
    rejection_counts = Counter()
    map_source_counts = Counter()
    map_errors = Counter()
    record_build_error_examples = []
    invalid_input_examples = []
    valid_labeled_snapshot_count = 0
    coverage = {
        "full_fov_actor_count_histogram": Counter(),
        "full_fov_rift_positive_count_histogram": Counter(),
        "forward_rift_positive_count_histogram": Counter(),
        "local_lane_count_histogram": Counter(),
        "rift_positive_region_counts": Counter(),
    }
    processed = 0
    for path in sorted(args.input_dir.rglob("*.json")):
        if args.max_input_files > 0 and processed >= args.max_input_files:
            break
        processed += 1
        try:
            scene = base.load_snapshot(path)
        except Exception as exc:
            rejection_counts[f"snapshot_load:{type(exc).__name__}"] += 1
            continue
        if not _is_rift_labeled_snapshot(scene):
            rejection_counts["not_rift_labeled_snapshot"] += 1
            if len(invalid_input_examples) < 20:
                invalid_input_examples.append({
                    "path": str(path),
                    "schema_version": scene.get("schema_version") if isinstance(scene, dict) else None,
                    "top_level_keys": sorted(scene.keys())[:20] if isinstance(scene, dict) else [],
                })
            continue
        valid_labeled_snapshot_count += 1
        snapshot_lanes = scene.get("map_lanes", []) or []
        use_hd_map = extractor is not None and (args.map_source == "hd_map" or not snapshot_lanes)
        if use_hd_map:
            try:
                scene["map_lanes"] = extractor.lanes(scene, args.fov)
                map_source_counts["hd_map"] += 1
            except Exception as exc:
                map_errors[type(exc).__name__] += 1
                rejection_counts["hd_map_error"] += 1
                continue
        else:
            map_source_counts["snapshot"] += 1
        lanes = v3.local_lanes(scene, args.fov)
        try:
            norm = base.normalize_snapshot(scene, actor_radius=float(args.actor_radius), fov=float(args.fov))
            _update_candidate_coverage(coverage, norm, lanes, args)
            record, reason = _build_record(scene, lanes, args, norm=norm)
        except (KeyError, TypeError, ValueError) as exc:
            rejection_counts[f"record_build:{type(exc).__name__}"] += 1
            if len(record_build_error_examples) < 20:
                record_build_error_examples.append({
                    "path": str(path), "type": type(exc).__name__, "message": str(exc),
                })
            continue
        if not record:
            rejection_counts[reason or "unknown"] += 1
            continue
        record["source_file_sha1"] = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
        raw_records.append(record)

    records, reject_counter, quality_rejected, duplicates = v2.filter_and_weight(raw_records, args)
    diagnostic_summary = _build_summary(
        args=args,
        processed=processed,
        raw_records=raw_records,
        records=records,
        rejection_counts=rejection_counts,
        quality_rejections=reject_counter,
        quality_rejected=quality_rejected,
        duplicates=duplicates,
        map_source_counts=map_source_counts,
        map_errors=map_errors,
        coverage=coverage,
        record_build_error_examples=record_build_error_examples,
        valid_labeled_snapshot_count=valid_labeled_snapshot_count,
        invalid_input_examples=invalid_input_examples,
    )
    if args.diagnostics_json:
        _write_diagnostics(args.diagnostics_json, diagnostic_summary)
    if args.diagnostics_only:
        print(json.dumps(diagnostic_summary, indent=2))
        if valid_labeled_snapshot_count == 0:
            raise ValueError(
                "input_dir contains no RIFT-labeled snapshot JSON. Verify --input_dir and export RAW_SNAPSHOTS; "
                "do not use the repository working directory."
            )
        return
    if not records:
        print(json.dumps(diagnostic_summary, indent=2))
        raise ValueError(
            "V4-A produced zero records. Inspect candidate_coverage_before_target_selection, "
            "selection_rejections, and quality_reject_reasons above (or --diagnostics_json) before collecting more data."
        )
    train, val, test = _stratified_episode_split(records, args)
    distribution = _strict_distribution_check(train, val, test, args)
    summary = _build_summary(
        args=args,
        processed=processed,
        raw_records=raw_records,
        records=records,
        rejection_counts=rejection_counts,
        quality_rejections=reject_counter,
        quality_rejected=quality_rejected,
        duplicates=duplicates,
        map_source_counts=map_source_counts,
        map_errors=map_errors,
        coverage=coverage,
        record_build_error_examples=record_build_error_examples,
        valid_labeled_snapshot_count=valid_labeled_snapshot_count,
        invalid_input_examples=invalid_input_examples,
        train=train,
        val=val,
        test=test,
        distribution=distribution,
    )
    if distribution["failures"]:
        # Do not write a dataset which fails the declared diversity contract.
        if args.diagnostics_json:
            _write_diagnostics(args.diagnostics_json, summary)
        raise ValueError(json.dumps(summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(str(args.output_dir / f"{name}.jsonl"), rows)
    for name in ("summary.json", "quality_report.json"):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    if args.diagnostics_json:
        _write_diagnostics(args.diagnostics_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
