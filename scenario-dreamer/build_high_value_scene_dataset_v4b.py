"""Build the V4B two-HV forward/side-forward dataset without changing V4A.

V4B keeps the frozen Scenario Dreamer graph contract used by V4A: full-FOV
ordinary-vehicle context and exactly two RIFT-positive targets in the forward
half of the ego-local 64 m square.  The two changes are deliberately limited
to data quality:

* target pairs are selected by their RIFT interaction strength, subject to a
  ground-truth oriented-box clearance check; and
* whole episodes are assigned with a deterministic town/topology/target-region
  balanced split rather than a town-only random episode split.

The JSONL record schema remains V4A-compatible because the model input
contract is unchanged.  V4B identity, split diagnostics and sampling weights
are stored as metadata only.  RIFT labels are never copied into model inputs;
the explicit V4C opt-in stores conflict geometry under ``target`` only.
"""

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import build_high_value_scene_dataset as base
import build_high_value_scene_dataset_v2 as v2
import build_high_value_scene_dataset_v3 as v3
import build_high_value_scene_dataset_v4 as v4
from high_value_init.rift_dataset import state_generation_schema_is_clean, write_jsonl


def parse_args():
    parser = v4.build_parser()
    parser.prog = "build_high_value_scene_dataset_v4b.py"
    parser.description = __doc__
    parser.set_defaults(source="real_positive_v4b_full_fov_forward")
    parser.add_argument(
        "--min_target_obb_axis_clearance", type=float, default=0.5,
        help="Minimum signed SAT-axis clearance between the two ground-truth targets.",
    )
    parser.add_argument(
        "--v4b_target_pair_policy",
        choices=("high_value_pair", "route_diversity"),
        default="high_value_pair",
        help=("high_value_pair ranks the weaker target score first, then pair score; "
              "route_diversity retains the historical V4A ordering for an ablation."),
    )
    parser.add_argument(
        "--min_target_interaction_score", type=float, default=0.0,
        help="Offline-only minimum RIFT label score required for each selected target.",
    )
    parser.add_argument(
        "--min_pair_interaction_score", type=float, default=0.0,
        help="Offline-only minimum sum of the two selected RIFT label scores.",
    )
    parser.add_argument(
        "--interaction_score_weight_power", type=float, default=0.0,
        help=("Train-only sampling weight exponent on the weaker selected target score. "
              "Zero disables score weighting; labels remain absent from JSONL."),
    )
    parser.add_argument(
        "--max_interaction_score_weight", type=float, default=3.0,
        help="Cap for normalized train-only interaction-score sampling weights.",
    )
    parser.add_argument(
        "--balance_weight_power", type=float, default=0.5,
        help="Inverse-frequency exponent for train-only town/topology/region weights.",
    )
    parser.add_argument(
        "--max_balance_weight", type=float, default=3.0,
        help="Upper cap applied after train-only inverse-frequency weighting.",
    )
    parser.add_argument(
        "--max_split_town_fraction", type=float, default=0.45,
        help="Diagnostic/optional hard upper bound for one town within each split.",
    )
    parser.add_argument(
        "--max_split_topology_fraction", type=float, default=0.80,
        help="Diagnostic/optional hard upper bound for one topology within each split.",
    )
    parser.add_argument(
        "--require_split_balance", action="store_true", default=False,
        help="Refuse to write JSONL when the V4B per-split balance contract fails.",
    )
    parser.add_argument(
        "--split_search_restarts", type=int, default=48,
        help=("Deterministic whole-episode split-search attempts. More attempts may find a "
              "feasible town/topology-balanced split without episode leakage."),
    )
    parser.add_argument(
        "--enforce_global_town_fraction", action="store_true", default=False,
        help=("Deterministically downsample only overrepresented towns after all quality and "
              "route-supervision filtering, preserving town/topology/target-region proportions. "
              "This is opt-in because it changes the training population."),
    )
    parser.add_argument(
        "--record_balanced_episode_split", action="store_true", default=False,
        help=("Add a record-count-aware whole-episode split candidate. It preserves episode "
              "isolation while avoiding large episodes randomly dominating validation or test."),
    )
    parser.add_argument(
        "--include_route_interaction_supervision", action="store_true", default=False,
        help=("Write RIFT conflict-point geometry under target supervision only. "
              "It is never copied into condition/model input and is intended for V4C training."),
    )
    parser.add_argument(
        "--route_interaction_sidecar", type=Path, default=None,
        help=("Optional JSONL from enrich_rift_interaction_routes.py. It supplies the exact "
              "RIFT trace_route polyline for target-only route-progress supervision."),
    )
    parser.add_argument(
        "--route_interaction_manifest", type=Path, default=None,
        help=("Optional build-only JSONL of deduplicated source hashes and selected target actor IDs. "
              "Use it to reconstruct RIFT paths only for V4C-selected targets."),
    )
    parser.add_argument(
        "--route_interaction_manifest_only", action="store_true", default=False,
        help=("Stop after writing the deduplicated route-target manifest. This avoids split work "
              "during the first V4C sidecar pass and writes no dataset JSONL."),
    )
    parser.add_argument(
        "--require_route_progress_supervision", action="store_true", default=False,
        help=("Reject selected pairs unless both targets have a RIFT trace_route polyline. "
              "Used by V4C direct route-distance/synchronization training."),
    )
    return parser.parse_args()


def validate_args(args):
    v4.validate_args(args)
    if args.positive_target_policy != "route_s_diversity":
        raise ValueError(
            "V4B does not use --positive_target_policy; use "
            "--v4b_target_pair_policy to make pair selection explicit"
        )
    if args.min_target_obb_axis_clearance < 0.0:
        raise ValueError("--min_target_obb_axis_clearance must be non-negative")
    if not 0.0 <= args.min_target_interaction_score <= 1.0:
        raise ValueError("--min_target_interaction_score must be within [0, 1]")
    if not 0.0 <= args.min_pair_interaction_score <= 2.0:
        raise ValueError("--min_pair_interaction_score must be within [0, 2]")
    if not 0.0 <= args.interaction_score_weight_power <= 2.0:
        raise ValueError("--interaction_score_weight_power must be within [0, 2]")
    if args.max_interaction_score_weight < 1.0:
        raise ValueError("--max_interaction_score_weight must be at least one")
    if not 0.0 <= args.balance_weight_power <= 1.0:
        raise ValueError("--balance_weight_power must be within [0, 1]")
    if args.max_balance_weight < 1.0:
        raise ValueError("--max_balance_weight must be at least one")
    if not 0.0 < args.max_split_town_fraction <= 1.0:
        raise ValueError("--max_split_town_fraction must be in (0, 1]")
    if not 0.0 < args.max_split_topology_fraction <= 1.0:
        raise ValueError("--max_split_topology_fraction must be in (0, 1]")
    if args.split_search_restarts < 1:
        raise ValueError("--split_search_restarts must be positive")
    if args.require_route_progress_supervision and not args.include_route_interaction_supervision:
        raise ValueError("--require_route_progress_supervision requires --include_route_interaction_supervision")
    if args.require_route_progress_supervision and args.route_interaction_sidecar is None:
        raise ValueError("--require_route_progress_supervision requires --route_interaction_sidecar")
    if args.route_interaction_manifest_only and args.route_interaction_manifest is None:
        raise ValueError("--route_interaction_manifest_only requires --route_interaction_manifest")


def _target_region(state) -> str:
    """Coarse audit bucket; it is never supplied to the model."""
    x, y = float(state[0]), float(state[1])
    lateral = abs(y)
    if x < 8.0:
        return "side" if lateral >= 6.0 else "near_front"
    return "front_side" if lateral >= 8.0 else "front"


def _pair_region(targets) -> str:
    return "+".join(sorted(_target_region(row["state"]) for row in targets))


def _obb_signed_axis_clearance(left, right) -> float:
    """Return the largest SAT signed separation axis for two labeled boxes."""
    ax, ay, ayaw, _aspeed, alen, awid = [float(value) for value in left[:6]]
    bx, by, byaw, _bspeed, blen, bwid = [float(value) for value in right[:6]]
    au = (math.cos(ayaw), math.sin(ayaw))
    av = (-au[1], au[0])
    bu = (math.cos(byaw), math.sin(byaw))
    bv = (-bu[1], bu[0])
    delta = (bx - ax, by - ay)

    def dot(one, two):
        return one[0] * two[0] + one[1] * two[1]

    gaps = []
    for axis in (au, av, bu, bv):
        ar = alen / 2.0 * abs(dot(axis, au)) + awid / 2.0 * abs(dot(axis, av))
        br = blen / 2.0 * abs(dot(axis, bu)) + bwid / 2.0 * abs(dot(axis, bv))
        gaps.append(abs(dot(delta, axis)) - ar - br)
    return max(gaps)


def _select_feasible_targets(forward_positive, lanes, args):
    """Choose an offline high-value pair; labels never reach model inputs."""
    best, best_key, rejected_clearance, rejected_score = None, None, 0, 0
    for left, right in itertools.combinations(forward_positive, 2):
        left_state, right_state = left["state"], right["state"]
        left_score = float(left["label"].get("score", 0.0))
        right_score = float(right["label"].get("score", 0.0))
        pair_score = left_score + right_score
        if (min(left_score, right_score) < float(args.min_target_interaction_score)
                or pair_score < float(args.min_pair_interaction_score)):
            rejected_score += 1
            continue
        clearance = _obb_signed_axis_clearance(left_state, right_state)
        if clearance < float(args.min_target_obb_axis_clearance):
            rejected_clearance += 1
            continue
        associations = [
            v3.associate(left_state, lanes, args.default_lane_half_width),
            v3.associate(right_state, lanes, args.default_lane_half_width),
        ]
        if any(item.get("distance") is None or item["distance"] > args.max_target_lane_distance
               for item in associations):
            continue
        left_distance = float(left["label"].get("ego_distance", left.get("ego_distance", 0.0)) or 0.0)
        right_distance = float(right["label"].get("ego_distance", right.get("ego_distance", 0.0)) or 0.0)
        route_diversity = abs(left_distance - right_distance)
        # Current-frame RIFT strength is the actual V4B supervision choice:
        # maximize the weaker target first so the two-target task does not
        # collapse into one strong actor plus one weak positive.  Geometry is
        # already a hard gate and route diversity is only a tie-breaker.
        if args.v4b_target_pair_policy == "high_value_pair":
            key = (min(left_score, right_score), pair_score, route_diversity, clearance)
        else:
            key = (route_diversity, pair_score, min(left_score, right_score), clearance)
        if best_key is None or key > best_key:
            best, best_key = ([left, right], key)
    return best or [], rejected_clearance, rejected_score


def _route_remaining_distance_local(point, polyline):
    """Remaining polyline arc length from the closest projection of ``point``."""
    if len(polyline) < 2:
        return None
    total = 0.0
    segment_lengths = []
    for start, end in zip(polyline[:-1], polyline[1:]):
        length = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        segment_lengths.append(length)
        total += length
    best = None
    traversed = 0.0
    px, py = float(point[0]), float(point[1])
    for index, (start, end) in enumerate(zip(polyline[:-1], polyline[1:])):
        dx, dy = float(end[0]) - float(start[0]), float(end[1]) - float(start[1])
        norm_sq = dx * dx + dy * dy
        if norm_sq <= 1e-9:
            traversed += segment_lengths[index]
            continue
        ratio = min(1.0, max(0.0, ((px - float(start[0])) * dx + (py - float(start[1])) * dy) / norm_sq))
        qx, qy = float(start[0]) + ratio * dx, float(start[1]) + ratio * dy
        distance_sq = (px - qx) ** 2 + (py - qy) ** 2
        progress = traversed + ratio * segment_lengths[index]
        if best is None or distance_sq < best[0]:
            best = (distance_sq, progress)
        traversed += segment_lengths[index]
    return None if best is None else float(total - best[1])


def _state_preview_point_local(state, preview_distance=2.0):
    """Match RIFT's candidate start point: two metres along vehicle heading."""
    try:
        x, y, heading = float(state[0]), float(state[1]), float(state[2])
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, heading)):
        return None
    return [
        x + float(preview_distance) * math.cos(heading),
        y + float(preview_distance) * math.sin(heading),
    ]


def _route_interaction_supervision(scene: Dict, target: Dict, sidecar_target: Dict = None) -> Dict:
    """Serialize a target-only RIFT conflict geometry label.

    The values are created from the RIFT label that selected this target.  They
    are deliberately separate from ``condition``: online generation never has
    an interaction waypoint and must not depend on one.

    ``target_route_distance_to_conflict_m`` is RIFT's planner distance, not an
    Euclidean approximation.  The trainable loss uses only the differentiable
    local anchor geometry below; it must not pretend that a heading projection
    is an exact CARLA route distance.
    """
    label = target.get("label") or {}
    waypoint = label.get("interaction_waypoint")
    if not isinstance(waypoint, dict) or "x" not in waypoint or "y" not in waypoint:
        return {}
    try:
        anchor_x, anchor_y = base.point_local(scene["ego"], waypoint)
        ego_route_distance = float(label["ego_distance"])
        target_route_distance = float(label["cbv_distance"])
        distance_difference = float(label["distance_diff"])
        threshold = float(label["min_interaction_dis_diff"])
        score = float(label["score"])
        state = target["state"]
        heading = float(state[2])
        dx, dy = float(anchor_x) - float(state[0]), float(anchor_y) - float(state[1])
    except (KeyError, TypeError, ValueError):
        return {}
    values = (
        anchor_x, anchor_y, ego_route_distance, target_route_distance,
        distance_difference, threshold, score, dx, dy, heading,
    )
    if not bool(label.get("is_interactive", False)) or not all(math.isfinite(value) for value in values):
        return {}
    if threshold <= 0.0 or distance_difference < 0.0 or target_route_distance < 0.0 or ego_route_distance < 0.0:
        return {}
    # Local target-frame coordinates of the RIFT conflict waypoint.  These
    # remain a faithful geometric relation even at connectors where a straight
    # line must not be substituted for the planner's route distance.
    cos_heading, sin_heading = math.cos(heading), math.sin(heading)
    along = dx * cos_heading + dy * sin_heading
    lateral = -dx * sin_heading + dy * cos_heading
    route_source = "unavailable_in_snapshot"
    route_points = label.get("target_route_to_interaction") or []
    planner_distance_error = None
    if sidecar_target and sidecar_target.get("target_route_to_interaction"):
        route_points = sidecar_target.get("target_route_to_interaction") or []
        route_source = "rift_global_route_planner_sidecar"
        try:
            planner_distance_error = float(sidecar_target["planner_distance_abs_error_to_rift_label_m"])
        except (KeyError, TypeError, ValueError):
            return {}
        if not math.isfinite(planner_distance_error) or planner_distance_error < 0.0:
            return {}
    elif route_points:
        route_source = "rift_global_route_planner_snapshot"
    target_route_polyline_local_xy = []
    for point in route_points:
        try:
            local = base.point_local(scene["ego"], point)
            if all(math.isfinite(float(value)) for value in local[:2]):
                target_route_polyline_local_xy.append([float(local[0]), float(local[1])])
        except (KeyError, TypeError, ValueError):
            continue
    # Base the calibration on the same two-metre preview point used by
    # BaseCBVRecog.get_preview_waypoint(), not on the vehicle centre.  The
    # direct V4C loss later applies exactly this relation to generated x/y/yaw.
    preview_point = _state_preview_point_local(state)
    route_remaining_at_target = (
        _route_remaining_distance_local(preview_point, target_route_polyline_local_xy)
        if preview_point is not None else None
    )
    route_offset = None if route_remaining_at_target is None else target_route_distance - route_remaining_at_target
    return {
        "schema_version": "rift_route_interaction_target_v1",
        "conflict_point_local_xy": [float(anchor_x), float(anchor_y)],
        "ego_route_distance_to_conflict_m": ego_route_distance,
        "target_route_distance_to_conflict_m": target_route_distance,
        "route_distance_difference_m": distance_difference,
        "route_distance_difference_threshold_m": threshold,
        "interaction_strength": score,
        "target_to_conflict_along_heading_m": float(along),
        "target_to_conflict_lateral_heading_m": float(lateral),
        # Historical snapshots do not contain this additive field.  V4C's
        # anchor loss remains usable for them; a future exact route-progress
        # loss must require at least two RIFT planner points rather than
        # substituting a straight line.
        "target_route_polyline_local_xy": target_route_polyline_local_xy,
        "target_route_polyline_source": route_source if len(target_route_polyline_local_xy) >= 2 else "unavailable_in_snapshot",
        "rift_preview_distance_m": 2.0,
        "planner_distance_abs_error_to_rift_label_m": planner_distance_error,
        # Calibrates the arc-length convention of trace_route's first waypoint
        # to RIFT's original actor-location distance. It is a target label,
        # never an online adjustment to a generated state.
        "target_route_distance_offset_m": route_offset,
    }


def _build_record(scene: Dict, lanes: List[Dict], args, norm=None, sidecar_targets=None) -> Tuple[Dict, str]:
    if norm is None:
        norm = base.normalize_snapshot(scene, actor_radius=float(args.actor_radius), fov=float(args.fov))
    if len(norm["positive_agents"]) < args.min_rift_positive_agents:
        return {}, "insufficient_rift_positive_in_full_fov"
    forward_positive = v4._forward_positive_agents(norm, args)
    if len(forward_positive) < args.num_generate_agents:
        return {}, "insufficient_forward_rift_positive"
    targets, clearance_rejections, score_rejections = _select_feasible_targets(forward_positive, lanes, args)
    if len(targets) != args.num_generate_agents:
        if score_rejections:
            return {}, "target_pair_interaction_score_rejected"
        return {}, ("target_pair_obb_clearance_rejected" if clearance_rejections else "forward_target_selection_failed")

    target_ids = {int(row["actor_id"]) for row in targets}
    observed = [row for row in norm["agents"] if int(row["actor_id"]) not in target_ids]
    full_fov_count = len(norm["agents"])
    if len(observed) + len(targets) != full_fov_count:
        return {}, "full_fov_accounting_failed"
    if 1 + len(observed) + len(targets) > args.max_num_agents:
        return {}, "agent_capacity_rejected"

    states = [row["state"] for row in targets]
    associations = [v3.associate(state, lanes, args.default_lane_half_width) for state in states]
    if any(item.get("distance") is None or item["distance"] > args.max_target_lane_distance for item in associations):
        return {}, "target_lane_rejected"

    map_payload = dict(norm["map"])
    map_payload["lanes_v3"] = lanes
    map_payload["lanes_v4"] = lanes
    pair_clearance = _obb_signed_axis_clearance(states[0], states[1])
    interaction_supervision = []
    if args.include_route_interaction_supervision:
        sidecar_targets = sidecar_targets or {}
        interaction_supervision = [
            _route_interaction_supervision(scene, row, sidecar_targets.get(str(int(row["actor_id"]))))
            for row in targets
        ]
        if len(interaction_supervision) != len(targets) or any(not item for item in interaction_supervision):
            return {}, "route_interaction_supervision_missing"
        if args.require_route_progress_supervision and any(
                len(item.get("target_route_polyline_local_xy") or []) < 2
                or item.get("target_route_distance_offset_m") is None
                or item.get("rift_preview_distance_m") is None
                or item.get("planner_distance_abs_error_to_rift_label_m") is None
                for item in interaction_supervision):
            return {}, "route_progress_supervision_missing"
    record = {
        # Input semantics are intentionally compatible with the V4A audit/evaluator.
        "schema_version": "hv_init_inpainting_v4a_full_fov_forward",
        "dataset_version": (
            "v4c_route_interaction_supervised"
            if args.include_route_interaction_supervision else "v4b_group_stratified"
        ),
        "scene_id": norm["scene_id"],
        "episode_key": norm["episode_key"],
        "town": norm["town"],
        "condition": {
            "ego_state": norm["ego_state"], "fov": float(args.fov),
            "ego_route": norm["ego_route"], "map": map_payload,
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
        "v4_contract": {
            "condition_region": "full_fov_non_target", "target_sector": "forward",
            "forward_min_x": float(args.forward_min_x),
            "target_fov_margin": float(args.target_fov_margin),
            "full_fov_actor_count": int(full_fov_count),
            "observed_actor_count": int(len(observed)), "target_actor_count": int(len(targets)),
            "agent_capacity_policy": "reject_record_no_truncation", "max_num_agents": int(args.max_num_agents),
        },
        "v4b_contract": {
            "target_pair_selection": str(args.v4b_target_pair_policy),
            "target_pair_obb_axis_clearance": float(pair_clearance),
            "min_target_obb_axis_clearance": float(args.min_target_obb_axis_clearance),
            "target_region_pair": _pair_region(targets),
        },
        "topology_v3": {"bucket": v3.topology_bucket(lanes), "lane_count": len(lanes)},
        "topology_v4": {"bucket": v3.topology_bucket(lanes), "lane_count": len(lanes)},
        "context_sampling": {"policy": "full_fov_non_target", "sparse_context_copy_index": None},
        "source": args.source,
        # Removed before filtering/serialization. This lets the build report
        # prove what was selected without leaking RIFT label values into a
        # graph record or model condition.
        "_v4b_build_only_target_interaction": {
            "min": float(min(float(row["label"].get("score", 0.0)) for row in targets)),
            "sum": float(sum(float(row["label"].get("score", 0.0)) for row in targets)),
        },
        "_v4b_build_only_target_source_actor_id_values": [int(row["actor_id"]) for row in targets],
    }
    if interaction_supervision:
        # This lives only on the supervised target.  Graph construction reads
        # generated_agent_states and never consumes these RIFT labels.
        record["target"]["route_interaction_supervision"] = interaction_supervision
        record["v4c_contract"] = {
            "supervision": "rift_conflict_point_target_only",
            "online_condition_contains_rift_interaction": False,
            "route_distance_source": "rift_global_route_planner",
        }
    if not state_generation_schema_is_clean(record):
        raise ValueError("forbidden selector/delta field leaked into %s" % record["scene_id"])
    return record, ""


def _topology(record):
    return str((record.get("topology_v4") or {}).get("bucket") or "unknown")


def _balance_key(record):
    return "%s|%s|%s" % (
        str(record.get("town") or "unknown"),
        _topology(record),
        str((record.get("v4b_contract") or {}).get("target_region_pair") or "unknown"),
    )


def _enforce_global_town_fraction(records, args):
    """Optionally cap only globally overrepresented towns without splitting episodes.

    Record removal is stratified by the existing training balance key and chosen
    by a stable hash.  It is deliberately an explicit data-population operation,
    not an undocumented relaxation of the global diversity contract.
    """
    if not args.enforce_global_town_fraction or not records:
        return records, {"enabled": False, "dropped_records": 0, "towns": {}}
    maximum = float(args.max_town_fraction)
    by_town = defaultdict(list)
    for record in records:
        by_town[str(record.get("town") or "unknown")].append(record)
    retained, diagnostics = [], {"enabled": True, "dropped_records": 0, "towns": {}}
    total = len(records)
    for town, rows in sorted(by_town.items()):
        other = total - len(rows)
        allowed = int(math.floor((maximum * other) / max(1.0 - maximum, 1e-12)))
        keep_count = min(len(rows), allowed)
        if keep_count >= len(rows):
            retained.extend(rows)
            continue
        strata = defaultdict(list)
        for row in rows:
            strata[_balance_key(row)].append(row)
        quotas, fractions = {}, []
        for key, values in strata.items():
            exact = len(values) * keep_count / len(rows)
            quotas[key] = int(math.floor(exact))
            fractions.append((exact - quotas[key], key))
        remaining = keep_count - sum(quotas.values())
        for _fraction, key in sorted(fractions, key=lambda item: (-item[0], item[1]))[:remaining]:
            quotas[key] += 1
        kept_by_stratum = {}
        for key, values in strata.items():
            values = sorted(values, key=lambda row: hashlib.sha1((
                "%s|%s|%s|%s" % (args.seed, row.get("source_file_sha1", ""), row.get("scene_id", ""), key)
            ).encode("utf-8")).hexdigest())
            chosen = values[:quotas[key]]
            retained.extend(chosen)
            kept_by_stratum[key] = len(chosen)
        diagnostics["towns"][town] = {
            "before": len(rows), "allowed": keep_count, "dropped": len(rows) - keep_count,
            "kept_by_balance_key": dict(sorted(kept_by_stratum.items())),
        }
        diagnostics["dropped_records"] += len(rows) - keep_count
    return retained, diagnostics


def _group_stratified_episode_split(records, args):
    """Assign complete episodes using bounded deterministic multi-start search."""
    groups = defaultdict(list)
    for record in records:
        groups[str(record["episode_key"])].append(record)

    split_names = ("train", "val", "test")
    ratios = {"train": 1.0 - float(args.val_ratio) - float(args.test_ratio),
              "val": float(args.val_ratio), "test": float(args.test_ratio)}
    if ratios["train"] <= 0.0:
        raise ValueError("train split ratio must be positive")

    total = len(records)
    total_by_town = Counter(str(row.get("town") or "unknown") for row in records)
    total_by_topology = Counter(_topology(row) for row in records)
    total_by_region = Counter(str((row.get("v4b_contract") or {}).get("target_region_pair") or "unknown") for row in records)
    group_profiles = {}
    stratum_frequency = Counter()
    for episode, rows in groups.items():
        town = Counter(str(row.get("town") or "unknown") for row in rows).most_common(1)[0][0]
        topology = Counter(_topology(row) for row in rows).most_common(1)[0][0]
        group_profiles[episode] = {
            "rows": rows, "town": Counter(str(row.get("town") or "unknown") for row in rows),
            "topology": Counter(_topology(row) for row in rows),
            "region": Counter(str((row.get("v4b_contract") or {}).get("target_region_pair") or "unknown") for row in rows),
            "stratum": (town, topology),
        }
        stratum_frequency[(town, topology)] += 1
    if len(group_profiles) < len(split_names):
        raise ValueError("V4B needs at least three distinct episodes for train/val/test")

    def run_attempt(attempt):
        rng = random.Random("v4b-group-split:%s:%d" % (args.seed, attempt))
        order = list(group_profiles)
        rng.shuffle(order)
        order.sort(key=lambda key: (
            -len(group_profiles[key]["rows"]),
            stratum_frequency[group_profiles[key]["stratum"]],
        ))
        current_total = Counter()
        current_town = {name: Counter() for name in split_names}
        current_topology = {name: Counter() for name in split_names}
        current_region = {name: Counter() for name in split_names}
        assigned = {name: [] for name in split_names}
        assigned_episodes = {name: [] for name in split_names}

        def capacity_ok(split, profile):
            # Use the intended split size as a hard capacity. This prevents a
            # single seed episode from making a held-out Town cap impossible.
            target_total = ratios[split] * total
            for current, limit, values in (
                    (current_town, args.max_split_town_fraction, profile["town"]),
                    (current_topology, args.max_split_topology_fraction, profile["topology"])):
                for key, increment in values.items():
                    if current[split][key] + increment > limit * target_total + 1e-12:
                        return False
            return True

        def projected_error(split, profile):
            if not capacity_ok(split, profile):
                return float("inf")
            target_total = ratios[split] * total
            score = 8.0 * ((current_total[split] + len(profile["rows"]) - target_total) / max(target_total, 1)) ** 2
            for source, current, weight, values in (
                    (total_by_town, current_town, 4.0, profile["town"]),
                    (total_by_topology, current_topology, 3.0, profile["topology"]),
                    (total_by_region, current_region, 2.0, profile["region"])):
                for key, increment in values.items():
                    target = ratios[split] * source[key]
                    score += weight * ((current[split][key] + increment - target) / max(target, 1)) ** 2
            if split != "train" and stratum_frequency[profile["stratum"]] < 3:
                score += 0.02
            return score

        def assign(episode, split):
            profile = group_profiles[episode]
            assigned[split].extend(profile["rows"])
            assigned_episodes[split].append(episode)
            current_total[split] += len(profile["rows"])
            current_town[split].update(profile["town"])
            current_topology[split].update(profile["topology"])
            current_region[split].update(profile["region"])

        train_seed = next((episode for episode in order if capacity_ok("train", group_profiles[episode])), None)
        if train_seed is None:
            return None, {"attempt": attempt, "failure": "no_feasible_train_seed"}
        assign(train_seed, "train")
        seeded = {train_seed}
        for split in ("val", "test"):
            candidates = [episode for episode in reversed(order)
                          if episode not in seeded and capacity_ok(split, group_profiles[episode])]
            if not candidates:
                return None, {"attempt": attempt, "failure": "no_feasible_%s_seed" % split}
            candidates = candidates[:max(1, len(candidates) // 3)]
            episode = candidates[rng.randrange(len(candidates))]
            assign(episode, split)
            seeded.add(episode)

        for episode in (item for item in order if item not in seeded):
            profile = group_profiles[episode]
            scores = {name: projected_error(name, profile) for name in split_names}
            best_score = min(scores.values())
            if not math.isfinite(best_score):
                return None, {
                    "attempt": attempt, "failure": "episode_exceeds_all_split_capacities",
                    "episode": episode, "episode_records": len(profile["rows"]),
                }
            candidates = [name for name in split_names if abs(scores[name] - best_score) < 1e-12]
            assign(episode, candidates[rng.randrange(len(candidates))])

        distribution = _split_distribution(
            assigned["train"], assigned["val"], assigned["test"], group_profiles, args,
        )
        objective = sum(
            ((len(assigned[name]) - ratios[name] * total) / max(ratios[name] * total, 1)) ** 2
            for name in split_names
        )
        return assigned, {
            "attempt": attempt, "objective": objective,
            "records": {name: len(assigned[name]) for name in split_names},
            "episodes": {name: len(assigned_episodes[name]) for name in split_names},
            "distribution_failure_count": len(distribution["failures"]),
        }

    def run_town_stratified_attempt():
        """Try V4A's town-wise episode split before global greedy search.

        The former V4B-only global split could put many complete Town03
        episodes in test even though a town-wise split was feasible.  This
        candidate preserves episode isolation and makes every populous town
        contribute proportionally to each split; it is accepted only if the
        same V4B split contract below reports no failures.
        """
        previous_seed = args.seed
        try:
            # `_stratified_episode_split` is deterministic from this value and
            # mutates only `idx`/`split`, both overwritten after selection.
            args.seed = "v4b-town-stratified:%s" % previous_seed
            train, val, test = v4._stratified_episode_split(records, args)
        finally:
            args.seed = previous_seed
        assigned = {"train": train, "val": val, "test": test}
        distribution = _split_distribution(train, val, test, group_profiles, args)
        objective = sum(
            ((len(assigned[name]) - ratios[name] * total) / max(ratios[name] * total, 1)) ** 2
            for name in split_names
        )
        episode_counts = {
            name: len({str(record["episode_key"]) for record in assigned[name]})
            for name in split_names
        }
        return assigned, {
            "attempt": "town_stratified_episode",
            "objective": objective,
            "records": {name: len(assigned[name]) for name in split_names},
            "episodes": episode_counts,
            "distribution_failure_count": len(distribution["failures"]),
        }

    def run_record_balanced_attempt(attempt):
        """Greedy plus local-search assignment using record counts, not episode counts."""
        rng = random.Random("v4b-record-balanced:%s:%d" % (args.seed, attempt))
        episodes = list(group_profiles)
        rng.shuffle(episodes)
        episodes.sort(key=lambda key: -len(group_profiles[key]["rows"]))
        assignment = {}

        def objective(current):
            totals = Counter()
            towns = {name: Counter() for name in split_names}
            topologies = {name: Counter() for name in split_names}
            regions = {name: Counter() for name in split_names}
            for episode, split in current.items():
                profile = group_profiles[episode]
                totals[split] += len(profile["rows"])
                towns[split].update(profile["town"])
                topologies[split].update(profile["topology"])
                regions[split].update(profile["region"])
            score = 0.0
            for split in split_names:
                target_total = ratios[split] * total
                score += 24.0 * ((totals[split] - target_total) / max(target_total, 1.0)) ** 2
                for source, current_counts, weight in (
                        (total_by_town, towns[split], 5.0),
                        (total_by_topology, topologies[split], 4.0),
                        (total_by_region, regions[split], 2.0)):
                    for key, global_count in source.items():
                        target = ratios[split] * global_count
                        score += weight * ((current_counts[key] - target) / max(target, 1.0)) ** 2
                if totals[split]:
                    for count in towns[split].values():
                        score += 400.0 * max(0.0, count / totals[split] - args.max_split_town_fraction) ** 2
                    for count in topologies[split].values():
                        score += 400.0 * max(0.0, count / totals[split] - args.max_split_topology_fraction) ** 2
            return score

        # A purely partial objective assigns every early episode to train:
        # train has the largest target capacity.  Seed each split with a full
        # episode closest to its record-count target before greedy filling.
        remaining = set(episodes)
        train_seed = max(remaining, key=lambda key: len(group_profiles[key]["rows"]))
        assignment[train_seed] = "train"
        remaining.remove(train_seed)
        for split in ("val", "test"):
            target = ratios[split] * total
            seed = min(
                remaining,
                key=lambda key: (abs(len(group_profiles[key]["rows"]) - target),
                                 -len(group_profiles[key]["rows"]), key),
            )
            assignment[seed] = split
            remaining.remove(seed)

        for episode in (key for key in episodes if key in remaining):
            scores = {}
            for split in split_names:
                assignment[episode] = split
                scores[split] = objective(assignment)
            best = min(scores.values())
            choices = [split for split in split_names if abs(scores[split] - best) < 1e-12]
            assignment[episode] = rng.choice(choices)

        # Whole-episode local moves repair greedy decisions around large episodes.
        for _ in range(12):
            changed = False
            order = list(episodes)
            rng.shuffle(order)
            for episode in order:
                current_split = assignment[episode]
                if sum(assignment[item] == current_split for item in episodes) <= 1:
                    continue
                before = objective(assignment)
                candidates = [(before, current_split)]
                for split in split_names:
                    if split == current_split:
                        continue
                    assignment[episode] = split
                    candidates.append((objective(assignment), split))
                best_score, best_split = min(candidates, key=lambda item: item[0])
                assignment[episode] = best_split
                changed = changed or best_split != current_split
            if not changed:
                break

        assigned = {name: [] for name in split_names}
        for episode, split in assignment.items():
            assigned[split].extend(group_profiles[episode]["rows"])
        distribution = _split_distribution(
            assigned["train"], assigned["val"], assigned["test"], group_profiles, args,
        )
        return assigned, {
            "attempt": "record_balanced_episode:%d" % attempt,
            "objective": objective(assignment),
            "records": {name: len(assigned[name]) for name in split_names},
            "episodes": {name: sum(assignment[item] == name for item in episodes) for name in split_names},
            "distribution_failure_count": len(distribution["failures"]),
        }

    best_assigned, best_info, failed_attempts = None, None, []
    town_assigned, town_info = run_town_stratified_attempt()
    best_assigned, best_info = town_assigned, town_info
    if args.record_balanced_episode_split:
        for attempt in range(int(args.split_search_restarts)):
            assigned, info = run_record_balanced_attempt(attempt)
            ranking = (int(info["distribution_failure_count"] > 0), info["distribution_failure_count"], info["objective"])
            if ranking < (
                    int(best_info["distribution_failure_count"] > 0),
                    best_info["distribution_failure_count"], best_info["objective"]):
                best_assigned, best_info = assigned, info
            if info["distribution_failure_count"] == 0:
                break
    for attempt in range(int(args.split_search_restarts)):
        if best_info["distribution_failure_count"] == 0:
            break
        assigned, info = run_attempt(attempt)
        if assigned is None:
            failed_attempts.append(info)
            continue
        ranking = (int(info["distribution_failure_count"] > 0), info["distribution_failure_count"], info["objective"])
        if best_info is None or ranking < (
                int(best_info["distribution_failure_count"] > 0),
                best_info["distribution_failure_count"], best_info["objective"]):
            best_assigned, best_info = assigned, info
            if info["distribution_failure_count"] == 0:
                break
    if best_assigned is None:
        raise ValueError(json.dumps({
            "error": "v4b_group_split_no_feasible_assignment",
            "attempts": int(args.split_search_restarts), "failed_attempt_examples": failed_attempts[:10],
        }, indent=2))

    index = 0
    for split in split_names:
        for record in best_assigned[split]:
            record["idx"] = index
            record["split"] = split
            index += 1
    search = {
        "restarts": int(args.split_search_restarts), "town_stratified_candidate": town_info,
        "selected": best_info,
        "failed_attempt_count": len(failed_attempts), "failed_attempt_examples": failed_attempts[:10],
    }
    return best_assigned["train"], best_assigned["val"], best_assigned["test"], group_profiles, search


def _split_distribution(train, val, test, groups, args):
    rows_by_split = {"train": train, "val": val, "test": test}
    all_rows = train + val + test
    all_towns = Counter(str(row.get("town") or "unknown") for row in all_rows)
    all_topologies = Counter(_topology(row) for row in all_rows)
    all_regions = Counter(str((row.get("v4b_contract") or {}).get("target_region_pair") or "unknown") for row in all_rows)
    failures, split_summary = [], {}
    for split, rows in rows_by_split.items():
        towns = Counter(str(row.get("town") or "unknown") for row in rows)
        topologies = Counter(_topology(row) for row in rows)
        regions = Counter(str((row.get("v4b_contract") or {}).get("target_region_pair") or "unknown") for row in rows)
        split_summary[split] = {
            "records": len(rows), "episodes": len({str(row["episode_key"]) for row in rows}),
            "towns": dict(sorted(towns.items())), "topologies": dict(sorted(topologies.items())),
            "target_region_pairs": dict(sorted(regions.items())),
        }
        if rows:
            for town, count in towns.items():
                if count / len(rows) > args.max_split_town_fraction + 1e-12:
                    failures.append("split_town_fraction_exceeded:%s:%s:%.6f>%.6f" % (
                        split, town, count / len(rows), args.max_split_town_fraction))
            for topology, count in topologies.items():
                if count / len(rows) > args.max_split_topology_fraction + 1e-12:
                    failures.append("split_topology_fraction_exceeded:%s:%s:%.6f>%.6f" % (
                        split, topology, count / len(rows), args.max_split_topology_fraction))
        for town in all_towns:
            if args.min_records_per_town_per_split and towns[town] < args.min_records_per_town_per_split:
                failures.append("town_split_coverage:%s:%s:%d" % (split, town, towns[town]))
        for topology in all_topologies:
            if args.min_records_per_topology_per_split and topologies[topology] < args.min_records_per_topology_per_split:
                failures.append("topology_split_coverage:%s:%s:%d" % (split, topology, topologies[topology]))

    membership = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            membership[str(row["episode_key"])].add(split)
    leakage = sorted(key for key, values in membership.items() if len(values) > 1)
    if leakage:
        failures.append("episode_split_leakage:%d" % len(leakage))
    return {
        "strategy": "group_stratified_episode_town_topology_target_region",
        "episode_count": len(groups),
        "split_summary": split_summary,
        "all_town_counts": dict(sorted(all_towns.items())),
        "all_topology_counts": dict(sorted(all_topologies.items())),
        "all_target_region_pair_counts": dict(sorted(all_regions.items())),
        "episode_leakage_examples": leakage[:20],
        "failures": sorted(set(failures)),
    }


def _apply_train_weights(train, args):
    counts = Counter(_balance_key(row) for row in train)
    mean = sum(counts.values()) / max(len(counts), 1)
    values = []
    for row in train:
        original = float(row.get(args.weight_key, args.base_training_weight))
        multiplier = (mean / max(counts[_balance_key(row)], 1)) ** float(args.balance_weight_power)
        multiplier = min(max(multiplier, 1.0 / args.max_balance_weight), args.max_balance_weight)
        row[args.weight_key] = float(original * multiplier)
        row["v4b_balance_bucket"] = _balance_key(row)
        values.append(row[args.weight_key])
    return {
        "weight_key": args.weight_key, "bucket_counts": dict(sorted(counts.items())),
        "min_weight": min(values) if values else None, "mean_weight": sum(values) / len(values) if values else None,
        "max_weight": max(values) if values else None,
    }


def _apply_interaction_train_weights(train, args):
    """Apply an offline-only sampling multiplier, then erase all score values.

    `training_weight` is consumed by the trainer's sampler, never by the graph
    builder. The raw label score exists only under the private build key until
    this function has computed an aggregate multiplier.
    """
    private_key = "_v4b_build_only_target_interaction"
    values = [float((row.get(private_key) or {}).get("min", 0.0)) for row in train]
    if args.interaction_score_weight_power == 0.0 or not values:
        multipliers = [1.0 for _ in train]
    else:
        # A tiny positive floor preserves low-score examples for support while
        # the mean normalization keeps the average training weight unchanged.
        raw = [max(value, 1e-3) ** float(args.interaction_score_weight_power) for value in values]
        normalizer = sum(raw) / len(raw)
        multipliers = [min(max(value / max(normalizer, 1e-12), 1.0 / args.max_interaction_score_weight),
                           args.max_interaction_score_weight) for value in raw]
    for row, multiplier in zip(train, multipliers):
        row[args.weight_key] = float(row.get(args.weight_key, 1.0)) * float(multiplier)
    return {
        "enabled": bool(args.interaction_score_weight_power > 0.0),
        "power": float(args.interaction_score_weight_power),
        "max_weight": float(args.max_interaction_score_weight),
        "minimum_target_score": _quantiles(values),
        "multiplier": _quantiles(multipliers),
    }


def _erase_build_only_interaction(records):
    for row in records:
        row.pop("_v4b_build_only_target_interaction", None)
        row.pop("_v4b_build_only_target_source_actor_id_values", None)


def _write_route_interaction_manifest(records, path: Path):
    """Write only selected target identities for an offline CARLA route pass."""
    if path is None:
        return None
    if path.exists():
        raise FileExistsError("refusing to overwrite route interaction manifest: %s" % path)
    rows = []
    for record in records:
        try:
            source_key = str(record["source_file_sha1"])
            actor_ids = [int(value) for value in record["_v4b_build_only_target_source_actor_id_values"]]
        except (KeyError, TypeError, ValueError):
            raise ValueError("record missing V4B target manifest identity: %s" % record.get("scene_id"))
        if len(actor_ids) != 2 or len(set(actor_ids)) != 2:
            raise ValueError("V4B target manifest must contain exactly two distinct actor IDs: %s" % record.get("scene_id"))
        rows.append({
            "schema_version": "rift_route_target_manifest_v1",
            "source_file_sha1": source_key,
            "scene_id": record.get("scene_id"),
            "target_actor_ids": actor_ids,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(path), rows)
    return {"path": str(path), "records": len(rows), "target_actors": len(rows) * 2}


def _quantiles(values):
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "max": None}
    values = sorted(float(value) for value in values)
    def percentile(fraction):
        return values[min(len(values) - 1, int(round((len(values) - 1) * fraction)))]
    return {
        "count": len(values), "min": values[0], "mean": sum(values) / len(values),
        "p50": percentile(0.50), "p90": percentile(0.90), "max": values[-1],
    }


def _load_route_interaction_sidecar(path: Path):
    """Index additive offline route labels by builder source hash and actor id."""
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    indexed = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                source_key = str(item["source_file_sha1"])
                actor_id = str(int(item["actor_id"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("invalid route interaction sidecar at line %d: %s" % (line_number, exc))
            if actor_id in indexed[source_key]:
                raise ValueError("duplicate sidecar actor entry: %s/%s" % (source_key, actor_id))
            indexed[source_key][actor_id] = item
    return indexed


def _summary(base_summary, split_distribution, train_weights, interaction_weights, target_interaction):
    base_summary["schema_version"] = "hv_scene_dataset_v4b_group_stratified_v1"
    base_summary["v4b"] = {
        "split_distribution": split_distribution,
        "train_weighting": train_weights,
        "interaction_score_train_weighting": interaction_weights,
        # V4B removes selection scores before serialization.  V4C may retain
        # target-only route-interaction supervision, never condition labels.
        "selected_target_interaction_score": target_interaction,
    }
    return base_summary


def main():
    args = parse_args()
    validate_args(args)
    extractor = None if args.map_source == "snapshot" else v3.HDMapLaneExtractor(args.rift_root, args.rift_hd_map_dir)
    route_sidecar = _load_route_interaction_sidecar(args.route_interaction_sidecar)
    raw_records, rejection_counts = [], Counter()
    map_source_counts, map_errors = Counter(), Counter()
    record_build_error_examples, invalid_input_examples = [], []
    valid_labeled_snapshot_count = processed = 0
    selected_pair_min_scores, selected_pair_sum_scores = [], []
    coverage = {
        "full_fov_actor_count_histogram": Counter(), "full_fov_rift_positive_count_histogram": Counter(),
        "forward_rift_positive_count_histogram": Counter(), "local_lane_count_histogram": Counter(),
        "rift_positive_region_counts": Counter(),
    }
    for path in sorted(args.input_dir.rglob("*.json")):
        if args.max_input_files > 0 and processed >= args.max_input_files:
            break
        processed += 1
        try:
            scene = base.load_snapshot(path)
        except Exception as exc:
            rejection_counts["snapshot_load:%s" % type(exc).__name__] += 1
            continue
        if not v4._is_rift_labeled_snapshot(scene):
            rejection_counts["not_rift_labeled_snapshot"] += 1
            if len(invalid_input_examples) < 20:
                invalid_input_examples.append({"path": str(path), "schema_version": scene.get("schema_version") if isinstance(scene, dict) else None})
            continue
        valid_labeled_snapshot_count += 1
        snapshot_lanes = scene.get("map_lanes", []) or []
        if extractor is not None and (args.map_source == "hd_map" or not snapshot_lanes):
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
            v4._update_candidate_coverage(coverage, norm, lanes, args)
            source_file_sha1 = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
            record, reason = _build_record(
                scene, lanes, args, norm,
                sidecar_targets=route_sidecar.get(source_file_sha1),
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejection_counts["record_build:%s" % type(exc).__name__] += 1
            if len(record_build_error_examples) < 20:
                record_build_error_examples.append({"path": str(path), "type": type(exc).__name__, "message": str(exc)})
            continue
        if not record:
            rejection_counts[reason or "unknown"] += 1
            continue
        build_only = record["_v4b_build_only_target_interaction"]
        selected_pair_min_scores.append(build_only["min"])
        selected_pair_sum_scores.append(build_only["sum"])
        record["source_file_sha1"] = source_file_sha1
        raw_records.append(record)

    records, quality_reasons, quality_rejected, duplicates = v2.filter_and_weight(raw_records, args)
    if not records:
        raise ValueError("V4B produced zero records; inspect diagnostics_json before changing collection.")
    records, global_town_balance = _enforce_global_town_fraction(records, args)
    if not records:
        raise ValueError("V4B global town balancing removed every record.")
    route_manifest = _write_route_interaction_manifest(records, args.route_interaction_manifest)
    if args.route_interaction_manifest_only:
        summary = {
            "schema_version": "rift_route_target_manifest_summary_v1",
            "processed_files": processed,
            "valid_rift_labeled_snapshot_count": int(valid_labeled_snapshot_count),
            "num_raw_records_before_v2_quality_and_dedup": len(raw_records),
            "num_records_after_v2_quality_and_dedup": len(records),
            "num_quality_rejected": quality_rejected,
            "num_duplicates": duplicates,
            "selection_rejections": dict(sorted(rejection_counts.items())),
            "quality_reject_reasons": dict(sorted(quality_reasons.items())),
            "route_interaction_manifest": route_manifest,
            "args": v2.json_safe(vars(args)),
        }
        if args.diagnostics_json:
            v4._write_diagnostics(args.diagnostics_json, summary)
        print(json.dumps(summary, indent=2))
        return
    train, val, test, groups, split_search = _group_stratified_episode_split(records, args)
    if not train or not val or not test:
        raise ValueError(json.dumps({
            "error": "v4b_group_split_produced_empty_split",
            "num_records": len(records),
            "num_episodes": len(groups),
            "train_records": len(train), "val_records": len(val), "test_records": len(test),
        }, indent=2))
    distribution = _split_distribution(train, val, test, groups, args)
    distribution["search"] = split_search
    # Retain every pre-existing V4-A global diversity guard.  V4B adds
    # per-split diagnostics; it does not weaken the accepted-data contract.
    legacy_distribution = v4._strict_distribution_check(train, val, test, args)
    distribution["v4a_global_contract"] = legacy_distribution
    train_weights = _apply_train_weights(train, args)
    interaction_weights = _apply_interaction_train_weights(train, args)
    _erase_build_only_interaction(train)
    _erase_build_only_interaction(val)
    _erase_build_only_interaction(test)
    base_summary = v4._build_summary(
        args=args, processed=processed, raw_records=raw_records, records=records,
        rejection_counts=rejection_counts, quality_rejections=quality_reasons,
        quality_rejected=quality_rejected, duplicates=duplicates,
        map_source_counts=map_source_counts, map_errors=map_errors, coverage=coverage,
        record_build_error_examples=record_build_error_examples,
        valid_labeled_snapshot_count=valid_labeled_snapshot_count,
        invalid_input_examples=invalid_input_examples, train=train, val=val, test=test,
        distribution=distribution,
    )
    summary = _summary(
        base_summary, distribution, train_weights, interaction_weights,
        {"minimum_target_score": _quantiles(selected_pair_min_scores),
         "pair_score_sum": _quantiles(selected_pair_sum_scores)},
    )
    summary["global_town_balance"] = global_town_balance
    summary["route_interaction_manifest"] = route_manifest
    if args.diagnostics_json:
        v4._write_diagnostics(args.diagnostics_json, summary)
    if args.diagnostics_only:
        print(json.dumps(summary, indent=2))
        return
    if len(records) < args.min_accepted_records:
        raise ValueError(json.dumps(summary, indent=2))
    if legacy_distribution["failures"] or (args.require_split_balance and distribution["failures"]):
        raise ValueError(json.dumps(summary, indent=2))
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite existing V4B output: %s" % args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        write_jsonl(str(args.output_dir / (name + ".jsonl")), rows)
    for name in ("summary.json", "quality_report.json"):
        (args.output_dir / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
