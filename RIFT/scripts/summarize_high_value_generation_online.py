#!/usr/bin/env python
"""Summarize online high-value generated-scene dumps."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--statistics_json", type=Path, default=None)
    parser.add_argument("--statistics_dir", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    w = pos - lo
    return values[lo] * (1.0 - w) + values[hi] * w


def numeric(values):
    values = [float(v) for v in values if v is not None]
    return {
        "min": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values) if values else None,
    }


def load_payloads(dump_dir):
    payloads = []
    for path in sorted(dump_dir.glob("*_online_generation.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        item["_path"] = str(path)
        payloads.append(item)
    return payloads


def load_lifecycle_events(dump_dir):
    path = dump_dir / "lifecycle.jsonl"
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                event["_line_number"] = line_number
                events.append(event)
    return events


def load_lifecycle_metadata(dump_dir):
    path = dump_dir / "lifecycle_metadata.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def summarize_lifecycle(events, metadata=None):
    metadata = metadata or {}
    event_counts = Counter(str(event.get("event") or "unknown") for event in events)

    def actor_key(event):
        """Avoid conflating actor ids if CARLA restarts its id counter."""
        try:
            actor_id = int(event.get("actor_id"))
        except (TypeError, ValueError):
            return None
        return str(event.get("scene_id") or "unknown"), actor_id

    spawned = {
        actor_key(event) for event in events
        if event.get("event") == "spawned" and actor_key(event) is not None
    }
    activated = {
        actor_key(event) for event in events
        if event.get("event") == "rift_cbv_activated" and actor_key(event) is not None
    }
    terminal_counts = Counter(
        str(event.get("event") or "unknown")
        for event in events
        if bool(event.get("terminal", False))
    )
    spawned_by_source = {}
    activated_by_source = {}
    for source in sorted({str(event.get("generator_source") or "unknown") for event in events}):
        source_spawned = {
            actor_key(event) for event in events
            if event.get("event") == "spawned" and actor_key(event) is not None and
            str(event.get("generator_source") or "unknown") == source
        }
        source_activated = {
            actor_key(event) for event in events
            if event.get("event") == "rift_cbv_activated" and actor_key(event) is not None and
            str(event.get("generator_source") or "unknown") == source
        }
        spawned_by_source[source] = len(source_spawned)
        activated_by_source[source] = len(source_activated)
    total_conversion_rate = len(activated) / len(spawned) if spawned else None
    high_value_spawned = spawned_by_source.get("high_value", 0)
    background_spawned = spawned_by_source.get("scenario_dreamer_background", 0)
    actor_histories = {}
    for event in events:
        actor_id = event.get("actor_id")
        if actor_id is None:
            continue
        try:
            actor_id = int(actor_id)
        except (TypeError, ValueError):
            continue
        # Actor ids are unique within a CARLA process, but a scene id makes
        # summaries robust to a simulator that restarts its id counter.
        key = (str(event.get("scene_id") or "unknown"), actor_id)
        history = actor_histories.setdefault(key, {
            "source": str(event.get("generator_source") or "unknown"),
            "spawned": None,
            "activated": None,
            "terminal": None,
            "green_periods": [],
            "green_active_frame_count": 0,
        })
        source = event.get("generator_source")
        if source is not None:
            history["source"] = str(source)
        kind = str(event.get("event") or "unknown")
        if kind == "spawned" and history["spawned"] is None:
            history["spawned"] = event
        elif kind == "rift_cbv_activated" and history["activated"] is None:
            history["activated"] = event
        elif kind == "rift_cbv_green_period_ended":
            history["green_periods"].append(event)
        if event.get("green_active_frame_count") is not None:
            try:
                history["green_active_frame_count"] = max(
                    int(history["green_active_frame_count"]),
                    int(event["green_active_frame_count"]),
                )
            except (TypeError, ValueError):
                pass
        if bool(event.get("terminal", False)) and history["terminal"] is None:
            history["terminal"] = event

    outcome_by_source = {}
    fidelity_rows = []
    acceptance_rows = []
    for source in sorted({history["source"] for history in actor_histories.values()}):
        histories = [history for history in actor_histories.values() if history["source"] == source]
        activation_frames = []
        green_active_frames = []
        green_period_frames = []
        terminal_events = Counter()
        activated_count = 0
        never_activated_terminal = 0
        for history in histories:
            spawned_event = history["spawned"] or {}
            activated_event = history["activated"]
            terminal_event = history["terminal"]
            if activated_event is not None:
                activated_count += 1
                value = activated_event.get("frames_from_spawn")
                if value is not None:
                    activation_frames.append(value)
            if history["green_active_frame_count"]:
                green_active_frames.append(history["green_active_frame_count"])
            for green_period in history["green_periods"]:
                value = green_period.get("green_period_frames")
                if value is not None:
                    green_period_frames.append(value)
            if terminal_event is not None:
                terminal_events[str(terminal_event.get("event") or "unknown")] += 1
                if activated_event is None:
                    never_activated_terminal += 1
            if spawned_event.get("post_spawn_observation_available"):
                fidelity_rows.append(spawned_event)
            if spawned_event.get("spawn_acceptance_observation_available"):
                acceptance_rows.append(spawned_event)
        outcome_by_source[source] = {
            "spawned_actor_count": len(histories),
            "rift_cbv_activated_actor_count": activated_count,
            "activation_rate": activated_count / len(histories) if histories else None,
            "activation_frames_from_spawn": numeric(activation_frames),
            "green_active_frames": numeric(green_active_frames),
            "green_period_frames": numeric(green_period_frames),
            "terminal_event_counts": dict(terminal_events.most_common()),
            "never_activated_terminal_actor_count": never_activated_terminal,
        }

    def fidelity_metric(key):
        return numeric(event.get(key) for event in fidelity_rows)

    def acceptance_metric(key):
        return numeric(event.get(key) for event in acceptance_rows)

    return {
        "enabled": bool(metadata.get("lifecycle_telemetry_enabled", False)),
        "metadata_present": bool(metadata),
        "event_count": len(events),
        "event_counts": dict(event_counts.most_common()),
        "spawned_actor_count": len(spawned),
        "rift_cbv_activated_actor_count": len(activated),
        "rift_cbv_conversion_rate": total_conversion_rate,
        # Source-specific names match the BEV state machine: SD blue ->
        # purple and HV orange -> green, only after normal RIFT activation.
        "blue_to_purple_conversion_rate": (
            activated_by_source.get("scenario_dreamer_background", 0) / background_spawned
            if background_spawned else None
        ),
        "orange_to_green_conversion_rate": (
            activated_by_source.get("high_value", 0) / high_value_spawned
            if high_value_spawned else None
        ),
        "spawned_actor_count_by_source": spawned_by_source,
        "rift_cbv_activated_actor_count_by_source": activated_by_source,
        "rift_cbv_conversion_rate_by_source": {
            source: activated_by_source[source] / count if count else None
            for source, count in spawned_by_source.items()
        },
        "terminal_event_counts": dict(terminal_counts.most_common()),
        "actor_outcomes_by_source": outcome_by_source,
        # Captured before the first world.tick after CARLA accepts the actor.
        # It is the direct raw x/y/yaw landing check; z remains excluded.
        "spawn_acceptance_raw_transform_fidelity": {
            "observed_actor_count": len(acceptance_rows),
            "xy_error_m": acceptance_metric("spawn_acceptance_xy_error_m"),
            "yaw_error_deg": acceptance_metric("spawn_acceptance_yaw_error_deg"),
            "speed_error_mps": acceptance_metric("spawn_acceptance_speed_error_mps"),
        },
        # Present only for newly collected telemetry.  z is intentionally
        # absent: terrain height is supplied by CARLA rather than the model.
        "post_spawn_raw_state_fidelity": {
            "observed_actor_count": len(fidelity_rows),
            "xy_error_m": fidelity_metric("post_spawn_xy_error_m"),
            "yaw_error_deg": fidelity_metric("post_spawn_yaw_error_deg"),
            "speed_error_mps": fidelity_metric("post_spawn_speed_error_mps"),
        },
    }


def summarize_payloads(payloads, lifecycle_events=None, lifecycle_metadata=None):
    failures = Counter()
    failure_stages = Counter()
    failure_stage_by_mode = Counter()
    generation_modes = Counter()
    generator_called_by_mode = Counter()
    skipped_by_stage = Counter()
    audit_failures = Counter()
    generated_rift_cbv_count_buckets = Counter()
    generated_counts = []
    spawned_counts = []
    generated_positive_counts = []
    cbv_registration_counts = []
    positive_after = []
    hidden_counts = []
    observed_counts = []
    v4a_context_counts = []
    v4a_forward_counts = []
    v4a_forward_violations = []
    v4a_raw_proposals = 0
    rolling_frames = []
    success = 0
    spawn_success = 0
    audit_success = 0
    audit_failed = 0
    skipped = 0
    generator_called = 0
    spawn_failure_count = 0
    cbv_registration_enabled = 0
    cbv_registration_failure_count = 0
    cbv_route_update_failures = Counter()
    spawned_by_source = Counter()
    immediate_positive_by_source = Counter()

    for payload in payloads:
        result = payload.get("result") or {}
        proposal = payload.get("proposal") or {}
        mode = str(result.get("generation_mode") or proposal.get("requested_generation_mode") or "unknown")
        generation_modes[mode] += 1
        called = bool(result.get("generator_called", False))
        if called:
            generator_called_by_mode[mode] += 1
        generated = proposal.get("generated_agent_states") or []
        v4a_contract = proposal.get("v4a_condition_contract") or {}
        v4a_diagnostics = proposal.get("v4a_raw_output_diagnostics") or {}
        if v4a_contract or v4a_diagnostics:
            v4a_raw_proposals += 1
            v4a_context_counts.append(v4a_contract.get("observed_actor_count"))
            v4a_forward_counts.append(v4a_diagnostics.get("raw_generated_forward_count"))
            v4a_forward_violations.append(v4a_diagnostics.get("raw_generated_forward_violation_count"))
        generated_counts.append(len(generated))
        spawned = result.get("spawned_actor_ids") or []
        spawned_counts.append(len(spawned))
        for source in (result.get("spawned_actor_sources") or {}).values():
            spawned_by_source[str(source)] += 1
        generated_positive_counts.append(result.get("num_generated_rift_positive_agents"))
        for source, count in (result.get("generated_rift_positive_count_by_source") or {}).items():
            immediate_positive_by_source[str(source)] += int(count)
        if called:
            generated_rift_cbv_count_buckets[str(result.get(
                "generated_rift_cbv_count_bucket",
                min(3, int(result.get("num_generated_rift_positive_agents") or 0)),
            ))] += 1
        cbv_registration_counts.append(result.get("cbv_registration_count"))
        cbv_registration_enabled += int(bool(result.get("cbv_registration_enabled", False)))
        cbv_registration_failure_count += len(result.get("cbv_registration_failures") or [])
        if result.get("cbv_route_update_failure"):
            cbv_route_update_failures[str(result["cbv_route_update_failure"])] += 1
        positive_after.append(result.get("num_rift_positive_agents_after_generation"))
        success += int(bool(result.get("success", False)))
        spawn_success += int(len(spawned) > 0)
        audit_success += int(bool(result.get("rift_audit_passed", False)))
        audit_failed_flag = bool(result.get("audit_failed", False))
        audit_failed += int(audit_failed_flag)
        if audit_failed_flag and result.get("audit_observation"):
            audit_failures[str(result["audit_observation"])] += 1
        skipped += int(bool(result.get("skipped", False)))
        generator_called += int(called)
        spawn_failure_count += len(result.get("spawn_failures") or [])
        hidden_counts.append(result.get("num_hidden_context_actors", len(result.get("hidden_context_actor_ids") or [])))
        observed_counts.append(result.get("observed_agent_count_before_generation"))
        if result.get("rolling_generation", False):
            rolling_frames.append(result.get("frame_id"))
        if result.get("failure_reason"):
            failures[str(result["failure_reason"])] += 1
        if result.get("failure_stage"):
            stage = str(result["failure_stage"])
            failure_stages[stage] += 1
            failure_stage_by_mode[f"{mode}:{stage}"] += 1
            if result.get("skipped", False):
                skipped_by_stage[stage] += 1

    total = len(payloads)
    v4a_forward_denominator = sum(
        float(forward) + float(violation)
        for forward, violation in zip(v4a_forward_counts, v4a_forward_violations)
        if forward is not None and violation is not None
    )
    v4a_forward_violation_total = sum(
        float(violation) for violation in v4a_forward_violations if violation is not None
    )
    generator_called_rate_by_mode = {
        mode: generator_called_by_mode[mode] / count
        for mode, count in generation_modes.items()
    }
    return {
        "schema_version": "hv_generation_online_summary_v1",
        "num_dumps": total,
        "generation_mode_counts": dict(generation_modes.most_common()),
        "generator_called_count": generator_called,
        "generator_called_count_by_mode": dict(generator_called_by_mode.most_common()),
        "generator_called_rate_by_mode": generator_called_rate_by_mode,
        "generation_success_rate": success / total if total else 0.0,
        "carla_spawn_success_rate": spawn_success / total if total else 0.0,
        "rift_audit_pass_rate": audit_success / total if total else 0.0,
        "generated_rift_audit_pass_rate": audit_success / total if total else 0.0,
        "rift_audit_failed_count": audit_failed,
        "rift_audit_failed_rate": audit_failed / total if total else 0.0,
        "skipped_count": skipped,
        "skipped_rate": skipped / total if total else 0.0,
        "generator_called_rate": generator_called / total if total else 0.0,
        "spawn_failure_count": spawn_failure_count,
        "generated_agent_count": numeric(generated_counts),
        "spawned_agent_count": numeric(spawned_counts),
        "spawned_agent_count_by_source": dict(spawned_by_source.most_common()),
        "generated_rift_positive_agent_count": numeric(generated_positive_counts),
        "immediate_generated_rift_positive_count_by_source": dict(immediate_positive_by_source.most_common()),
        "generated_rift_cbv_count_buckets": dict(generated_rift_cbv_count_buckets),
        "cbv_registration_enabled_count": cbv_registration_enabled,
        "cbv_registration_enabled_rate": cbv_registration_enabled / total if total else 0.0,
        "cbv_registration_count": numeric(cbv_registration_counts),
        "cbv_registration_failure_count": cbv_registration_failure_count,
        "cbv_route_update_failures": dict(cbv_route_update_failures.most_common()),
        "hidden_context_actor_count": numeric(hidden_counts),
        "observed_agent_count_before_generation": numeric(observed_counts),
        "v4a_raw_contract": {
            "proposal_count": v4a_raw_proposals,
            "full_fov_context_actor_count": numeric(v4a_context_counts),
            "raw_generated_forward_count": numeric(v4a_forward_counts),
            "raw_generated_forward_violation_count": numeric(v4a_forward_violations),
            "raw_generated_forward_violation_rate": (
                v4a_forward_violation_total / v4a_forward_denominator
                if v4a_forward_denominator > 0.0 else None
            ),
            "state_postprocessing": "none; diagnostics only",
        },
        "num_rift_positive_agents_after_generation": numeric(positive_after),
        "rolling_trigger_count": len(rolling_frames),
        "rolling_trigger_frames": sorted(int(frame) for frame in rolling_frames if frame is not None),
        "failure_stage_counts": dict(failure_stages.most_common()),
        "failure_stage_by_mode": dict(failure_stage_by_mode.most_common()),
        "skipped_by_stage": dict(skipped_by_stage.most_common()),
        "failure_reasons": dict(failures.most_common()),
        "audit_failure_reasons": dict(audit_failures.most_common()),
        "lifecycle": summarize_lifecycle(lifecycle_events or [], lifecycle_metadata),
    }


def find_statistics(args):
    if args.statistics_json:
        return args.statistics_json
    if args.statistics_dir:
        candidates = sorted(
            args.statistics_dir.rglob("simulation_results.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None
    return None


def count_record_collisions(record):
    infractions = record.get("infractions", {})
    keys = ("collisions_pedestrian", "collisions_vehicle", "collisions_layout")
    return sum(len(infractions.get(key, []) or []) for key in keys)


def summarize_statistics(path):
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    records = checkpoint.get("records", [])
    route_completion = [float(record.get("scores", {}).get("score_route", 0.0)) for record in records]
    driving_score = [float(record.get("scores", {}).get("score_composed", 0.0)) for record in records]
    collisions = [count_record_collisions(record) for record in records]
    return {
        "statistics_json": str(path),
        "entry_status": data.get("entry_status"),
        "eligible": data.get("eligible"),
        "num_routes": len(records),
        "route_completion": numeric(route_completion),
        "driving_score": numeric(driving_score),
        "collision_routes_ratio": (
            sum(1 for count in collisions if count > 0) / len(collisions)
            if collisions else None
        ),
        "collision_count_per_route": numeric(collisions),
        "global_scores_mean": global_record.get("scores_mean", {}),
    }


def main():
    args = parse_args()
    payloads = load_payloads(args.dump_dir)
    summary = summarize_payloads(
        payloads,
        load_lifecycle_events(args.dump_dir),
        load_lifecycle_metadata(args.dump_dir),
    )
    stats_path = find_statistics(args)
    if stats_path is not None:
        summary["rift_statistics"] = summarize_statistics(stats_path)
    print(json.dumps(summary, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
