#!/usr/bin/env python
"""Summarize eventual normal-RIFT activation per online HV generation event.

This script is read-only analysis.  It does not rerank samples, control an
actor, or treat an orange-to-green conversion as a training label.  Its main
metric answers the closed-loop question for a fixed-two-HV generator: among
events that really spawned two HV actors, how often did both later become
normal RIFT CBVs?
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from summarize_high_value_generation_online import load_lifecycle_events, load_payloads


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=Path, required=True)
    parser.add_argument("--source", default="high_value")
    parser.add_argument("--expected_hv_per_event", type=int, default=2)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def actor_key(scene_id, actor_id):
    try:
        return str(scene_id or "unknown"), int(actor_id)
    except (TypeError, ValueError):
        return None


def numeric(values):
    values = sorted(float(value) for value in values if value is not None)
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p90": None, "max": None}

    def percentile(q):
        return values[min(len(values) - 1, int(round((len(values) - 1) * q)))]

    return {
        "count": len(values), "min": values[0], "mean": sum(values) / len(values),
        "p50": percentile(0.50), "p90": percentile(0.90), "max": values[-1],
    }


def lifecycle_histories(events):
    histories = {}
    for event in events:
        key = actor_key(event.get("scene_id"), event.get("actor_id"))
        if key is None:
            continue
        history = histories.setdefault(key, {
            "source": str(event.get("generator_source") or "unknown"),
            "activated": None,
            "green_active_frame_count": 0,
            "terminal": None,
        })
        if event.get("generator_source") is not None:
            history["source"] = str(event["generator_source"])
        if event.get("event") == "rift_cbv_activated" and history["activated"] is None:
            history["activated"] = event
        try:
            history["green_active_frame_count"] = max(
                int(history["green_active_frame_count"]),
                int(event.get("green_active_frame_count") or 0),
            )
        except (TypeError, ValueError):
            pass
        if bool(event.get("terminal", False)) and history["terminal"] is None:
            history["terminal"] = event
    return histories


def hv_actor_ids(result, source):
    sources = result.get("spawned_actor_sources") or {}
    ids = []
    for value in result.get("spawned_actor_ids") or []:
        try:
            actor_id = int(value)
        except (TypeError, ValueError):
            continue
        actor_source = str(sources.get(str(actor_id), sources.get(actor_id, "high_value")))
        if actor_source == source:
            ids.append(actor_id)
    return ids


def main():
    args = parse_args()
    if args.expected_hv_per_event < 1:
        raise ValueError("--expected_hv_per_event must be positive")
    payloads = load_payloads(args.dump_dir)
    events = load_lifecycle_events(args.dump_dir)
    histories = lifecycle_histories(events)
    rows, latencies, green_frames, terminal_reasons = [], [], [], Counter()
    successful_events = 0
    for payload in payloads:
        result = payload.get("result") or {}
        if not bool(result.get("generator_called", False)):
            continue
        if not bool(result.get("success", False)):
            rows.append({
                "scene_id": result.get("scene_id"), "frame_id": result.get("frame_id"),
                "success": False, "failure_stage": result.get("failure_stage"),
                "failure_reason": result.get("failure_reason"),
            })
            continue
        successful_events += 1
        scene_id = result.get("scene_id")
        actor_ids = hv_actor_ids(result, args.source)
        immediate_ids = {
            int(value) for value in (result.get("generated_rift_positive_actor_ids") or [])
            if str((result.get("spawned_actor_sources") or {}).get(str(value), "high_value")) == args.source
        }
        activated_ids, event_terminal = [], []
        actor_rows = []
        for actor_id in actor_ids:
            history = histories.get(actor_key(scene_id, actor_id), {})
            activated = history.get("activated")
            terminal = history.get("terminal")
            if activated is not None:
                activated_ids.append(actor_id)
                latency = activated.get("frames_from_spawn")
                if latency is not None:
                    latencies.append(latency)
            frames = int(history.get("green_active_frame_count") or 0)
            if frames:
                green_frames.append(frames)
            terminal_reason = None if terminal is None else str(terminal.get("event") or "unknown")
            if terminal_reason:
                terminal_reasons[terminal_reason] += 1
                event_terminal.append(terminal_reason)
            actor_rows.append({
                "actor_id": actor_id,
                "immediate_rift_positive": actor_id in immediate_ids,
                "eventually_rift_cbv": activated is not None,
                "activation_latency_frames": None if activated is None else activated.get("frames_from_spawn"),
                "green_active_frame_count": frames,
                "terminal_reason": terminal_reason,
            })
        rows.append({
            "scene_id": scene_id, "frame_id": result.get("frame_id"), "success": True,
            "spawned_hv_actor_count": len(actor_ids),
            "immediate_rift_positive_hv_count": len(set(actor_ids) & immediate_ids),
            "eventual_rift_cbv_hv_count": len(activated_ids),
            "eventual_two_rift_cbv": len(activated_ids) >= args.expected_hv_per_event,
            "actor_rows": actor_rows, "terminal_reasons": event_terminal,
        })

    spawned_expected = [row for row in rows if row.get("success") and row["spawned_hv_actor_count"] == args.expected_hv_per_event]
    all_successful = [row for row in rows if row.get("success")]
    report = {
        "schema_version": "hv_eventual_rift_cbv_event_analysis_v1",
        "dump_dir": str(args.dump_dir), "source": args.source,
        "expected_hv_per_event": int(args.expected_hv_per_event),
        "generation_events_called": len(rows), "generation_events_successful": successful_events,
        "generation_events_with_exact_expected_hv_spawn": len(spawned_expected),
        "eventually_at_least_one_rift_cbv_rate_among_successful_events": (
            sum(row["eventual_rift_cbv_hv_count"] >= 1 for row in all_successful) / len(all_successful)
            if all_successful else None
        ),
        "eventually_two_rift_cbv_rate_among_exact_two_hv_spawn_events": (
            sum(row["eventual_two_rift_cbv"] for row in spawned_expected) / len(spawned_expected)
            if spawned_expected else None
        ),
        "eventual_rift_cbv_hv_count_histogram": dict(sorted(Counter(
            str(row["eventual_rift_cbv_hv_count"]) for row in all_successful
        ).items())),
        "immediate_rift_positive_hv_count_histogram": dict(sorted(Counter(
            str(row["immediate_rift_positive_hv_count"]) for row in all_successful
        ).items())),
        "activation_latency_frames": numeric(latencies),
        "green_active_frame_count": numeric(green_frames),
        "terminal_reason_counts": dict(terminal_reasons.most_common()),
        "event_rows": rows,
        "interpretation": (
            "RIFT remains observational here. Eventual RIFT-CBV activation is a closed-loop outcome, "
            "not a sampling filter, online loss, actor-state edit, or direct registration rule."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "event_rows"}, indent=2))


if __name__ == "__main__":
    main()
