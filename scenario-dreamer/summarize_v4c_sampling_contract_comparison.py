#!/usr/bin/env python
"""Read-only C0/C2 sampling-contract comparison.

This tool summarizes files emitted by ``audit_hv_sampling_contract.py``.  It
does not sample, filter, clamp, or rewrite generated states.  In particular,
it keeps the three diagnostic paths separate:

* AE oracle: true target latent is available;
* noisy-x0: true target latent is available;
* reverse diffusion: condition-only path equivalent to online generation.

The paired C2-C0 section pairs only same ``record_index`` and same method.
Correlation is descriptive and must not be interpreted as a causal loss study.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


METHODS = (
    "ae_oracle_true_target_latent",
    "training_time_noisy_true_target_x0",
    "reverse_diffusion_condition_only_s1",
)
STATE_FIELDS = ("x_local_m", "y_local_m", "heading_rad", "speed_mps", "length_m", "width_m")
BOUNDS = {
    "speed_mps": (0.0, 30.0),
    "length_m": (2.0, 8.0),
    "width_m": (1.4, 3.2),
}
PAIR_METRICS = (
    "position_l2",
    "position_l1",
    "heading_abs",
    "speed_l1",
    "size_l1",
    "physical_invalid_rate",
    "nearest_lane_corridor_failure_count",
    "associated_lane_corridor_failure_count",
    "forward_violation_count",
    "obb_overlap_count",
    "interaction_route_distance_abs_error_m",
    "interaction_route_sync_abs_gap_m",
    "interaction_route_preview_lateral_error_m",
    "interaction_route_heading_tangent_error",
)
ROUTE_METRICS = (
    "interaction_route_distance_abs_error_m",
    "interaction_route_sync_abs_gap_m",
    "interaction_route_preview_lateral_error_m",
    "interaction_route_heading_tangent_error",
)
FEASIBILITY_METRICS = (
    "physical_invalid_rate",
    "nearest_lane_corridor_failure_count",
    "associated_lane_corridor_failure_count",
    "forward_violation_count",
    "obb_overlap_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c0_report", required=True)
    parser.add_argument("--c0_rows", required=True)
    parser.add_argument("--c2_report", required=True)
    parser.add_argument("--c2_rows", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_markdown", default=None)
    return parser.parse_args()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = int(math.floor(index)), int(math.ceil(index))
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)


def distribution(values):
    values = [v for value in values if (v := number(value)) is not None]
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p50": quantile(values, 0.50),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "max": max(values) if values else None,
    }


def pearson(xs, ys):
    pairs = [(number(x), number(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x - mx) ** 2 for x in xs)
    yy = sum((y - my) ** 2 for y in ys)
    if xx <= 0.0 or yy <= 0.0:
        return None, len(pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(xx * yy), len(pairs)


def ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[indexed[k][0]] = rank
        i = j
    return result


def spearman(xs, ys):
    pairs = [(number(x), number(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    px, py = zip(*pairs)
    return pearson(ranks(px), ranks(py))


def rows_by_method(rows):
    grouped = defaultdict(dict)
    for row in rows:
        method = row.get("method")
        index = row.get("record_index")
        if method not in METHODS or not isinstance(index, int):
            continue
        if index in grouped[method]:
            raise ValueError("duplicate (%s, record_index=%s)" % (method, index))
        grouped[method][index] = row
    return grouped


def state_summary(rows):
    values = defaultdict(list)
    invalid = {name: {"below": 0, "above": 0} for name in BOUNDS}
    actor_count = 0
    for row in rows:
        for state in row.get("states") or []:
            if not isinstance(state, (list, tuple)):
                continue
            actor_count += 1
            for index, name in enumerate(STATE_FIELDS):
                if len(state) > index:
                    values[name].append(state[index])
            for name, (lower, upper) in BOUNDS.items():
                index = STATE_FIELDS.index(name)
                value = number(state[index]) if len(state) > index else None
                if value is not None and value < lower:
                    invalid[name]["below"] += 1
                if value is not None and value > upper:
                    invalid[name]["above"] += 1
    for name in invalid:
        invalid[name]["below_rate"] = invalid[name]["below"] / actor_count if actor_count else None
        invalid[name]["above_rate"] = invalid[name]["above"] / actor_count if actor_count else None
        invalid[name]["outside_rate"] = (
            (invalid[name]["below"] + invalid[name]["above"]) / actor_count if actor_count else None
        )
    return {
        "events": len(rows),
        "actors": actor_count,
        "state_distributions": {name: distribution(values[name]) for name in STATE_FIELDS},
        "bound_violations": invalid,
    }


def paired_summary(c0_rows, c2_rows, method):
    c0, c2 = c0_rows.get(method, {}), c2_rows.get(method, {})
    common = sorted(set(c0) & set(c2))
    metrics = {}
    for name in PAIR_METRICS:
        deltas = []
        for index in common:
            before = number((c0[index].get("metrics") or {}).get(name))
            after = number((c2[index].get("metrics") or {}).get(name))
            if before is None or after is None:
                continue
            deltas.append(after - before)  # negative means C2 lower/better for all listed error metrics.
        eps = 1e-12
        metrics[name] = {
            "paired_count": len(deltas),
            "c2_minus_c0": distribution(deltas),
            "c2_lower_count": sum(delta < -eps for delta in deltas),
            "unchanged_count": sum(abs(delta) <= eps for delta in deltas),
            "c2_higher_count": sum(delta > eps for delta in deltas),
        }
    return {
        "method": method,
        "c0_records": len(c0),
        "c2_records": len(c2),
        "paired_records": len(common),
        "c0_only_records": sorted(set(c0) - set(c2))[:20],
        "c2_only_records": sorted(set(c2) - set(c0))[:20],
        "error_metric_deltas": metrics,
        "policy": "negative c2_minus_c0 means lower C2 value; this is descriptive, not causal",
    }


def correlation_summary(rows):
    metrics = [(row.get("metrics") or {}) for row in rows]
    report = {}
    for route_name in ROUTE_METRICS:
        report[route_name] = {}
        xs = [metric.get(route_name) for metric in metrics]
        for feas_name in FEASIBILITY_METRICS:
            ys = [metric.get(feas_name) for metric in metrics]
            pearson_value, count = pearson(xs, ys)
            spearman_value, _ = spearman(xs, ys)
            report[route_name][feas_name] = {
                "paired_count": count,
                "pearson_r": pearson_value,
                "spearman_rho": spearman_value,
            }
    return report


def report_identity(report):
    return {
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "jsonl": report.get("jsonl"),
        "jsonl_sha256": report.get("jsonl_sha256"),
        "num_records": report.get("num_records"),
        "base_seed": report.get("base_seed"),
    }


def markdown(payload):
    lines = ["# V4C C0/C2 Sampling Contract Comparison", ""]
    lines.append("This file is generated from unmodified sampling-contract rows.")
    lines.append("")
    for name in METHODS:
        pair = payload["paired_c2_minus_c0"][name]
        lines.append("## %s" % name)
        lines.append("paired records: %s" % pair["paired_records"])
        for metric in ("position_l2", "physical_invalid_rate", "interaction_route_distance_abs_error_m",
                       "interaction_route_sync_abs_gap_m", "associated_lane_corridor_failure_count"):
            item = pair["error_metric_deltas"].get(metric, {})
            delta = item.get("c2_minus_c0") or {}
            lines.append("- %s: mean C2-C0=%s; lower/unchanged/higher=%s/%s/%s" % (
                metric, delta.get("mean"), item.get("c2_lower_count"),
                item.get("unchanged_count"), item.get("c2_higher_count")))
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    paths = (args.c0_report, args.c0_rows, args.c2_report, args.c2_rows)
    missing = [path for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("missing input(s): %s" % missing)
    c0_report, c2_report = read_json(args.c0_report), read_json(args.c2_report)
    c0_all, c2_all = read_jsonl(args.c0_rows), read_jsonl(args.c2_rows)
    c0_by_method, c2_by_method = rows_by_method(c0_all), rows_by_method(c2_all)
    payload = {
        "schema_version": "v4c_sampling_contract_c0_c2_comparison_v1",
        "policy": {
            "read_only": True,
            "state_postprocessing": "none",
            "pairing": "same record_index and same diagnostic method",
            "correlation_interpretation": "descriptive association only; not causal route-loss attribution",
        },
        "inputs": {
            "c0_report": str(Path(args.c0_report)), "c0_rows": str(Path(args.c0_rows)),
            "c2_report": str(Path(args.c2_report)), "c2_rows": str(Path(args.c2_rows)),
        },
        "report_identity": {"c0": report_identity(c0_report), "c2": report_identity(c2_report)},
        "by_model_and_method": {
            "c0": {method: state_summary(list(c0_by_method.get(method, {}).values())) for method in METHODS},
            "c2": {method: state_summary(list(c2_by_method.get(method, {}).values())) for method in METHODS},
        },
        "paired_c2_minus_c0": {
            method: paired_summary(c0_by_method, c2_by_method, method) for method in METHODS
        },
        "c2_reverse_route_feasibility_correlations": correlation_summary(
            list(c2_by_method.get("reverse_diffusion_condition_only_s1", {}).values())
        ),
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_markdown:
        Path(args.output_markdown).write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
