from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any

from .p1_world_guard import load_verified_world, assert_same_episode_map

from .common import (
    ContractError,
    lane_uid_str,
    read_json,
    read_jsonl,
    require,
    sha256_bytes,
    write_json,
    write_jsonl,
    yaw_wrap,
)


# These are the exact RIFT BaseCBVRecog semantics currently in the repository.
# They are used as a read-only oracle, never as a generator fallback.
DEFAULT = {
    "search_radius_m": 60.0,
    "min_distance_m": 10.0,
    "max_interaction_fov_deg": 150.0,
    "max_interaction_dis_m": 90.0,
    "min_interaction_dis_diff_m": 20.0,
    "down_sample_step": 4,
    "max_ego_route_len": 50,
    "preview_m": 2.0,
}


def wait_for_carla_ready(
    carla: Any,
    host: str,
    port: int,
    startup_timeout: float,
    startup_rpc_timeout: float = 5.0,
    interval: float = 2.0,
) -> tuple[Any, dict[str, Any]]:
    """
    Wait for CARLA RPC/world/map readiness with a bounded startup window.

    startup_rpc_timeout limits ONE RPC attempt.
    startup_timeout limits the WHOLE readiness phase.

    A fresh Client is created for every attempt so a client created before
    the RPC endpoint becomes ready is never reused.

    This is startup synchronization only. It never repairs or retries a
    scene result.
    """
    require(startup_timeout > 0.0, "startup_timeout must be > 0")
    require(
        startup_rpc_timeout > 0.0,
        "startup_rpc_timeout must be > 0",
    )
    require(interval > 0.0, "startup_poll_interval must be > 0")

    start = time.monotonic()
    deadline = start + startup_timeout

    last_stage = "get_server_version"
    last_error = "no response"

    while time.monotonic() < deadline:
        client = carla.Client(host, port)
        client.set_timeout(startup_rpc_timeout)

        try:
            last_stage = "get_server_version"
            server_version = client.get_server_version()

            last_stage = "get_world"
            world = client.get_world()

            last_stage = "get_map"
            map_name = world.get_map().name

            client_version = client.get_client_version()

            return client, {
                "host": host,
                "port": int(port),
                "server_version": str(server_version),
                "client_version": str(client_version),
                "map_name": str(map_name),
                "startup_wait_seconds": (
                    time.monotonic() - start
                ),
                "startup_rpc_timeout_seconds": float(
                    startup_rpc_timeout
                ),
                "startup_poll_interval_seconds": float(interval),
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        elapsed = time.monotonic() - start

        print(
            "Waiting for CARLA RPC "
            f"({elapsed:.0f}s), "
            f"stage={last_stage}: {last_error}",
            flush=True,
        )

        remaining = deadline - time.monotonic()

        if remaining <= 0.0:
            break

        time.sleep(min(interval, remaining))

    raise TimeoutError(
        "CARLA did not become ready within "
        f"{startup_timeout:.0f}s; "
        f"stage={last_stage}; "
        f"last_error={last_error}"
    )


def rift_parity_status(
    parity: dict[str, Any],
    interaction: dict[str, Any],
) -> tuple[str, bool]:
    """
    Strict parity state.

    unavailable / unknown / error can NEVER become PASS.
    """
    status = parity.get("status")

    if status == "unavailable":
        return "unavailable", False

    if status == "no_preview":
        passed = (
            interaction.get("failure")
            == "no_preview_waypoint"
        )
        return ("pass" if passed else "fail"), passed

    if status != "pass":
        return "unknown", False

    # Our independent implementation must have completed.
    if interaction.get("status") != "complete":
        return "unknown", False

    same_native = (
        bool(parity.get("native_interaction"))
        == bool(interaction.get("native_interaction"))
    )

    same_route_index = (
        parity.get("selected_ego_route_index")
        == interaction.get("selected_ego_route_index")
    )

    passed = same_native and same_route_index

    return ("pass" if passed else "fail"), passed


def native_truth_status(
    candidate: dict[str, Any],
    interaction: dict[str, Any],
) -> tuple[str, bool | None]:
    """
    Incomplete native evaluation is UNKNOWN, never NEGATIVE.

    Returns:
        ("pass", True)
        ("fail", False)
        ("unknown", None)
    """
    if interaction.get("status") != "complete":
        return "unknown", None

    passed = bool(
        candidate["candidate_filter_pass"]
        and interaction["native_interaction"]
    )

    return ("pass" if passed else "fail"), passed


def apply_rift_config(
    path: str | None,
    mode: str = "eval",
) -> dict[str, float]:
    if not path:
        return dict(DEFAULT)

    import yaml

    cfg = yaml.safe_load(
        Path(path).read_text(encoding="utf-8")
    )

    require(
        isinstance(cfg, dict) and mode in cfg,
        f"RIFT config missing mode={mode}: {path}",
    )

    m = cfg[mode]

    mapping = {
        "search_radius": "search_radius_m",
        "max_interaction_fov": "max_interaction_fov_deg",
        "max_interaction_dis": "max_interaction_dis_m",
        "min_interaction_dis_diff": "min_interaction_dis_diff_m",
        "max_ego_route_len": "max_ego_route_len",
        "down_sample_step": "down_sample_step",
    }

    for src, dst in mapping.items():
        require(
            src in m,
            f"RIFT config {path}:{mode} missing {src}",
        )

        DEFAULT[dst] = (
            float(m[src])
            if dst.endswith("_m")
            or dst.endswith("_deg")
            else int(m[src])
        )

    return dict(DEFAULT)


def add_python_paths(
    rift_root: str,
    carla_pythonapi: str | None,
) -> None:
    """
    Add known CARLA/RIFT source roots.

    Supports both:
        <CARLA>/PythonAPI
    and:
        <CARLA>/PythonAPI/carla

    This removes dependence on a manually pre-populated PYTHONPATH for
    the known CARLA agents source layout.
    """
    paths: list[Path] = []

    if rift_root:
        paths.append(Path(rift_root).resolve())

    if carla_pythonapi:
        p = Path(carla_pythonapi).resolve()
        paths.append(p)

        if p.name == "carla":
            paths.append(p.parent)
        elif (p / "carla").is_dir():
            paths.append(p / "carla")

    seen = set()

    for p in paths:
        s = str(p)

        if s in seen:
            continue

        seen.add(s)
        sys.path.insert(0, s)


def import_carla_and_grp(
    rift_root: str,
    carla_pythonapi: str | None,
):
    add_python_paths(
        rift_root,
        carla_pythonapi,
    )

    import carla

    from rift.scenario.tools.global_route_planner import (
        GlobalRoutePlanner,
    )

    return carla, GlobalRoutePlanner


def normalize_map_name(name: str) -> str:
    name = (
        str(name)
        .replace("\\", "/")
        .rstrip("/")
    )

    return name.split("/")[-1]


def vec2(loc: Any) -> tuple[float, float]:
    return float(loc.x), float(loc.y)


def distance2(a: Any, b: Any) -> float:
    return math.hypot(
        float(a.x) - float(b.x),
        float(a.y) - float(b.y),
    )


def is_within_distance_ahead(
    target_transform: Any,
    ref_transform: Any,
    max_distance: float,
    max_fov_deg: float,
) -> bool:
    # Same mathematical predicate as RIFT's imported helper.
    target = target_transform.location
    ref = ref_transform.location

    dx = float(target.x) - float(ref.x)
    dy = float(target.y) - float(ref.y)

    norm = math.hypot(dx, dy)

    if norm < 1e-6:
        return True

    if norm > max_distance:
        return False

    fx = math.cos(
        math.radians(
            float(ref_transform.rotation.yaw)
        )
    )

    fy = math.sin(
        math.radians(
            float(ref_transform.rotation.yaw)
        )
    )

    c = max(
        -1.0,
        min(
            1.0,
            (fx * dx + fy * dy) / norm,
        ),
    )

    return (
        math.degrees(math.acos(c))
        < max_fov_deg
    )


def carla_transform(
    carla: Any,
    row: dict[str, Any],
) -> Any:
    return carla.Transform(
        carla.Location(
            float(row["x"]),
            float(row["y"]),
            float(row["z"]),
        ),
        carla.Rotation(
            pitch=float(
                row.get("pitch_deg", 0.0)
            ),
            yaw=float(row["yaw_deg"]),
            roll=float(
                row.get("roll_deg", 0.0)
            ),
        ),
    )


def lane_match(
    a: Any,
    b: Any,
    s_tol: float = 0.05,
    pos_tol: float = 0.30,
) -> dict[str, Any]:
    pos = distance2(
        a.transform.location,
        b.transform.location,
    )

    return {
        "road_id_equal": (
            int(a.road_id)
            == int(b.road_id)
        ),
        "section_id_equal": (
            int(a.section_id)
            == int(b.section_id)
        ),
        "lane_id_equal": (
            int(a.lane_id)
            == int(b.lane_id)
        ),
        "s_abs_error_m": abs(
            float(a.s) - float(b.s)
        ),
        "position_error_m": pos,
        "pass": (
            int(a.road_id)
            == int(b.road_id)
            and int(a.section_id)
            == int(b.section_id)
            and int(a.lane_id)
            == int(b.lane_id)
            and abs(
                float(a.s) - float(b.s)
            ) <= s_tol
            and pos <= pos_tol
        ),
    }


# P1_ROUTE_CARLA_SCOPE_FIX_V1
def route_waypoints(
    carla: Any,
    carla_map: Any,
    scene: dict[str, Any],
) -> tuple[
    list[Any],
    list[dict[str, Any]],
]:
    rows = scene.get("ego_route", [])

    # Read road lengths from the same runtime OpenDRIVE used by CARLA. This
    # is used only to classify an exact road-end XODR lookup miss.
    import xml.etree.ElementTree as _p1_ET

    _p1_road_lengths = {}
    _p1_opendrive_error = None
    try:
        _p1_root = _p1_ET.fromstring(carla_map.to_opendrive())
        for _p1_road in _p1_root.findall("road"):
            _p1_road_lengths[int(_p1_road.attrib["id"])] = float(
                _p1_road.attrib["length"]
            )
    except Exception as _p1_exc:
        _p1_opendrive_error = repr(_p1_exc)

    require(
        rows,
        "P1/P2 require ego_route; "
        "do not reconstruct it from the target",
    )

    out = []
    checks = []

    for i, r in enumerate(
        rows[: int(DEFAULT["max_ego_route_len"])]
    ):
        for k in (
            "x",
            "y",
            "z",
            "yaw_deg",
            "road_id",
            "section_id",
            "lane_id",
            "s",
        ):
            require(
                k in r,
                f"ego_route[{i}] missing {k}",
            )

        xodr_wp = carla_map.get_waypoint_xodr(
            int(r["road_id"]),
            int(r["lane_id"]),
            float(r["s"]),
        )

        # Construct the serialized transform before classifying an XODR miss.
        # Do not modify r["s"] or substitute a nearby waypoint.
        expected = carla.Transform(
            carla.Location(
                float(r["x"]),
                float(r["y"]),
                float(r["z"]),
            ),
            carla.Rotation(
                yaw=float(r["yaw_deg"])
            ),
        )

        # Exact serialized-location lookup.
        # This is validation, not repair.
        exact_wp = carla_map.get_waypoint(
            expected.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )

        require(
            exact_wp is not None,
            "ego_route[%d] exact get_waypoint "
            "returned None" % i,
        )

        require(
            (
                int(exact_wp.road_id)
                == int(r["road_id"])
                and int(exact_wp.section_id)
                == int(r["section_id"])
                and int(exact_wp.lane_id)
                == int(r["lane_id"])
            ),
            (
                f"ego_route[{i}] exact lane "
                "identity mismatch"
            ),
        )

        # Classify only the proven road-end boundary case. The map lookup
        # remains visibly failed in the audit output; this is not an XODR pass.
        xodr_road_length_m = _p1_road_lengths.get(int(r["road_id"]))
        xodr_boundary_verified = False
        xodr_endpoint_delta_m = None
        exact_s_error_m = abs(float(exact_wp.s) - float(r["s"]))
        exact_position_error_m = exact_wp.transform.location.distance(
            expected.location
        )
        exact_yaw_error_deg = abs((
            float(exact_wp.transform.rotation.yaw)
            - float(expected.rotation.yaw) + 180.0
        ) % 360.0 - 180.0)

        if xodr_wp is None:
            if xodr_road_length_m is not None:
                xodr_endpoint_delta_m = (
                    float(r["s"]) - float(xodr_road_length_m)
                )

            require(
                xodr_road_length_m is not None
                and abs(xodr_endpoint_delta_m) <= 1e-4
                and exact_s_error_m <= 1e-4
                and exact_position_error_m <= 0.05
                and exact_yaw_error_deg <= 0.5,
                (
                    f"ego_route[{i}] get_waypoint_xodr returned None; "
                    "not a verified road-end boundary: "
                    f"road_length={xodr_road_length_m}, "
                    f"s_minus_road_end={xodr_endpoint_delta_m}, "
                    f"exact_s_error={exact_s_error_m}, "
                    f"exact_position_error_m={exact_position_error_m}, "
                    f"exact_yaw_error_deg={exact_yaw_error_deg}"
                ),
            )
            xodr_boundary_verified = True

        wp = exact_wp

        check = lane_match(
            wp,
            wp,
            s_tol=0.0,
            pos_tol=0.0,
        )

        check["route_index"] = i

        check["xodr_found"] = xodr_wp is not None
        check["xodr_lookup_status"] = (
            "ok" if xodr_wp is not None
            else "verified_road_end_boundary"
        )
        check["xodr_endpoint_exception_verified"] = xodr_boundary_verified
        check["xodr_road_length_m"] = xodr_road_length_m
        check["xodr_s_minus_road_end_m"] = xodr_endpoint_delta_m
        check["xodr_exact_s_error_m"] = exact_s_error_m
        check["xodr_exact_position_error_m"] = exact_position_error_m
        check["xodr_exact_yaw_error_deg"] = exact_yaw_error_deg
        check["xodr_opendrive_parse_error"] = _p1_opendrive_error

        check["expected_road_id"] = int(
            r["road_id"]
        )
        check["expected_section_id"] = int(
            r["section_id"]
        )
        check["expected_lane_id"] = int(
            r["lane_id"]
        )

        check["actual_road_id"] = int(
            wp.road_id
        )
        check["actual_section_id"] = int(
            wp.section_id
        )
        check["actual_lane_id"] = int(
            wp.lane_id
        )

        check["s_abs_error_m"] = abs(
            float(wp.s) - float(r["s"])
        )

        check["position_error_m"] = distance2(
            wp.transform.location,
            expected.location,
        )

        check["yaw_error_deg"] = abs(
            (
                (
                    float(
                        wp.transform.rotation.yaw
                    )
                    - float(r["yaw_deg"])
                    + 180.0
                )
                % 360.0
            )
            - 180.0
        )

        if xodr_wp is not None:
            check[
                "xodr_exact_position_error_m"
            ] = distance2(
                xodr_wp.transform.location,
                exact_wp.transform.location,
            )
        else:
            # XODR lookup failed; keep the comparison explicitly unavailable.
            check["xodr_exact_position_error_m"] = None

        if xodr_wp is not None:
            check[
                "xodr_exact_yaw_error_deg"
            ] = abs(
                (
                    (
                        float(
                            xodr_wp.transform.rotation.yaw
                        )
                        - float(
                            exact_wp.transform.rotation.yaw
                        )
                        + 180.0
                    )
                    % 360.0
                )
                - 180.0
            )
        else:
            # XODR lookup failed; keep the comparison explicitly unavailable.
            check["xodr_exact_yaw_error_deg"] = None

        # P1_ROUTE140_XODR_SOURCE_ANCHOR_V1
        # Preserve the independent exact-location probe, including its measured
        # s deviation. A CARLA location lookup is not the source waypoint.
        check["exact_location_s_abs_error_m"] = check["s_abs_error_m"]
        check["exact_location_xyz_error_m"] = float(
            exact_wp.transform.location.distance(expected.location)
        )
        check["exact_location_s_over_005"] = bool(
            check["s_abs_error_m"] > 0.05
        )

        if xodr_wp is not None:
            check["xodr_source_road_equal"] = (
                int(xodr_wp.road_id) == int(r["road_id"])
            )
            check["xodr_source_section_equal"] = (
                int(xodr_wp.section_id) == int(r["section_id"])
            )
            check["xodr_source_lane_equal"] = (
                int(xodr_wp.lane_id) == int(r["lane_id"])
            )
            check["xodr_source_s_abs_error_m"] = abs(
                float(xodr_wp.s) - float(r["s"])
            )
            check["xodr_source_xyz_error_m"] = float(
                xodr_wp.transform.location.distance(expected.location)
            )
            check["xodr_source_yaw_error_deg"] = abs(
                (float(xodr_wp.transform.rotation.yaw)
                 - float(r["yaw_deg"]) + 180.0) % 360.0 - 180.0
            )
            xodr_source_anchor_pass = bool(
                check["xodr_source_road_equal"]
                and check["xodr_source_section_equal"]
                and check["xodr_source_lane_equal"]
                and check["xodr_source_s_abs_error_m"] <= 1e-4
                and check["xodr_source_xyz_error_m"] <= 0.05
                and check["xodr_source_yaw_error_deg"] <= 0.5
            )
        else:
            # None remains None; only the earlier verified endpoint may pass.
            check["xodr_source_road_equal"] = None
            check["xodr_source_section_equal"] = None
            check["xodr_source_lane_equal"] = None
            check["xodr_source_s_abs_error_m"] = None
            check["xodr_source_xyz_error_m"] = None
            check["xodr_source_yaw_error_deg"] = None
            xodr_source_anchor_pass = False

        check["xodr_source_anchor_pass"] = bool(xodr_source_anchor_pass)
        check["route_reference_kind"] = (
            "xodr_original_s" if xodr_wp is not None
            else "verified_road_end_exact"
        )
        # Exact-location query is a lane-membership/cross-check only; its s
        # is not substituted for the serialized historical route's s.
        exact_location_membership_pass = bool(
            int(exact_wp.road_id) == int(r["road_id"])
            and int(exact_wp.section_id) == int(r["section_id"])
            and int(exact_wp.lane_id) == int(r["lane_id"])
            and check["position_error_m"] <= 0.30
            and check["exact_location_xyz_error_m"] <= 0.30
            and check["yaw_error_deg"] <= 2.0
        )
        check["exact_location_membership_pass"] = exact_location_membership_pass
        route_identity_pass = bool(
            exact_location_membership_pass
            and (
                xodr_source_anchor_pass if xodr_wp is not None
                else xodr_boundary_verified
            )
        )

        check["route_identity_pass"] = route_identity_pass

        if xodr_wp is not None:
            xodr_position_error = check["xodr_exact_position_error_m"]
            xodr_yaw_error = check["xodr_exact_yaw_error_deg"]
            xodr_comparison_pass = (
                xodr_position_error is not None
                and xodr_position_error <= 0.30
                and xodr_yaw_error is not None
                and xodr_yaw_error <= 2.0
            )
            check["xodr_exact_comparison_status"] = (
                "pass" if xodr_comparison_pass else "fail"
            )
        else:
            xodr_comparison_pass = bool(xodr_boundary_verified)
            check["xodr_exact_comparison_status"] = (
                "not_evaluable_verified_endpoint_exception"
                if xodr_boundary_verified else "unknown"
            )

        check["pass"] = bool(route_identity_pass and xodr_comparison_pass)

        checks.append(check)

        require(
            check["pass"],
            f"ego_route[{i}] CARLA identity "
            f"mismatch: {check}",
        )

        # Preserve the original serialized route geometry in native replay.
        # No snap, projection, GT adjustment or candidate substitution.
        out.append(xodr_wp if xodr_wp is not None else exact_wp)

    require(
        len(out) >= 2,
        "ego_route must contain at least "
        "2 validated waypoints",
    )

    return out, checks


def cumulative_route(
    route: list[Any],
) -> dict[int, float]:
    idxs = list(
        range(
            0,
            len(route),
            int(DEFAULT["down_sample_step"]),
        )
    )

    if idxs[-1] != len(route) - 1:
        idxs.append(len(route) - 1)

    total = 0.0
    out = {}

    prev = route[
        idxs[0]
    ].transform.location

    for idx in idxs[1:]:
        cur = route[idx].transform.location

        total += float(
            prev.distance(cur)
        )

        out[idx] = total
        prev = cur

    return out


def preview_waypoint(
    carla: Any,
    carla_map: Any,
    transform: Any,
):
    fwd = (
        transform
        .get_forward_vector()
        .make_unit_vector()
    )

    p = transform.location

    preview = carla.Location(
        x=p.x
        + fwd.x * DEFAULT["preview_m"],
        y=p.y
        + fwd.y * DEFAULT["preview_m"],
        z=p.z
        + fwd.z * DEFAULT["preview_m"],
    )

    # Intentionally mirrors RIFT:
    # project_to_road=None.
    # Do not change to True.
    return carla_map.get_waypoint(
        preview,
        project_to_road=None,
        lane_type=carla.LaneType.Driving,
    )


def exact_interaction(
    carla: Any,
    grp: Any,
    carla_map: Any,
    route: list[Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    tr = carla_transform(
        carla,
        actor,
    )

    pwp = preview_waypoint(
        carla,
        carla_map,
        tr,
    )

    result = {
        "actor_id": actor.get("actor_id"),
        "status": "complete",
        "preview": None,
        "targets_ahead": 0,
        "best_distance_diff_m": None,
        "selected_ego_route_index": None,
        "selected_ego_distance_m": None,
        "selected_cbv_distance_m": None,
        "native_interaction": False,
        "trace_route_errors": [],
    }

    if pwp is None:
        result["failure"] = (
            "no_preview_waypoint"
        )
        return result

    result["preview"] = {
        "road_id": int(pwp.road_id),
        "section_id": int(
            pwp.section_id
        ),
        "lane_id": int(pwp.lane_id),
        "s": float(pwp.s),
        "x": float(
            pwp.transform.location.x
        ),
        "y": float(
            pwp.transform.location.y
        ),
        "z": float(
            pwp.transform.location.z
        ),
        "yaw_deg": float(
            pwp.transform.rotation.yaw
        ),
        "lane_width_m": float(
            pwp.lane_width
        ),
        "is_junction": bool(
            pwp.is_junction
        ),
    }

    ego_dist = cumulative_route(route)

    temp = DEFAULT[
        "min_interaction_dis_diff_m"
    ]

    for idx, ed in ego_dist.items():
        target = route[idx]

        if not is_within_distance_ahead(
            target.transform,
            pwp.transform,
            DEFAULT[
                "max_interaction_dis_m"
            ],
            DEFAULT[
                "max_interaction_fov_deg"
            ] / 2.0,
        ):
            continue

        result["targets_ahead"] += 1

        try:
            (
                trace,
                route_ids,
                cbv_distance,
            ) = grp.trace_route(
                pwp.transform.location,
                target.transform.location,
            )

        except Exception as e:
            # IMPORTANT:
            # Do not silently turn planner failure into
            # "no native interaction".
            result["status"] = "error"

            result[
                "trace_route_errors"
            ].append(
                {
                    "route_index": int(idx),
                    "error": (
                        f"{type(e).__name__}: {e}"
                    ),
                }
            )

            continue

        diff = abs(
            float(cbv_distance)
            - float(ed)
        )

        if (
            result[
                "best_distance_diff_m"
            ]
            is None
            or diff
            < result[
                "best_distance_diff_m"
            ]
        ):
            result[
                "best_distance_diff_m"
            ] = float(diff)

        if diff < temp:
            temp = diff

            result.update(
                {
                    "selected_ego_route_index": int(
                        idx
                    ),
                    "selected_ego_distance_m": float(
                        ed
                    ),
                    "selected_cbv_distance_m": float(
                        cbv_distance
                    ),
                    "native_interaction": True,
                    "selected_cbv_route_ids": route_ids,
                    "selected_cbv_route_lane_uids": [
                        [
                            int(w.road_id),
                            int(w.section_id),
                            int(w.lane_id),
                        ]
                        for w, _ in trace
                    ],
                }
            )

    if result["status"] == "error":
        result["failure"] = (
            "trace_route_error"
        )

    elif not result["native_interaction"]:
        result["failure"] = (
            "no_native_interaction"
        )

    return result


class _NoopRoutePlanner:
    def update_inter_info(
        self,
        *args,
        **kwargs,
    ):
        return None


def rift_reference_interaction(
    carla: Any,
    grp: Any,
    carla_map: Any,
    route: list[Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the repository's current BaseCBVRecog method
    as an independent parity oracle.
    """
    try:
        from rift.cbv.recognition.base_cbv import (
            BaseCBVRecog,
        )
    except Exception as e:
        return {
            "status": "unavailable",
            "reason": (
                "rift_import:"
                f"{type(e).__name__}:{e}"
            ),
        }

    tr = carla_transform(
        carla,
        actor,
    )

    pwp = preview_waypoint(
        carla,
        carla_map,
        tr,
    )

    if pwp is None:
        return {
            "status": "no_preview",
            "native_interaction": False,
        }

    ref = BaseCBVRecog(
        {"desired_speed": 0.0},
        logger=None,
    )

    ref._map = carla_map
    ref.global_route_planner = grp
    ref.route_planner = _NoopRoutePlanner()

    ref.max_route_len = DEFAULT[
        "max_ego_route_len"
    ]

    ref.max_interaction_dis = DEFAULT[
        "max_interaction_dis_m"
    ]

    ref.max_interaction_fov = DEFAULT[
        "max_interaction_fov_deg"
    ]

    ref.min_interaction_dis_diff = DEFAULT[
        "min_interaction_dis_diff_m"
    ]

    ego_dist = cumulative_route(route)

    try:
        wp = ref.find_interaction_waypoint(
            route,
            pwp,
            actor.get("actor_id", -1),
            ego_dist,
        )

        dbg = ref.last_interaction_debug.get(
            int(actor.get("actor_id", -1)),
            {},
        )

    except Exception as e:
        return {
            "status": "error",
            "reason": (
                "rift_runtime:"
                f"{type(e).__name__}:{e}"
            ),
        }

    return {
        "status": "pass",
        "native_interaction": wp is not None,
        "selected_ego_route_index": dbg.get(
            "selected_ego_route_index"
        ),
        "best_distance_diff_m": dbg.get(
            "best_distance_diff"
        ),
    }


def candidate_semantics(
    carla: Any,
    carla_map: Any,
    actor: dict[str, Any],
) -> dict[str, Any]:
    tr = carla_transform(
        carla,
        actor,
    )

    ego_loc = actor["_ego_location"]

    dist = float(
        ego_loc.distance(
            tr.location
        )
    )

    pwp = preview_waypoint(
        carla,
        carla_map,
        tr,
    )

    out = {
        "distance_m": dist,
        "distance_pass": (
            DEFAULT["min_distance_m"]
            < dist
            <= DEFAULT["search_radius_m"]
        ),
        "preview_exists": (
            pwp is not None
        ),
        "junction_heading_pass": True,
    }

    if (
        pwp is not None
        and pwp.is_junction
    ):
        a = tr.get_forward_vector()
        b = (
            pwp.transform
            .get_forward_vector()
        )

        out[
            "junction_heading_error_deg"
        ] = float(
            a.get_vector_angle(b)
        )

        out[
            "junction_heading_pass"
        ] = (
            out[
                "junction_heading_error_deg"
            ]
            <= 20.0
        )

    out["candidate_filter_pass"] = (
        out["distance_pass"]
        and out["preview_exists"]
        and out["junction_heading_pass"]
    )

    return out


def coord_roundtrip(
    carla: Any,
    ego: dict[str, Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    ego_tr = carla_transform(
        carla,
        ego,
    )

    a_tr = carla_transform(
        carla,
        actor,
    )

    p = a_tr.location
    e = ego_tr.location

    yaw = math.radians(
        float(
            ego_tr.rotation.yaw
        )
    )

    dx = float(p.x - e.x)
    dy = float(p.y - e.y)
    dz = float(p.z - e.z)

    # CARLA world is left-handed;
    # this is the v7 contract x-forward, y-right.
    c = math.cos(yaw)
    s = math.sin(yaw)

    lx = c * dx + s * dy
    ly = -s * dx + c * dy

    bx = (
        c * lx
        - s * ly
        + e.x
    )

    by = (
        s * lx
        + c * ly
        + e.y
    )

    actor_yaw = math.radians(
        float(
            a_tr.rotation.yaw
        )
    )

    local_yaw = yaw_wrap(
        actor_yaw - yaw
    )

    world_yaw = yaw_wrap(
        local_yaw + yaw
    )

    return {
        "local_x_m": lx,
        "local_y_m": ly,
        "roundtrip_position_error_m": math.hypot(
            bx - p.x,
            by - p.y,
        ),
        "roundtrip_z_error_m": abs(
            (dz + e.z) - p.z
        ),
        "local_yaw_rad": local_yaw,
        "roundtrip_yaw_error_rad": abs(
            yaw_wrap(
                world_yaw
                - actor_yaw
            )
        ),
    }


def blueprint_measurement(
    world: Any,
    carla: Any,
) -> dict[str, Any]:
    lib = world.get_blueprint_library()

    try:
        bp = lib.find(
            "vehicle.tesla.model3"
        )
    except Exception as e:
        return {
            "status": "fail",
            "reason": (
                "blueprint_not_found:"
                f"{e}"
            ),
        }

    sp = (
        world
        .get_map()
        .get_spawn_points()
    )

    require(
        sp,
        "CARLA map has no "
        "vehicle spawn points",
    )

    # Exactly one official spawn point,
    # exactly one attempt. No retry.
    actor = world.try_spawn_actor(
        bp,
        sp[0],
    )

    if actor is None:
        return {
            "status": "fail",
            "reason": (
                "blueprint_probe_spawn_rejected"
            ),
            "attempted_spawn_index": 0,
        }

    try:
        bb = actor.bounding_box

        return {
            "status": "pass",
            "type_id": actor.type_id,
            "extent_xyz_m": [
                float(bb.extent.x),
                float(bb.extent.y),
                float(bb.extent.z),
            ],
            "box_size_lwh_m": [
                2 * float(bb.extent.x),
                2 * float(bb.extent.y),
                2 * float(bb.extent.z),
            ],
            "offset_xyz_m": [
                float(bb.location.x),
                float(bb.location.y),
                float(bb.location.z),
            ],
        }

    finally:
        actor.destroy()


def raw_sedan_spawn(
    world: Any,
    carla: Any,
    actor: dict[str, Any],
) -> dict[str, Any]:
    # P1_RAW_POSE_SNAPSHOT_FIX_V1
    # A non-None try_spawn_actor result only means CARLA returned an actor
    # proxy. Pose acceptance requires a post-spawn transform matching the
    # independently saved requested pose.
    bp = (
        world
        .get_blueprint_library()
        .find("vehicle.tesla.model3")
    )

    tr = carla_transform(
        carla,
        actor,
    )

    actor_id = actor.get("actor_id")

    # Save independent scalar values before the RPC. Never calculate errors
    # from tr after try_spawn_actor(); use this immutable numeric snapshot.
    expected = {
        "x": float(tr.location.x),
        "y": float(tr.location.y),
        "z": float(tr.location.z),
        "roll": float(tr.rotation.roll),
        "pitch": float(tr.rotation.pitch),
        "yaw": float(tr.rotation.yaw),
    }

    spawned = world.try_spawn_actor(
        bp,
        tr,
    )

    if spawned is None:
        return {
            "actor_id": actor_id,
            "status": "rejected",
            "spawn_rpc_accepted": False,
            "pose_validation_status": "not_spawned",
            "raw_pose_correction": False,
            "expected_pose_before_spawn": expected,
            "actual_pose": None,
            "position_error_m": None,
            "yaw_error_deg": None,
            "reason": "CARLA_rejected_raw_pose",
        }

    def pose_dict(transform: Any) -> dict[str, float]:
        return {
            "x": float(transform.location.x),
            "y": float(transform.location.y),
            "z": float(transform.location.z),
            "roll": float(transform.rotation.roll),
            "pitch": float(transform.rotation.pitch),
            "yaw": float(transform.rotation.yaw),
        }

    def pose_is_zero(pose: Any) -> bool:
        return (
            pose is not None
            and abs(pose["x"]) < 1e-6
            and abs(pose["y"]) < 1e-6
            and abs(pose["z"]) < 1e-6
        )

    expected_is_zero = (
        abs(expected["x"]) < 1e-6
        and abs(expected["y"]) < 1e-6
        and abs(expected["z"]) < 1e-6
    )

    snapshot_pose = None
    actor_pose = None
    actual_pose = None
    actual_pose_source = None
    snapshot_frame = None
    wait_errors = []
    wait_attempts = 0
    immediate_spawned_pose = None
    immediate_world_actor_pose = None

    # Preserve the immediate readings too. These are diagnostic only; the
    # validated result below is based on the post-spawn observation.
    try:
        immediate_spawned_pose = pose_dict(spawned.get_transform())
    except Exception as exc:
        wait_errors.append(
            "immediate spawned.get_transform: %s: %s"
            % (type(exc).__name__, str(exc))
        )
    try:
        immediate_world_actor = world.get_actor(spawned.id)
        if immediate_world_actor is not None:
            immediate_world_actor_pose = pose_dict(
                immediate_world_actor.get_transform()
            )
    except Exception as exc:
        wait_errors.append(
            "immediate world.get_actor/get_transform: %s: %s"
            % (type(exc).__name__, str(exc))
        )

    try:
        # Do not call world.tick(): P1 must not advance the simulation itself.
        # Wait for a server-produced snapshot. A zero transform for a
        # non-origin requested pose is treated as unavailable, not as success.
        for attempt in range(5):
            wait_attempts = attempt + 1
            snapshot = None

            try:
                snapshot = world.wait_for_tick(1.0)
                snapshot_frame = getattr(snapshot, "frame", None)
            except Exception as exc:
                wait_errors.append(
                    "wait_for_tick[%d]: %s: %s"
                    % (attempt + 1, type(exc).__name__, str(exc))
                )

            snapshot_actor = None
            if snapshot is not None:
                try:
                    snapshot_actor = snapshot.find(spawned.id)
                except Exception as exc:
                    wait_errors.append(
                        "snapshot.find[%d]: %s: %s"
                        % (attempt + 1, type(exc).__name__, str(exc))
                    )

            snapshot_transform = None
            if snapshot_actor is not None:
                try:
                    snapshot_transform = snapshot_actor.get_transform()
                    snapshot_pose = pose_dict(snapshot_transform)
                except Exception as exc:
                    wait_errors.append(
                        "snapshot_transform[%d]: %s: %s"
                        % (attempt + 1, type(exc).__name__, str(exc))
                    )

            current_actor = None
            try:
                current_actor = world.get_actor(spawned.id)
            except Exception as exc:
                wait_errors.append(
                    "world.get_actor[%d]: %s: %s"
                    % (attempt + 1, type(exc).__name__, str(exc))
                )

            if current_actor is not None:
                try:
                    actor_pose = pose_dict(current_actor.get_transform())
                except Exception as exc:
                    wait_errors.append(
                        "actor.get_transform[%d]: %s: %s"
                        % (attempt + 1, type(exc).__name__, str(exc))
                    )

            # Prefer a non-zero snapshot pose, then a non-zero live actor pose.
            # If the expected pose itself is at the origin, zero can be valid.
            if snapshot_pose is not None and (
                not pose_is_zero(snapshot_pose) or expected_is_zero
            ):
                actual_pose = snapshot_pose
                actual_pose_source = "world_snapshot"
            elif actor_pose is not None and (
                not pose_is_zero(actor_pose) or expected_is_zero
            ):
                actual_pose = actor_pose
                actual_pose_source = "world_actor"
            elif snapshot_pose is not None:
                actual_pose = snapshot_pose
                actual_pose_source = "world_snapshot_zero_suspect"
            elif actor_pose is not None:
                actual_pose = actor_pose
                actual_pose_source = "world_actor_zero_suspect"

            if actual_pose is not None and (
                not pose_is_zero(actual_pose) or expected_is_zero
            ):
                break

        position_error = None
        yaw_error = None
        pose_status = "unknown"
        status = "pose_unverified"
        reason = None

        if actual_pose is None:
            reason = "no_post_spawn_transform_observed"
        else:
            position_error = math.hypot(
                actual_pose["x"] - expected["x"],
                actual_pose["y"] - expected["y"],
            )
            yaw_error = abs(
                (
                    actual_pose["yaw"] - expected["yaw"] + 180.0
                ) % 360.0 - 180.0
            )

            zero_sentinel = (
                pose_is_zero(actual_pose) and not expected_is_zero
            )

            if zero_sentinel:
                pose_status = "unknown"
                status = "pose_unverified"
                reason = (
                    "all_zero_actual_pose_after_snapshot_wait; "
                    "not counted as accepted"
                )
            elif position_error <= 0.30 and yaw_error <= 2.0:
                pose_status = "pass"
                status = "accepted"
                reason = None
            else:
                pose_status = "fail"
                status = "pose_mismatch"
                reason = "post_spawn_pose_outside_tolerance"

        return {
            "actor_id": actor_id,
            "spawned_actor_id": int(spawned.id),
            "spawned_type_id": str(spawned.type_id),
            "status": status,
            "spawn_rpc_accepted": True,
            "pose_validation_status": pose_status,
            "raw_pose_correction": False,
            "expected_pose_before_spawn": expected,
            "immediate_spawned_actor_pose": immediate_spawned_pose,
            "immediate_world_actor_pose": immediate_world_actor_pose,
            "actual_pose": actual_pose,
            "actual_pose_source": actual_pose_source,
            "snapshot_pose": snapshot_pose,
            "live_actor_pose": actor_pose,
            "world_snapshot_frame": snapshot_frame,
            "snapshot_wait_attempts": wait_attempts,
            "snapshot_wait_errors": wait_errors,
            "position_error_m": position_error,
            "yaw_error_deg": yaw_error,
            "position_tolerance_m": 0.30,
            "yaw_tolerance_deg": 2.0,
            "reason": reason,
        }

    finally:
        spawned.destroy()

def scene_occupancy_spawn(
    client: Any,
    world: Any,
    carla: Any,
    scene: dict[str, Any],
    target_actors: list[dict[str, Any]],
) -> dict[str, Any]:
    # P1_OCCUPANCY_AGENTS_JOINT_V1
    # Diagnostically replay exact snapshot states. NO pose correction,
    # retry, transform projection, world.tick(), or native-label fallback.
    # Never treat 'complete' as physical scene or Model3 feasibility PASS.
    result = {
        'schema_version': 'p1_occupancy_agents_joint_v1',
        'status': 'unknown',
        'reason': None,
        'source': 'snapshot.agents + snapshot.rift_labels',
        'background': [], 'targets': [],
        'expected_background_actor_ids': [],
        'expected_target_actor_ids': [],
        'expected_background_count': None,
        'expected_target_count': None,
        'joint_model3_spawn_accepted': None,
        'full_scene_physical_feasibility': 'unknown',
        'source_pose_modified': False,
        'world_tick_called': False,
        'cleanup_verified': None,
    }
    created_ids = []
    ownership_unknown = False
    def fail(reason):
        result['status'] = 'unknown'
        result['reason'] = reason
        return result

    # Validate original source completeness BEFORE spawning anything.
    if not isinstance(scene, dict) or not isinstance(scene.get('ego'), dict):
        return fail('ego_missing_or_invalid')
    ego = scene['ego']
    if not str(ego.get('type_id', '')).startswith('vehicle.'):
        return fail('ego_vehicle_blueprint_missing')
    agents = scene.get('agents')
    labels = scene.get('rift_labels')
    if not isinstance(agents, list) or not isinstance(labels, list):
        return fail('agents_or_rift_labels_missing')
    if not agents or not labels or any(not isinstance(a, dict) for a in agents) or any(not isinstance(l, dict) for l in labels):
        return fail('agents_or_labels_invalid_or_empty')
    if any(a.get('actor_id') is None or not str(a.get('type_id', '')).startswith('vehicle.') for a in agents):
        return fail('agent_id_or_vehicle_type_invalid')
    if any(l.get('actor_id') is None or not isinstance(l.get('is_interactive'), bool) for l in labels):
        return fail('rift_label_id_or_boolean_invalid')
    ids = [str(a['actor_id']) for a in agents]
    label_ids = [str(l['actor_id']) for l in labels]
    if len(ids) != len(set(ids)) or len(label_ids) != len(set(label_ids)) or set(ids) != set(label_ids):
        return fail('agents_labels_identity_conflict')
    positive_ids = {str(l['actor_id']) for l in labels if l['is_interactive']}
    received_ids = [str(t.get('actor_id')) for t in target_actors]
    if (not positive_ids or len(received_ids) != len(set(received_ids))
            or set(received_ids) != positive_ids):
        return fail('target_selection_not_equal_rift_positive')
    if ego.get('actor_id') is not None and str(ego['actor_id']) in set(ids):
        return fail('ego_duplicated_in_agents')

    ego_row = dict(ego)
    ego_row['actor_id'] = '__ego__'
    backgrounds = [ego_row] + [a for a in agents if str(a['actor_id']) not in positive_ids]
    ordered_targets = [a for a in agents if str(a['actor_id']) in positive_ids]
    result['expected_background_actor_ids'] = [str(a['actor_id']) for a in backgrounds]
    result['expected_target_actor_ids'] = [str(a['actor_id']) for a in ordered_targets]
    result['expected_background_count'] = len(backgrounds)
    result['expected_target_count'] = len(ordered_targets)
    # For record 100: exactly 7 background (ego + 6 agents) and 2 targets.
    if len(backgrounds) + len(ordered_targets) != len(agents) + 1:
        return fail('population_count_inconsistent')

    try:
        present = world.get_actors()
        if len(present.filter('vehicle.*')) or len(present.filter('walker.pedestrian.*')):
            return fail('nonempty_world_refuse_spawns')
    except Exception as exc:
        return fail('world_actor_guard_unavailable:%s:%s' % (type(exc).__name__, exc))

    try:
        lib = world.get_blueprint_library()
        pending = []
        for a in backgrounds:
            aid = str(a['actor_id'])
            try:
                bp = lib.find(str(a['type_id']))
                tr = carla_transform(carla, a)
                pending.append((a, carla.command.SpawnActor(bp, tr)))
            except Exception as exc:
                result['background'].append({
                    'actor_id': aid, 'status': 'not_attempted',
                    'reason': 'background_blueprint_or_pose_invalid:%s:%s' % (type(exc).__name__, exc)})
        if len(pending) != len(backgrounds):
            return fail('background_source_cannot_be_replayed')

        replies = client.apply_batch_sync([cmd for a, cmd in pending], False)
        if len(replies) != len(pending):
            # Capture returned actor IDs, but unreported successes may exist.
            created_ids.extend(int(r.actor_id) for r in replies
                               if not getattr(r, 'error', None) and getattr(r, 'actor_id', None) is not None)
            ownership_unknown = True
            return fail('background_batch_response_count_mismatch')
        for (a, _), reply in zip(pending, replies):
            aid = str(a['actor_id'])
            err = getattr(reply, 'error', None)
            new_id = getattr(reply, 'actor_id', None)
            if err or new_id is None:
                result['background'].append({'actor_id': aid, 'status': 'spawn_rejected',
                                             'error': str(err or 'missing_spawned_id')})
            else:
                created_ids.append(int(new_id))
                result['background'].append({'actor_id': aid, 'status': 'spawned',
                                             'spawned_actor_id': int(new_id)})
        if any(r['status'] != 'spawned' for r in result['background']):
            return fail('background_scene_replay_incomplete')

        bp = lib.find('vehicle.tesla.model3')
        # Keep previously spawned targets alive while subsequent targets spawn.
        # This tests joint occupancy rather than isolated success.
        for a in ordered_targets:
            aid = str(a['actor_id'])
            try:
                response = client.apply_batch_sync([
                    carla.command.SpawnActor(bp, carla_transform(carla, a))
                ], False)
                if len(response) != 1:
                    created_ids.extend(int(r.actor_id) for r in response
                                       if not getattr(r, 'error', None) and getattr(r, 'actor_id', None) is not None)
                    ownership_unknown = True
                    result['targets'].append({'actor_id': aid, 'status': 'unknown',
                                              'reason': 'target_response_count_mismatch'})
                    return fail('target_response_count_mismatch')
                reply = response[0]
                err = getattr(reply, 'error', None)
                new_id = getattr(reply, 'actor_id', None)
                if err or new_id is None:
                    result['targets'].append({'actor_id': aid, 'status': 'rejected_in_scene',
                                              'error': str(err or 'missing_spawned_id'),
                                              'raw_pose_correction': False})
                else:
                    created_ids.append(int(new_id))
                    result['targets'].append({'actor_id': aid, 'status': 'accepted_in_scene',
                                              'spawned_actor_id': int(new_id),
                                              'raw_pose_correction': False,
                                              'post_spawn_pose_verified': False})
            except Exception as exc:
                result['targets'].append({'actor_id': aid, 'status': 'unknown',
                                          'reason': 'target_rpc_exception:%s:%s' % (type(exc).__name__, exc)})
                return fail('target_rpc_exception')
        result['joint_model3_spawn_accepted'] = all(
            t['status'] == 'accepted_in_scene' for t in result['targets'])
        result['status'] = 'complete'
        result['reason'] = 'all_original_pose_spawn_requests_have_definitive_outcomes'
        return result
    except Exception as exc:
        return fail('occupancy_rpc_or_blueprint_error:%s:%s' % (type(exc).__name__, exc))
    finally:
        # Synchronous, acknowledged cleanup of only the actors created here.
        # If cleanup cannot be independently verified, never report complete.
        if created_ids:
            try:
                replies = client.apply_batch_sync([
                    carla.command.DestroyActor(i) for i in created_ids
                ], False)
                if len(replies) != len(created_ids) or any(getattr(x, 'error', None) for x in replies):
                    result['cleanup_verified'] = False
                    result['status'] = 'unknown'
                    result['reason'] = 'owned_actor_cleanup_rpc_failed'
                else:
                    remaining = {int(a.id) for a in world.get_actors()}
                    result['cleanup_verified'] = not bool(set(created_ids) & remaining)
                    if not result['cleanup_verified']:
                        result['status'] = 'unknown'
                        result['reason'] = 'owned_actors_remain_after_cleanup'
            except Exception as exc:
                result['cleanup_verified'] = False
                result['status'] = 'unknown'
                result['reason'] = 'owned_actor_cleanup_unverified:%s:%s' % (type(exc).__name__, exc)
        else:
            result['cleanup_verified'] = True
        if ownership_unknown:
            # Never mark cleanup proven when RPC response length was unexpected.
            result['cleanup_verified'] = False
            result['status'] = 'unknown'
            result['reason'] = 'incomplete_rpc_actor_ownership_evidence'




def select_target_actors(row, scene, actors):
    # Target identity comes only from rift_labels.is_interactive == True.
    # Actor pose/state rows come from scene["agents"], joined by actor_id.
    # Optional manifest/snapshot IDs are consistency constraints, not fallbacks.
    validation_errors = []

    if not isinstance(row, dict):
        row = {}
        validation_errors.append("manifest row is not a dict")
    if not isinstance(scene, dict):
        scene = {}
        validation_errors.append("snapshot is not a dict")

    if not isinstance(actors, list):
        actors = []
        validation_errors.append("snapshot agents is missing or not a list")

    actor_rows = [dict(a) for a in actors if isinstance(a, dict)]
    if len(actor_rows) != len(actors):
        validation_errors.append("agents contains non-dict rows")

    agent_id_values = [
        str(a["actor_id"])
        for a in actor_rows
        if a.get("actor_id") is not None
    ]
    agent_ids = set(agent_id_values)
    if len(agent_id_values) != len(agent_ids):
        validation_errors.append("duplicate actor_id values in agents")
    if len(agent_id_values) != len(actor_rows):
        validation_errors.append("one or more agents rows lack actor_id")

    labels = scene.get("rift_labels")
    if not isinstance(labels, list):
        labels = []
        validation_errors.append(
            "snapshot.rift_labels is missing or not a list"
        )

    label_rows = [x for x in labels if isinstance(x, dict)]
    if len(label_rows) != len(labels):
        validation_errors.append("rift_labels contains non-dict rows")

    label_id_values = [
        str(label["actor_id"])
        for label in label_rows
        if label.get("actor_id") is not None
    ]
    label_ids = set(label_id_values)
    if len(label_id_values) != len(label_ids):
        validation_errors.append("duplicate actor_id values in rift_labels")
    if len(label_id_values) != len(label_rows):
        validation_errors.append("one or more rift_labels rows lack actor_id")

    positive_labels = [
        label for label in label_rows
        if label.get("is_interactive") is True
    ]
    positive_id_values = [
        str(label["actor_id"])
        for label in positive_labels
        if label.get("actor_id") is not None
    ]
    positive_ids = set(positive_id_values)

    if len(positive_id_values) != len(positive_ids):
        validation_errors.append(
            "duplicate interactive actor_id values in rift_labels"
        )
    if len(positive_id_values) != len(positive_labels):
        validation_errors.append(
            "an interactive rift_label lacks actor_id"
        )
    if not positive_ids:
        validation_errors.append(
            "no targets found from rift_labels.is_interactive == True"
        )

    if agent_ids != label_ids:
        validation_errors.append(
            "agents actor_id set differs from rift_labels actor_id set"
        )

    def normalize_optional_ids(value, field_name):
        if value is None:
            return set()
        if not isinstance(value, list):
            validation_errors.append(
                field_name + " exists but is not a list"
            )
            return set()
        if any(item is None for item in value):
            validation_errors.append(
                field_name + " contains null actor IDs"
            )
        normalized = [
            str(item) for item in value if item is not None
        ]
        if len(normalized) != len(set(normalized)):
            validation_errors.append(
                field_name + " contains duplicate actor IDs"
            )
        return set(normalized)

    manifest_keys = normalize_optional_ids(
        row.get("target_actor_ids"),
        "manifest.target_actor_ids",
    )
    snapshot_keys = normalize_optional_ids(
        scene.get("target_actor_ids"),
        "snapshot.target_actor_ids",
    )

    if manifest_keys and manifest_keys != positive_ids:
        validation_errors.append(
            "manifest.target_actor_ids conflicts with "
            "rift_labels.is_interactive IDs"
        )
    if snapshot_keys and snapshot_keys != positive_ids:
        validation_errors.append(
            "snapshot.target_actor_ids conflicts with "
            "rift_labels.is_interactive IDs"
        )

    manifest_snapshot_ids_match = None
    if manifest_keys and snapshot_keys:
        manifest_snapshot_ids_match = (
            manifest_keys == snapshot_keys
        )
        if not manifest_snapshot_ids_match:
            validation_errors.append(
                "manifest.target_actor_ids conflicts with "
                "snapshot.target_actor_ids"
            )

    # This diagnostic list is a cross-check only; it never supplies fallback
    # target IDs when the authoritative labels are missing.
    diagnostics = scene.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    rift_diagnostics = diagnostics.get("rift_labeler")
    if not isinstance(rift_diagnostics, dict):
        rift_diagnostics = {}

    candidate_raw = rift_diagnostics.get("rift_candidate_actor_ids")
    diagnostic_candidate_ids = None
    diagnostic_candidate_ids_match = None
    if candidate_raw is not None:
        if not isinstance(candidate_raw, list):
            validation_errors.append(
                "diagnostics.rift_labeler.rift_candidate_actor_ids "
                "exists but is not a list"
            )
        else:
            candidate_values = [
                str(value) for value in candidate_raw
                if value is not None
            ]
            if len(candidate_values) != len(candidate_raw):
                validation_errors.append(
                    "diagnostic candidate IDs contain null values"
                )
            if len(candidate_values) != len(set(candidate_values)):
                validation_errors.append(
                    "diagnostic candidate IDs contain duplicates"
                )
            diagnostic_candidate_ids = set(candidate_values)
            diagnostic_candidate_ids_match = (
                diagnostic_candidate_ids == positive_ids
            )
            if not diagnostic_candidate_ids_match:
                validation_errors.append(
                    "diagnostic candidate IDs conflict with "
                    "rift_labels.is_interactive IDs"
                )

    selected_ids = set(positive_ids)
    actor_by_id = {}
    duplicate_actor_ids = set()
    for actor in actor_rows:
        actor_id = actor.get("actor_id")
        if actor_id is not None:
            key = str(actor_id)
            if key in actor_by_id:
                duplicate_actor_ids.add(key)
            else:
                actor_by_id[key] = actor
    for key in duplicate_actor_ids:
        actor_by_id.pop(key, None)

    matched_ids = selected_ids.intersection(set(actor_by_id))
    missing_ids = sorted(selected_ids - matched_ids)
    if missing_ids:
        validation_errors.append(
            "interactive target IDs missing from agents: "
            + ",".join(missing_ids)
        )

    target_actors = [
        dict(actor_by_id[key])
        for key in sorted(matched_ids)
        if key in actor_by_id
    ]

    selection_status = (
        "pass"
        if selected_ids and not validation_errors
        else "fail"
    )
    selection = {
        "status": selection_status,
        "source": "snapshot.rift_labels.is_interactive",
        "manifest_target_actor_ids": sorted(manifest_keys),
        "snapshot_target_actor_ids": sorted(snapshot_keys),
        "selected_target_actor_ids": sorted(selected_ids),
        "matched_target_actor_ids": sorted(matched_ids),
        "missing_target_actor_ids": missing_ids,
        "manifest_snapshot_ids_match": manifest_snapshot_ids_match,
        "actor_rows": len(actor_rows),
        "target_actor_rows": len(target_actors),
        "rift_label_rows": len(label_rows),
        "interactive_label_actor_ids": sorted(positive_ids),
        "agents_label_ids_match": agent_ids == label_ids,
        "diagnostic_candidate_actor_ids": (
            sorted(diagnostic_candidate_ids)
            if diagnostic_candidate_ids is not None else None
        ),
        "diagnostic_candidate_ids_match_labels": (
            diagnostic_candidate_ids_match
        ),
        "validation_errors": validation_errors,
    }
    return target_actors, selection

def audit_record(
    client: Any,
    carla: Any,
    grp_cls: Any,
    row: dict[str, Any],
    out_dir: Path,
    map_identity_manifest: Any = None,
) -> dict[str, Any]:
    scene = read_json(
        Path(row["snapshot_path"])
    )

    town = (
        scene.get("town")
        or scene.get("map_name")
    )

    require(
        town,
        f"record {row.get('record_index')}: "
        "snapshot missing town",
    )

    expected_xodr, expected_xodr_source, expected_xodr_reason = (
        _p1_resolve_expected_opendrive_sha256(
            scene, town, map_identity_manifest
        )
    )

    # New world episode MUST be ready before ANY map/route/native query.
    # One load_world call, bounded read-only polling, immutable SHA check.
    # Record-specific evidence persists even if a later scientific gate fails.
    world, carla_map, xodr_sha, map_transition = load_verified_world(
        client, str(town), expected_xodr,
        record_id=int(row["record_index"]),
        evidence_path=out_dir / (
            "map_transition_record_%04d.json" % int(row["record_index"])
        ),
    )

    opendrive_status = _p1_opendrive_status(
        xodr_sha, expected_xodr
    )

    map_identity = {
        "town_expected": town,
        "town_actual": carla_map.name,
        "town_pass": (
            normalize_map_name(
                carla_map.name
            )
            == normalize_map_name(
                town
            )
        ),
        "opendrive_sha256_actual": xodr_sha,
        "opendrive_sha256_expected": expected_xodr,
        "opendrive_expected_source": expected_xodr_source,
        "opendrive_expected_reason": expected_xodr_reason,
        "opendrive_status": opendrive_status,
        "opendrive_pass": (
            opendrive_status == "pass"
        ),
    }

    route, route_checks = route_waypoints(
        carla,
        carla_map,
        scene,
    )

    grp = grp_cls(
        carla_map,
        2.0,
    )

    ego = scene["ego"]

    ego_tr = carla_transform(
        carla,
        ego,
    )

    # Pose/state rows use the actual hv_scene_dataset_v3 schema.
    # Target identity is resolved only from rift_labels.is_interactive.
    actors = scene.get("agents", [])
    target_actors, target_selection = select_target_actors(
        row,
        scene,
        actors,
    )

    for a in target_actors:
        a["_ego_location"] = (
            ego_tr.location
        )

    native_rows = []

    for a in target_actors:
        csem = candidate_semantics(
            carla,
            carla_map,
            a,
        )

        inter = exact_interaction(
            carla,
            grp,
            carla_map,
            route,
            a,
        )

        parity = rift_reference_interaction(
            carla,
            grp,
            carla_map,
            route,
            a,
        )

        rt = coord_roundtrip(
            carla,
            ego,
            a,
        )

        parity_status, parity_pass = (
            rift_parity_status(
                parity,
                inter,
            )
        )

        native_status, native_positive = (
            native_truth_status(
                csem,
                inter,
            )
        )

        native_rows.append(
            {
                "actor_id": a.get(
                    "actor_id"
                ),
                "candidate_semantics": csem,
                "interaction": inter,
                "rift_reference_parity": parity,
                "rift_reference_parity_status": (
                    parity_status
                ),
                "rift_reference_parity_pass": (
                    parity_pass
                ),
                "coordinate_roundtrip": rt,
                "native_positive_status": (
                    native_status
                ),
                "native_positive": (
                    native_positive
                ),
            }
        )

    bp = blueprint_measurement(
        world,
        carla,
    )

    sedan_rows = [
        raw_sedan_spawn(
            world,
            carla,
            a,
        )
        for a in target_actors
    ]

    scene_occupancy = (
        scene_occupancy_spawn(
            client,
            world,
            carla,
            scene,
            target_actors,
        )
    )

    # A concurrent map reload must never produce a scientific PASS.
    assert_same_episode_map(
        client, map_transition["verified_episode_id"],
        town, xodr_sha,
    )

    record = {
        "schema_version": (
            "hv_v8_p1_record_v2"
        ),
        "record_index": row[
            "record_index"
        ],
        "scene_id": row[
            "scene_id"
        ],
        "map_identity": map_identity,
        "route_checks": route_checks,
        "native_rows": native_rows,
        "target_selection": target_selection,
        "fixed_model3_blueprint": bp,
        "fixed_model3_raw_pose_spawn": (
            sedan_rows
        ),
        "fixed_model3_scene_occupancy": (
            scene_occupancy
        ),
        "physics_ticks": 0,
        "post_spawn_pose_correction": False,
        "forbidden_fallbacks_used": [],
    }

    write_json(
        out_dir
        / (
            f"record_"
            f"{int(row['record_index']):04d}"
            ".json"
        ),
        record,
    )

    return record


def summarize(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    coord = []
    native_statuses = []
    sedan = []
    map_ok = []
    parity_statuses = []
    target_selection_statuses = []

    interaction_error_count = 0

    for r in records:
        target_selection_statuses.append(
            r.get("target_selection", {}).get("status", "unknown")
        )
        map_ok.append(
            r["map_identity"][
                "town_pass"
            ]
            and r["map_identity"].get(
                "opendrive_status"
            )
            == "pass"
        )

        for n in r["native_rows"]:
            rt = n[
                "coordinate_roundtrip"
            ]

            coord.append(
                rt[
                    "roundtrip_position_error_m"
                ]
                <= 1e-6
                and rt[
                    "roundtrip_z_error_m"
                ]
                <= 1e-6
                and rt[
                    "roundtrip_yaw_error_rad"
                ]
                <= 1e-9
            )

            native_statuses.append(
                n.get(
                    "native_positive_status",
                    "unknown",
                )
            )

            parity_statuses.append(
                n.get(
                    "rift_reference_parity_status",
                    "unknown",
                )
            )

            if (
                n.get(
                    "interaction",
                    {},
                ).get("status")
                == "error"
            ):
                interaction_error_count += 1

        sedan += [
            x["status"] == "accepted"
            for x in r[
                "fixed_model3_raw_pose_spawn"
            ]
        ]

    native_pass_count = sum(
        s == "pass"
        for s in native_statuses
    )

    native_fail_count = sum(
        s == "fail"
        for s in native_statuses
    )

    native_unknown_count = sum(
        s == "unknown"
        for s in native_statuses
    )

    parity_pass_count = sum(
        s == "pass"
        for s in parity_statuses
    )

    parity_fail_count = sum(
        s == "fail"
        for s in parity_statuses
    )

    parity_unavailable_count = sum(
        s == "unavailable"
        for s in parity_statuses
    )

    parity_unknown_count = sum(
        s == "unknown"
        for s in parity_statuses
    )

    parity_evaluable_count = (
        parity_pass_count
        + parity_fail_count
    )

    parity_all_pass = (
        bool(parity_statuses)
        and parity_unavailable_count == 0
        and parity_unknown_count == 0
        and parity_fail_count == 0
        and parity_pass_count
        == len(parity_statuses)
    )

    target_selection_all_pass = (
        bool(target_selection_statuses)
        and all(status == "pass" for status in target_selection_statuses)
    )

    return {
        "schema_version": (
            "hv_v8_p1_summary_v2"
        ),
        "records": len(records),
        "target_selection_all_pass": target_selection_all_pass,
        "target_selection_pass_count": sum(
            status == "pass" for status in target_selection_statuses
        ),
        "target_selection_fail_count": sum(
            status == "fail" for status in target_selection_statuses
        ),
        "target_selection_unknown_count": sum(
            status not in ("pass", "fail")
            for status in target_selection_statuses
        ),

        "map_identity_all_pass": (
            bool(map_ok)
            and all(map_ok)
        ),

        "coordinate_roundtrip_all_pass": (
            bool(coord)
            and all(coord)
        ),

        "rift_reference_parity_all_pass": (
            parity_all_pass
        ),

        "rift_reference_parity_pass_count": (
            parity_pass_count
        ),

        "rift_reference_parity_fail_count": (
            parity_fail_count
        ),

        "rift_reference_parity_unavailable_count": (
            parity_unavailable_count
        ),

        "rift_reference_parity_unknown_count": (
            parity_unknown_count
        ),

        "rift_reference_parity_evaluable_count": (
            parity_evaluable_count
        ),

        "rift_reference_parity_rows": (
            len(parity_statuses)
        ),

        "historical_native_positive_count": (
            native_pass_count
        ),

        "historical_native_negative_count": (
            native_fail_count
        ),

        "historical_native_unknown_count": (
            native_unknown_count
        ),

        "historical_native_rows": (
            len(native_statuses)
        ),

        "interaction_error_count": (
            interaction_error_count
        ),

        "fixed_model3_raw_pose_spawn_accepted": (
            sum(sedan)
        ),

        "fixed_model3_raw_pose_spawn_rows": (
            len(sedan)
        ),

        "p1_pass": (
            bool(map_ok)
            and all(map_ok)
            and target_selection_all_pass
            and bool(coord)
            and all(coord)
            and parity_all_pass
            and native_unknown_count == 0
            and interaction_error_count == 0
            and all(
                r[
                    "fixed_model3_blueprint"
                ].get("status")
                == "pass"
                for r in records
            )
        ),

        "important": (
            "Raw sedan rejection is a measured "
            "fact, not repaired. "
            "Native truth and sedan spawn are "
            "separate. "
            "Unknown native truth is never "
            "converted to negative. "
            "RIFT parity is fail-closed: "
            "unavailable/unknown/error is not PASS."
        ),
    }



# P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2
import json


def _p1_normalize_town(value):
    text = str(value or "").replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.split("/")[-1].casefold()


def _p1_valid_sha256(value):
    if not isinstance(value, str):
        return False
    value = value.strip()
    return len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _p1_validate_map_identity_manifest_data(data):
    if not isinstance(data, dict):
        raise ContractError("map identity manifest root must be an object")
    if data.get("schema_version") != "p1_map_identity_manifest_v1":
        raise ContractError(
            "unsupported map identity manifest schema_version: {!r}".format(
                data.get("schema_version")
            )
        )
    maps = data.get("maps")
    if not isinstance(maps, list) or not maps:
        raise ContractError("map identity manifest has no non-empty maps list")

    seen = set()
    for item in maps:
        if not isinstance(item, dict):
            raise ContractError("map identity manifest contains a non-object map row")
        town_key = _p1_normalize_town(item.get("town"))
        sha = item.get("opendrive_sha256")
        if not town_key:
            raise ContractError("map identity manifest row is missing town")
        if town_key in seen:
            raise ContractError(
                "duplicate normalized town in map identity manifest: {!r}".format(
                    item.get("town")
                )
            )
        seen.add(town_key)
        if not _p1_valid_sha256(sha):
            raise ContractError(
                "invalid OpenDRIVE SHA-256 for town {!r}".format(item.get("town"))
            )
        if not isinstance(item.get("provenance_record_count"), int) or item.get(
            "provenance_record_count"
        ) < 1:
            raise ContractError(
                "missing provenance evidence count for town {!r}".format(
                    item.get("town")
                )
            )
        if not item.get("source_sha256_values") or not item.get(
            "calibrated_map_sha256_values"
        ):
            raise ContractError(
                "missing source/calibrated-map evidence for town {!r}".format(
                    item.get("town")
                )
            )
    return data


def _p1_load_map_identity_manifest(path):
    if not path:
        return None

    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractError(
            "cannot read --map-identity-manifest {!r}: {}".format(path, exc)
        )

    return _p1_validate_map_identity_manifest_data(data)


def _p1_resolve_expected_opendrive_sha256(scene, town, map_identity_manifest):
    scene_sha = scene.get("opendrive_sha256")
    if scene_sha is not None and scene_sha != "":
        if not _p1_valid_sha256(scene_sha):
            raise ContractError("snapshot opendrive_sha256 is not a valid SHA-256")
        scene_sha = scene_sha.lower()
    else:
        scene_sha = None

    if map_identity_manifest is not None:
        town_key = _p1_normalize_town(town)
        matches = [
            item for item in map_identity_manifest.get("maps", [])
            if _p1_normalize_town(item.get("town")) == town_key
        ]
        if len(matches) != 1:
            raise ContractError(
                "map identity manifest must contain exactly one row for town {!r}; found {}".format(
                    town, len(matches)
                )
            )
        manifest_sha = matches[0]["opendrive_sha256"].lower()
        if scene_sha is not None and scene_sha != manifest_sha:
            raise ContractError(
                "snapshot/manifest OpenDRIVE hash conflict for town {!r}: snapshot={}, manifest={}".format(
                    town, scene_sha, manifest_sha
                )
            )
        return manifest_sha, "map_identity_manifest", None

    if scene_sha is not None:
        return scene_sha, "snapshot.opendrive_sha256", None
    return None, "missing", "no_valid_snapshot_hash_or_map_identity_manifest"


def _p1_opendrive_status(actual_sha256, expected_sha256):
    if not _p1_valid_sha256(actual_sha256) or not _p1_valid_sha256(expected_sha256):
        return "unknown"
    return "pass" if actual_sha256.lower() == expected_sha256.lower() else "fail"


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument(
        "--manifest",
        required=True,
        help=(
            "JSONL with record_index, "
            "scene_id, snapshot_path"
        ),
    )

    p.add_argument(
        "--rift-root",
        required=True,
    )

    p.add_argument(
        "--rift-recognition-config"
    )

    p.add_argument(
        "--rift-mode",
        default="eval",
    )

    p.add_argument(
        "--carla-pythonapi"
    )

    p.add_argument(
        "--host",
        default="127.0.0.1",
    )

    p.add_argument(
        "--port",
        type=int,
        default=31120,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    p.add_argument(
        "--limit",
        type=int,
    )

    p.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help=(
            "Whole CARLA readiness "
            "window in seconds."
        ),
    )

    p.add_argument(
        "--startup-rpc-timeout",
        type=float,
        default=5.0,
        help=(
            "Timeout for one readiness "
            "RPC attempt."
        ),
    )

    p.add_argument(
        "--startup-poll-interval",
        type=float,
        default=2.0,
        help=(
            "Delay between readiness "
            "attempts."
        ),
    )

    p.add_argument(
        "--rpc-timeout",
        type=float,
        default=120.0,
        help=(
            "Per-call CARLA timeout "
            "after startup readiness."
        ),
    )

    p.add_argument(
        "--expected-carla-version",
        default="0.9.15",
        help=(
            "Required CARLA client/server "
            "version."
        ),
    )

    p.add_argument(
        "--map-identity-manifest",
        default=None,
        help="Trusted per-town map identity manifest with OpenDRIVE SHA-256.",
    )

    args = p.parse_args()

    try:
        map_identity_manifest = _p1_load_map_identity_manifest(
            args.map_identity_manifest
        )
        apply_rift_config(
            args.rift_recognition_config,
            args.rift_mode,
        )

        carla, grp = (
            import_carla_and_grp(
                args.rift_root,
                args.carla_pythonapi,
            )
        )

        rows = read_jsonl(
            Path(args.manifest)
        )

        require(
            rows,
            "P1 manifest is empty",
        )

        if args.limit is not None:
            rows = rows[:args.limit]

        out = Path(
            args.output
        ).resolve()

        require(
            not out.exists(),
            f"Refuse overwrite: {out}",
        )

        out.mkdir(
            parents=True
        )

        # ------------------------------------------------------------
        # CARLA STARTUP READINESS
        #
        # This is deliberately different from a 30s single RPC.
        # A fresh client is used on every attempt until:
        #
        #   get_server_version
        #   get_world
        #   get_map
        #
        # all succeed.
        #
        # No scene-level retry or pose repair is performed.
        # ------------------------------------------------------------
        client, readiness = (
            wait_for_carla_ready(
                carla=carla,
                host=args.host,
                port=args.port,
                startup_timeout=(
                    args.startup_timeout
                ),
                startup_rpc_timeout=(
                    args.startup_rpc_timeout
                ),
                interval=(
                    args.startup_poll_interval
                ),
            )
        )

        write_json(
            out
            / "startup_readiness.json",
            readiness,
        )

        require(
            readiness["server_version"]
            == args.expected_carla_version,
            "CARLA server version mismatch: "
            f"expected "
            f"{args.expected_carla_version}, "
            f"got "
            f"{readiness['server_version']}",
        )

        require(
            readiness["client_version"]
            == args.expected_carla_version,
            "CARLA client version mismatch: "
            f"expected "
            f"{args.expected_carla_version}, "
            f"got "
            f"{readiness['client_version']}",
        )

        # After startup, use a longer per-RPC timeout.
        #
        # This is NOT retry/fallback. It only prevents a legitimate
        # load_world / map query from being confused with startup failure.
        client.set_timeout(
            args.rpc_timeout
        )

        # ------------------------------------------------------------
        # DEDICATED SERVER GUARD
        # ------------------------------------------------------------
        current = client.get_world()

        vehicle_count = len(
            current
            .get_actors()
            .filter("vehicle.*")
        )

        require(
            vehicle_count == 0,
            (
                "Refuse P1 on non-empty "
                f"CARLA world "
                f"({vehicle_count} vehicles). "
                "Use a dedicated server."
            ),
        )

        records = []

        for row in rows:
            records.append(
                audit_record(
                    client,
                    carla,
                    grp,
                    row,
                    out,
                    map_identity_manifest,
                )
            )

        summary = summarize(
            records
        )

        write_json(
            out / "summary.json",
            summary,
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

        raise SystemExit(
            0
            if summary["p1_pass"]
            else 3
        )

    except (
        ContractError,
        FileNotFoundError,
        ImportError,
        TimeoutError,
    ) as e:
        print(
            f"P1_FAIL {e}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()