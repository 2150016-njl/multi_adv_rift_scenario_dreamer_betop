#!/usr/bin/env python3
# Patch hv_v8/p1_native_audit.py so raw sedan spawn acceptance requires a
# post-spawn CARLA world snapshot and a validated pose.
import argparse
import datetime
import math
from pathlib import Path


PATCH_MARKER = "P1_RAW_POSE_SNAPSHOT_FIX_V1"

REPLACEMENT = r'''
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
'''


def self_test():
    def classify(expected, actual, pos_tol=0.30, yaw_tol=2.0):
        if actual is None:
            return "unknown"
        expected_zero = all(abs(expected[k]) < 1e-6 for k in ("x", "y", "z"))
        actual_zero = all(abs(actual[k]) < 1e-6 for k in ("x", "y", "z"))
        if actual_zero and not expected_zero:
            return "unknown"
        pos = math.hypot(actual["x"] - expected["x"], actual["y"] - expected["y"])
        yaw = abs((actual["yaw"] - expected["yaw"] + 180.0) % 360.0 - 180.0)
        return "pass" if pos <= pos_tol and yaw <= yaw_tol else "fail"

    expected = {"x": 1305.1177, "y": 4999.2324, "z": 371.02, "yaw": -0.1424}
    good = dict(expected)
    zero = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    bad = {"x": 1306.0, "y": 4999.2324, "z": 371.02, "yaw": -0.1424}

    assert classify(expected, good) == "pass"
    assert classify(expected, zero) == "unknown"
    assert classify(expected, bad) == "fail"
    assert classify(expected, None) == "unknown"
    print("P1_RAW_POSE_CONTRACT_SELF_TEST_PASS")
    print("matching pose => pass")
    print("all-zero pose for non-origin target => unknown")
    print("out-of-tolerance pose => fail")
    print("missing pose => unknown")


def patch_file(path: Path):
    if not path.is_file():
        raise SystemExit("ERROR: target file does not exist: %s" % path)

    source = path.read_text(encoding="utf-8")

    if PATCH_MARKER in source:
        raise SystemExit(
            "ERROR: patch marker already present; refusing to patch twice: %s"
            % path
        )

    start_marker = "def raw_sedan_spawn("
    end_marker = "\ndef scene_occupancy_spawn("

    if source.count(start_marker) != 1:
        raise SystemExit(
            "ERROR: expected exactly one raw_sedan_spawn definition; found %d"
            % source.count(start_marker)
        )

    start = source.index(start_marker)
    end = source.find(end_marker, start)
    if end < 0:
        raise SystemExit(
            "ERROR: could not find scene_occupancy_spawn after raw_sedan_spawn"
        )

    old_function = source[start:end]
    if "world.try_spawn_actor" not in old_function:
        raise SystemExit("ERROR: raw_sedan_spawn does not match expected source")
    if "spawned.get_transform()" not in old_function:
        raise SystemExit(
            "ERROR: current raw_sedan_spawn has changed; refusing blind replacement"
        )

    replacement = REPLACEMENT.strip("\n") + "\n"
    updated = source[:start] + replacement + source[end:]

    # Compile in memory before changing the server file.
    compile(updated, str(path), "exec")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + ".bak_raw_pose_snapshot_" + timestamp)
    backup.write_text(source, encoding="utf-8")

    path.write_text(updated, encoding="utf-8")
    try:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    except Exception:
        path.write_text(source, encoding="utf-8")
        raise

    print("P1_RAW_POSE_SNAPSHOT_PATCH_APPLIED")
    print("file =", path)
    print("backup =", backup)
    print("waits for up to 5 server snapshots; does not call world.tick()")
    print("accepted requires pose_validation_status=pass")
    print("all-zero pose for non-origin target is unknown, never PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--file", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        if args.file is None:
            return

    if args.file is None:
        parser.error("provide --file PATH, or use --self-test alone")
    patch_file(args.file)


if __name__ == "__main__":
    main()
