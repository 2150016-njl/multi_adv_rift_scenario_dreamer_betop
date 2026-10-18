#!/usr/bin/env python3
"""Resume P1 full-95 from the already-PASS frozen V3 capsule (v2).

No source patching. No rerun of the passed one-record smoke. Fail-closed on
hash/input drift. Never resets a CARLA world unless explicitly requested on a
dedicated server; only a newly loaded same town is used to clear residual actors.
Compatible with Python 3.8+.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

KNOWN_FROZEN_SHA = "34a80666b1030434c528d0b91f7bb7f2994eb65bba21b5b4c8540fa972898d2c"
CAPSULE_BASENAME = "p1_frozen_capsule_20261018_033841_9535"
SMOKE_BASENAME = "p1_real_frozen_smoke_20261018_033841_9535"
EXPECTED_FIRST_IDS = {"6121", "6125", "6137", "6138"}


class GuardError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise GuardError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Tee:
    def __init__(self, *targets):
        self.targets = targets
    def write(self, text):
        for t in self.targets:
            t.write(text)
            t.flush()
    def flush(self):
        for t in self.targets:
            t.flush()
    def isatty(self):
        return False


def contract_check(root, capsule, previous_smoke):
    frozen = capsule / "hv_v8/p1_native_audit.py"
    metadata = capsule / "p1_source_provenance.json"
    require(frozen.is_file(), "frozen P1 source is missing: {}".format(frozen))
    require(metadata.is_file(), "frozen provenance is missing: {}".format(metadata))
    info = read_json(metadata)
    source_sha = digest(frozen)
    require(source_sha == KNOWN_FROZEN_SHA,
            "frozen SHA differs from the previously PASSED source: {}; expected {}".format(
                source_sha, KNOWN_FROZEN_SHA))
    require(info.get("frozen_sha256") == source_sha,
            "capsule provenance frozen SHA mismatch")
    require(Path(info.get("root", "")).resolve() == root.resolve(),
            "capsule belongs to another project root")
    require(info.get("patches_applied_to") == "FROZEN_CAPSULE_ONLY",
            "capsule provenance cannot attest frozen-only changes")
    required_contracts = (
        "map_ok", "route_ok", "strict_agents_rift_labels_ok",
        "world_snapshot_raw_pose_ok", "verified_road_end_contract_ok",
    )
    for key in required_contracts:
        require(info.get("contracts", {}).get(key) is True,
                "frozen code contract is missing: {}".format(key))

    inputs = info.get("inputs") or {}
    for key in ("p1_manifest", "map_identity_manifest", "rift_config"):
        value = inputs.get(key) or {}
        file_path = Path(value.get("path", ""))
        require(file_path.is_file(), "original input file no longer exists: {}={}".format(key, file_path))
        require(digest(file_path) == value.get("sha256"),
                "original input changed since one-record PASS: {}={}".format(key, file_path))
    p1_manifest = Path(inputs["p1_manifest"]["path"])
    with p1_manifest.open(encoding="utf-8") as f:
        count = sum(1 for line in f if line.strip())
    require(count == 95, "P1 manifest row count is {}, expected 95".format(count))

    summary = previous_smoke / "summary.json"
    record = previous_smoke / "record_0000.json"
    require(summary.is_file() and record.is_file(), "previous PASSED smoke evidence unavailable")
    s, r = read_json(summary), read_json(record)
    require(s.get("p1_pass") is True and s.get("records") == 1,
            "previous smoke scientific acceptance is not PASS")
    require(s.get("rift_reference_parity_pass_count") == 4,
            "previous smoke is not the 4/4 RIFT PASS record")
    require(s.get("map_identity_all_pass") is True
            and s.get("target_selection_all_pass") is True
            and s.get("coordinate_roundtrip_all_pass") is True
            and s.get("rift_reference_parity_all_pass") is True,
            "previous smoke had an incomplete/failed scientific predicate")
    selection = r.get("target_selection") or {}
    require(set(map(str, selection.get("selected_target_actor_ids") or [])) == EXPECTED_FIRST_IDS,
            "previous smoke did not select the verified four targets")
    identity = r.get("map_identity") or {}
    require(identity.get("opendrive_expected_source") == "map_identity_manifest"
            and identity.get("opendrive_status") == "pass",
            "previous smoke map identity is not independently verified")
    print("P1_REUSE_PREVIOUS_SINGLE_PASS")
    print("FROZEN_EXECUTABLE_SHA256={}".format(source_sha))
    print("P1_INPUT_HASHES_UNCHANGED_PASS")
    print("PREVIOUS_SINGLE_SUMMARY={}".format(summary))
    return inputs, source_sha


def python_environment(root, capsule):
    api = root / "carla/PythonAPI/carla"
    egg = api / "dist/carla-0.9.15-py3.7-linux-x86_64.egg"
    paths = [capsule, root, root / "RIFT", root / "scenario-dreamer",
             root / "carla/PythonAPI", api]
    if egg.exists():
        paths.append(egg)
    require(api.is_dir(), "CARLA PythonAPI missing: {}".format(api))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in paths) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def list_vehicles(world):
    actors = world.get_actors().filter("vehicle.*")
    return [(int(a.id), str(a.type_id)) for a in actors]


def normalize_town(value):
    return str(value).replace("\\", "/").rstrip("/").split("/")[-1]


def prepare_world(client, reset, map_manifest):
    world = client.get_world()
    map_name = str(world.get_map().name)
    town = normalize_town(map_name)
    maps = read_json(map_manifest).get("maps") or []
    require(isinstance(maps, list) and bool(maps),
            "trusted P1 map identity manifest has no map rows")
    allowed = {normalize_town(row.get("town")) for row in maps}
    # The currently *idle* CARLA map is not itself a P1 evidence map.
    # audit_record() loads each snapshot's town using client.load_world(town)
    # and validates the actual OpenDRIVE SHA-256 against this manifest.
    # A clean dedicated server may therefore begin at Town10HD_Opt.
    if town not in allowed:
        print("P1_IDLE_MAP_OUTSIDE_MANIFEST_ALLOWED={}".format(map_name))
        print("P1_NOTE=per-record CARLA map identity is still fail-closed")
    actors_before = list_vehicles(world)
    print("CARLA_MAP_BEFORE={}".format(map_name))
    print("CARLA_VEHICLES_BEFORE={}".format(json.dumps(actors_before, ensure_ascii=False)))
    if actors_before:
        if not reset:
            raise GuardError(
                "CARLA world is not clean ({} vehicles). No actors were destroyed. "
                "On an exclusively dedicated CARLA server, rerun with "
                "--reset-dedicated-world. Otherwise start a separate clean server.".format(len(actors_before))
            )
        # This is a destructive action, and it is never implicit. The flag
        # means the user confirms exclusive ownership of this CARLA instance.
        print("DEDICATED_WORLD_RESET_REQUESTED=yes")
        print("DEDICATED_WORLD_RESET_ACTION=client.load_world({!r})".format(town))
        client.set_timeout(120.0)
        world = client.load_world(town)
        new_map = normalize_town(world.get_map().name)
        require(new_map == town, "world reset did not reload the same town")
        for attempt in range(4):
            present = list_vehicles(world)
            if not present:
                break
            time.sleep(0.4)
        require(not present, "vehicle actors remain after dedicated-world reset: {}".format(present))
        print("P1_DEDICATED_WORLD_RESET_PASS")
    else:
        print("P1_DEDICATED_WORLD_ALREADY_CLEAN_PASS")
    # Prevent a dirty world between these stages. This is not a substitute for
    # a dedicated instance: no other CARLA user/process should touch the world.
    remain = list_vehicles(client.get_world())
    require(not remain, "vehicles reappeared after clean-world check: {}".format(remain))
    print("P1_CLEAN_WORLD_BEFORE_FULL95_PASS")


def run_full95(root, py, capsule, inputs, env, output):
    require(not output.exists(), "full95 output path already exists: {}".format(output))
    command = [
        str(py), "-m", "hv_v8.p1_native_audit",
        "--manifest", inputs["p1_manifest"]["path"],
        "--rift-root", str(root / "RIFT"),
        "--rift-recognition-config", inputs["rift_config"]["path"],
        "--rift-mode", "eval",
        "--carla-pythonapi", str(root / "carla/PythonAPI/carla"),
        "--map-identity-manifest", inputs["map_identity_manifest"]["path"],
        "--host", "127.0.0.1", "--port", "31120",
        "--startup-timeout", "600",
        "--startup-rpc-timeout", "5",
        "--startup-poll-interval", "2",
        "--rpc-timeout", "120",
        "--expected-carla-version", "0.9.15",
        "--limit", "95",
        "--output", str(output),
    ]
    print("=== CARLA P1 FULL95: SAME PASSED FROZEN SOURCE, NO REPEATED SMOKE ===")
    print("P1_FULL95_OUTPUT={}".format(output))
    print("EXECUTING={}".format(" ".join(command)))
    process = subprocess.Popen(command, cwd=str(capsule), env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    for line in process.stdout:
        sys.stdout.write(line)
    code = process.wait()
    print("P1_FULL95_PROCESS_EXIT={}".format(code))
    return code


def strict_full95_acceptance(root, result_dir):
    src = root / "hv_v8/p1_frozen_mainline_runner_v3.py"
    require(src.is_file(), "V3 strict evaluator not found: {}".format(src))
    spec = importlib.util.spec_from_file_location("p1_v3_strict_acceptance", str(src))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._strict_acceptance(result_dir, 95)


def main(args):
    root = Path(__file__).resolve().parent.parent
    py = Path("/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python")
    data_dir = root / "data/hv_init/v7"
    capsule = Path(args.capsule).resolve() if args.capsule else data_dir / CAPSULE_BASENAME
    prior = Path(args.previous_smoke).resolve() if args.previous_smoke else data_dir / SMOKE_BASENAME

    require(py.is_file(), "expected rift Python binary unavailable: {}".format(py))
    require(data_dir.is_dir(), "P1 result data directory missing")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid())
    out = data_dir / ("p1_real_frozen_full95_resume_" + stamp)
    log_path = data_dir / ("p1_full95_resume_" + stamp + ".log")

    with log_path.open("x", encoding="utf-8") as logfile:
        original_out, original_err = sys.stdout, sys.stderr
        sys.stdout = Tee(original_out, logfile)
        sys.stderr = Tee(original_err, logfile)
        try:
            print("P1_FULL95_RESUME_V2")
            print("ROOT={}".format(root))
            print("CAPSULE={}".format(capsule))
            print("LOG={}".format(log_path))
            print("P1_FULL95_OUTPUT={}".format(out))
            inputs, frozen_sha = contract_check(root, capsule, prior)
            env = python_environment(root, capsule)

            path_check = subprocess.run(
                [str(py), "-c", "import hv_v8.p1_native_audit as m; import os; print(os.path.realpath(m.__file__))"],
                cwd=str(capsule), env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            require(path_check.returncode == 0, "frozen module import failed:\n{}".format(path_check.stdout))
            actual_path = Path(path_check.stdout.strip().splitlines()[-1]).resolve()
            require(actual_path == (capsule / "hv_v8/p1_native_audit.py").resolve(),
                    "wrong module entry point: {}".format(actual_path))
            print("P1_FROZEN_MODULE_IDENTITY_PASS")
            require(digest(actual_path) == frozen_sha, "frozen module hash drifted before run")

            sys.path[:0] = [str(root / "carla/PythonAPI"), str(root / "carla/PythonAPI/carla")]
            egg = root / "carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
            if egg.exists():
                sys.path.insert(0, str(egg))
            try:
                import carla
            except Exception as exc:
                raise GuardError("CARLA Python module import failed: {}: {}".format(type(exc).__name__, exc))
            client = carla.Client("127.0.0.1", 31120)
            client.set_timeout(30.0)
            if args.inspect_only:
                world = client.get_world()
                print("P1_INSPECTION_ONLY_MAP={}".format(world.get_map().name))
                print("P1_INSPECTION_ONLY_VEHICLES={}".format(
                    json.dumps(list_vehicles(world), ensure_ascii=False)))
                print("P1_INSPECTION_ONLY; no world reset or full95 run was started")
                return 0
            prepare_world(client, args.reset_dedicated_world, Path(inputs["map_identity_manifest"]["path"]))

            rc = run_full95(root, py, capsule, inputs, env, out)
            if out.joinpath("summary.json").is_file():
                print("=== SUMMARY ===")
                print(out.joinpath("summary.json").read_text(encoding="utf-8"))
            else:
                print("P1_FULL95_SUMMARY_NOT_GENERATED")
            passed, errors, evidence = strict_full95_acceptance(root, out)
            print("P1_FULL95_STRICT_EVALUATOR_SHA256={}".format(
                digest(root / "hv_v8/p1_frozen_mainline_runner_v3.py")))
            print("P1_FULL95_STRICT_EVIDENCE={}".format(json.dumps(evidence, ensure_ascii=False, sort_keys=True)))
            print("P1_FULL95_STRICT_ERRORS={}".format(json.dumps(errors[:80], ensure_ascii=False)))
            if digest(capsule / "hv_v8/p1_native_audit.py") != frozen_sha:
                raise GuardError("frozen source was mutated during full95")
            print("P1_FROZEN_SOURCE_UNCHANGED_DURING_FULL95_PASS")
            if rc == 0 and passed:
                print("P1_FULL95_STRICT_ACCEPTANCE_PASS")
                return 0
            print("P1_FULL95_STRICT_ACCEPTANCE_FAIL")
            return rc if rc != 0 else 32
        except Exception as exc:
            print("P1_FULL95_RESUME_EXCEPTION={}: {}".format(type(exc).__name__, exc))
            raise
        finally:
            print("RUN_LOG={}".format(log_path))
            print("P1_FULL95_OUTPUT={}".format(out))
            sys.stdout, sys.stderr = original_out, original_err


def self_test():
    require(normalize_town("Carla/Maps/Town12/Town12") == "Town12", "town normalization")
    class Actor:
        def __init__(self, a_id, t):
            self.id = a_id
            self.type_id = t
    class Actors:
        def __init__(self, actors):
            self.actors = actors
        def filter(self, pattern):
            require(pattern == "vehicle.*", "test mock filter contract")
            return [a for a in self.actors if a.type_id.startswith("vehicle.")]
    class Map:
        def __init__(self, name):
            self.name = name
    class World:
        def __init__(self, actors, map_name="Carla/Maps/Town12/Town12"):
            self.actors = Actors(actors)
            self.map_name = map_name
        def get_actors(self):
            return self.actors
        def get_map(self):
            return Map(self.map_name)
    class Client:
        def __init__(self, world):
            self.world = world
            self.reset_count = 0
        def get_world(self):
            return self.world
        def set_timeout(self, seconds):
            self.timeout = seconds
        def load_world(self, town):
            require(town == normalize_town(self.world.map_name), "reset same town")
            self.reset_count += 1
            self.world = World([], self.world.map_name)
            return self.world
    with tempfile.TemporaryDirectory() as d:
        manifest = Path(d) / "maps.json"
        manifest.write_text(json.dumps({"maps": [{"town": "Town12"}]}), encoding="utf-8")
        clean = Client(World([]))
        prepare_world(clean, False, manifest)
        require(clean.reset_count == 0, "clean server must not be reset")
        dirty = Client(World([Actor(3701, "vehicle.tesla.model3"), Actor(3702, "vehicle.tesla.model3")]))
        try:
            prepare_world(dirty, False, manifest)
            raise AssertionError("dirty server was not rejected")
        except GuardError as exc:
            require("not clean" in str(exc), "wrong dirty-world error")
        require(dirty.reset_count == 0, "dirty server was modified without opt-in")
        prepare_world(dirty, True, manifest)
        require(dirty.reset_count == 1, "explicit clean-world reset missing")
        require(not list_vehicles(dirty.get_world()), "reset did not clean actors")
        # Regression for the live server's initial default idle CARLA map.
        # It is not one of the P1 data maps, but the P1 audit loads those later.
        default_map = "Carla/Maps/Town10HD_Opt"
        clean_default = Client(World([], default_map))
        prepare_world(clean_default, False, manifest)
        require(clean_default.reset_count == 0,
                "a clean idle Town10HD_Opt was modified before P1")
        dirty_default = Client(World([Actor(999, "vehicle.tesla.model3")], default_map))
        try:
            prepare_world(dirty_default, False, manifest)
            raise AssertionError("dirty default-map world was not rejected")
        except GuardError as exc:
            require("not clean" in str(exc), "wrong default-world dirty error")
        require(dirty_default.reset_count == 0, "dirty default-map world was touched")
        prepare_world(dirty_default, True, manifest)
        require(dirty_default.reset_count == 1, "explicit default-map reset was ignored")
        require(not list_vehicles(dirty_default.get_world()), "default-map reset did not clean")
        print("P1_FULL95_RESUME_OFFLINE_SELF_TEST_PASS")
        print("clean world => not reset; dirty world => fail without opt-in")
        print("initial Town10HD_Opt with no vehicles => ALLOWED, no reset")
        print("dirty Town10HD_Opt => reject unless explicitly dedicated reset")
        print("explicit dedicated-world reset => reload same map, then zero vehicles")
        print("no CARLA server was contacted")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", help="Override prior PASS frozen capsule path")
    parser.add_argument("--previous-smoke", help="Override prior PASS single-record output path")
    parser.add_argument("--reset-dedicated-world", action="store_true",
                        help="Destructively reload the current town ONLY on an exclusively dedicated CARLA server")
    parser.add_argument("--inspect-only", action="store_true",
                        help="Inspect without launching 95 records (world reset still requires explicit flag)")
    parser.add_argument("--self-test", action="store_true")
    opts = parser.parse_args()
    try:
        if opts.self_test:
            self_test()
            rc = 0
        else:
            rc = main(opts)
    except GuardError as err:
        print("P1_FULL95_RESUME_STOP={}".format(err), file=sys.stderr)
        rc = 42
    except Exception:
        traceback.print_exc()
        rc = 43
    print("P1_FULL95_RESUME_EXIT={}".format(rc))
    raise SystemExit(rc)
