#!/usr/bin/env python3
"""Run a reproducible P1 one-record CARLA audit from a FROZEN code snapshot.

- Does not modify the user's live hv_v8/p1_native_audit.py.
- Inspects previous backups to identify source-state regression.
- Builds one auditable frozen capsule and reconciles BOTH known API contracts there.
- Records hashes/diff, verifies the actual module import + CLI + contracts.
- Starts CARLA audit only after the complete code-contract preflight passes.
- Does not change native truth, RIFT model, routes, spawn poses, or evaluation results.

Python >=3.8, standard library only. Run with:
    python hv_v8/p1_frozen_mainline_runner.py
or test the script itself without CARLA with --self-test.
"""
import argparse
import ast
import datetime
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

P1_MODULE = "hv_v8.p1_native_audit"
MAP_MARKER = "P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2"
ROUTE_MARKER = "P1_ROUTE_CARLA_SCOPE_FIX_V1"


class GuardError(RuntimeError):
    pass


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha(Path(path).read_bytes())


def argument_names(fn):
    return [x.arg for x in getattr(fn.args, "posonlyargs", []) + fn.args.args]


def one_function(tree, name):
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise GuardError("expected exactly one def {} (found {})".format(name, len(matches)))
    return matches[0]


def call_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return call_name(node.value) + "." + node.attr
    return ""


def all_calls(node, target):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and call_name(n.func) == target]


def named_arg(call, index, name):
    return len(call.args) > index and isinstance(call.args[index], ast.Name) and call.args[index].id == name


def contract_state(source):
    tree = ast.parse(source)
    route = one_function(tree, "route_waypoints")
    audit = one_function(tree, "audit_record")
    main = one_function(tree, "main")
    route_calls = all_calls(audit, "route_waypoints")
    if len(route_calls) != 1:
        raise GuardError("audit_record must have exactly one call to route_waypoints (got {})".format(len(route_calls)))
    audited_calls = all_calls(main, "audit_record")
    if len(audited_calls) != 1:
        raise GuardError("main must have exactly one call to audit_record (got {})".format(len(audited_calls)))
    cli_calls = all_calls(main, "p.add_argument")
    cli_map_flag = any(c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "--map-identity-manifest" for c in cli_calls)
    map_load_calls = all_calls(main, "_p1_load_map_identity_manifest")
    map_resolve_calls = all_calls(audit, "_p1_resolve_expected_opendrive_sha256")
    identity_record_fields = [n.value for n in ast.walk(audit) if isinstance(n, ast.Constant) and isinstance(n.value, str)]

    route_ok = (
        argument_names(route)[:3] == ["carla", "carla_map", "scene"]
        and "carla" in argument_names(audit)
        and named_arg(route_calls[0], 0, "carla")
        and named_arg(route_calls[0], 1, "carla_map")
        and named_arg(route_calls[0], 2, "scene")
    )
    map_ok = (
        cli_map_flag
        and "map_identity_manifest" in argument_names(audit)
        and any(named_arg(c, len(c.args)-1, "map_identity_manifest") for c in audited_calls if c.args)
        and len(map_load_calls) == 1
        and len(map_resolve_calls) == 1
        and all_function_names_present(tree, [
            "_p1_load_map_identity_manifest",
            "_p1_resolve_expected_opendrive_sha256",
            "_p1_opendrive_status",
        ])
        and "opendrive_expected_source" in identity_record_fields
        and "opendrive_status" in identity_record_fields
        and MAP_MARKER in source
    )
    return {
        "route_ok": bool(route_ok),
        "map_ok": bool(map_ok),
        "route_marker_present": ROUTE_MARKER in source,
        "map_marker_present": MAP_MARKER in source,
        "map_cli_present": bool(cli_map_flag),
        "route_args": argument_names(route),
        "route_call_args": [call_name(a) for a in route_calls[0].args],
        "audit_args": argument_names(audit),
        "audit_call_args": [call_name(a) for a in audited_calls[0].args],
    }


def all_function_names_present(tree, names):
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    return all(n in defined for n in names)


def load_patch_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise GuardError("unable to import patch helper: {}".format(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reconcile(source, map_helper, route_helper):
    """Perform both existing contract changes in memory. No live-file edits."""
    trace = []
    initial = contract_state(source)
    if not initial["map_ok"]:
        if initial["map_cli_present"] or initial["map_marker_present"]:
            raise GuardError("map CLI/marker is partially present: refuse blind rewrite; state={}".format(initial))
        # Existing map patch takes a file path, so use a *temporary staging copy*
        # rather than ever writing the source or deployed p1_native_audit.py.
        with tempfile.TemporaryDirectory(prefix="p1_map_patch_stage_") as d:
            stage = Path(d) / "p1_native_audit.py"
            stage.write_text(source, encoding="utf-8")
            prepared_path, prepared_source = map_helper.prepare_patch(stage)
            if Path(prepared_path).resolve() != stage.resolve():
                raise GuardError("map helper returned unexpected staging path")
            source = prepared_source
            trace.append("map_manifest_contract_applied_in_memory")
    else:
        trace.append("map_manifest_contract_already_valid")

    after_map = contract_state(source)
    if not after_map["map_ok"]:
        raise GuardError("map manifest contract FAILED after in-memory reconciliation")

    if not after_map["route_ok"] and after_map["route_marker_present"]:
        raise GuardError("route marker present but route contract missing: refuse inconsistent patch")
    patched, changed = route_helper._build_patched_source(source)
    source = patched
    trace.append("route_carla_contract_applied_in_memory" if changed else "route_carla_contract_already_valid")

    final = contract_state(source)
    if not final["route_ok"] or not final["map_ok"]:
        raise GuardError("combined contract FAILED: {}".format(final))
    compile(source, "frozen/hv_v8/p1_native_audit.py", "exec")
    return source, final, trace


def forensic_files(source_path):
    p = Path(source_path)
    patterns = [
        "p1_native_audit.py.bak_map_identity_runtime_v2_*",
        "p1_native_audit.py.bak_route_carla_scope_*",
    ]
    found = []
    for pattern in patterns:
        found.extend(p.parent.glob(pattern))
    found = sorted(set(found), key=lambda x: x.stat().st_mtime)
    found.append(p)
    seen = set()
    for path in found:
        if path in seen:
            continue
        seen.add(path)
        try:
            raw = path.read_bytes()
            st = contract_state(raw.decode("utf-8"))
            yield {
                "path": str(path),
                "sha256": sha(raw),
                "bytes": len(raw),
                "mtime": datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "route_ok": st["route_ok"],
                "map_ok": st["map_ok"],
                "route_marker": st["route_marker_present"],
                "map_marker": st["map_marker_present"],
            }
        except Exception as exc:
            yield {"path": str(path), "sha256": sha_file(path), "inspect_error": str(exc)}


def as_absolute(path):
    return str(Path(path).resolve())


def input_checks(root, py):
    files = {
        "p1_manifest": root / "data/hv_init/v7/manifests/p1_calibration95.jsonl",
        "map_identity_manifest": root / "data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json",
        "rift_config": root / "RIFT/rift/cbv/recognition/config/attention.yaml",
    }
    for key, path in files.items():
        if not path.is_file():
            raise GuardError("{} is missing: {}".format(key, path))
    pythonapi = root / "carla/PythonAPI/carla"
    if not pythonapi.is_dir():
        raise GuardError("CARLA PythonAPI missing: {}".format(pythonapi))
    if not Path(py).is_file():
        raise GuardError("Python executable missing: {}".format(py))

    manifest = json.loads(files["map_identity_manifest"].read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "p1_map_identity_manifest_v1":
        raise GuardError("unexpected map manifest schema_version")
    maps = manifest.get("maps", [])
    if len(maps) != 5:
        raise GuardError("expected 5 town hashes; got {}".format(len(maps)))
    map_names = [m.get("town") for m in maps]
    if set(map_names) != {"Town04", "Town11", "Town12", "Town13", "Town15"}:
        raise GuardError("unexpected towns in map identity manifest: {}".format(map_names))
    expected_town12 = "26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f"
    town12 = next((m.get("opendrive_sha256") for m in maps if m.get("town") == "Town12"), None)
    if town12 != expected_town12:
        raise GuardError("Town12 OpenDRIVE hash differs from previously verified native map value")
    return files


def test_helpers(root):
    route_helper = load_patch_module(root / "hv_v8/p1_route_carla_scope_fix_v1.py", "p1routefixisolated")
    map_helper = load_patch_module(root / "hv_v8/p1_map_identity_manifest_runtime_v2_patch.py", "p1mapfixisolated")
    return map_helper, route_helper


def log_json(name, obj):
    print("{}={}".format(name, json.dumps(obj, ensure_ascii=False, sort_keys=True)), flush=True)


def run_process(cmd, cwd, env):
    print("EXECUTING: {}".format(" ".join(cmd)), flush=True)
    p = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    return p.wait()


def print_result(result_dir):
    summary_path = result_dir / "summary.json"
    record_path = result_dir / "record_0000.json"
    if not summary_path.is_file():
        print("P1_SUMMARY_NOT_GENERATED", flush=True)
        if result_dir.exists():
            for p in sorted(result_dir.glob("*")):
                print("P1_RESULT_FILE={}".format(p), flush=True)
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        identity = {}
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            identity = record.get("map_identity", {})
        out = {
            "p1_pass": summary.get("p1_pass"),
            "records": summary.get("records"),
            "map_identity_all_pass": summary.get("map_identity_all_pass"),
            "coordinate_roundtrip_all_pass": summary.get("coordinate_roundtrip_all_pass"),
            "target_selection_all_pass": summary.get("target_selection_all_pass"),
            "rift_reference_parity_all_pass": summary.get("rift_reference_parity_all_pass"),
            "historical_native_unknown_count": summary.get("historical_native_unknown_count"),
            "interaction_error_count": summary.get("interaction_error_count"),
            "map_identity": identity,
        }
        log_json("P1_ACTUAL_RESULT", out)
        if summary.get("p1_pass") is True and summary.get("records") == 1:
            print("P1_SINGLE_RECORD_SCIENTIFIC_ACCEPTANCE=PASS")
            print("P1_FULL_95_ACCEPTANCE=NOT_RUN")
        else:
            print("P1_SINGLE_RECORD_SCIENTIFIC_ACCEPTANCE=FAIL_OR_INCOMPLETE")
            print("P1_FULL_95_ACCEPTANCE=NOT_RUN")
        print("P1_SUMMARY_FILE={}".format(summary_path))
        print("P1_RECORD_FILE={}".format(record_path) if record_path.is_file() else "P1_RECORD_MISSING")
    except Exception as exc:
        print("P1_OUTPUT_PARSE_ERROR={}".format(exc))


class Tee:
    def __init__(self, writers):
        self.writers = writers
    def write(self, data):
        for writer in self.writers:
            writer.write(data)
            writer.flush()
    def flush(self):
        for writer in self.writers:
            writer.flush()
    def isatty(self):
        return False


def run(root, py, no_carla=False):
    root = root.resolve()
    log_dir = root / "data/hv_init/v7"
    if not log_dir.is_dir():
        raise GuardError("log directory missing (will not create unexpected project data): {}".format(log_dir))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid())
    capsule = log_dir / ("p1_frozen_capsule_" + timestamp)
    result_dir = log_dir / ("p1_real_frozen_smoke_" + timestamp)
    log_path = log_dir / ("p1_frozen_mainline_" + timestamp + ".log")
    if capsule.exists() or result_dir.exists() or log_path.exists():
        raise GuardError("unique output collision, refusing overwrite")

    with log_path.open("x", encoding="utf-8") as log:
        orig_stdout, orig_stderr = sys.stdout, sys.stderr
        sys.stdout = Tee([orig_stdout, log])
        sys.stderr = Tee([orig_stderr, log])
        try:
            print("P1_FROZEN_MAINLINE_V1")
            print("ROOT={}".format(root))
            print("PY={}".format(py))
            print("RUN_LOG={}".format(log_path))
            print("CAPSULE={}".format(capsule))
            print("P1_OUTPUT={}".format(result_dir))
            original_path = root / "hv_v8/p1_native_audit.py"
            if not original_path.is_file():
                raise GuardError("live P1 module is missing: {}".format(original_path))
            base = original_path.read_bytes()
            base_hash = sha(base)
            print("LIVE_SOURCE_SHA256={}".format(base_hash))
            forensic = list(forensic_files(original_path))
            log_json("SOURCE_BACKUP_FORENSICS", forensic)
            files = input_checks(root, py)
            map_helper, route_helper = test_helpers(root)

            # Immutable capsule for the current run. Changes below touch only this copy.
            capsule.mkdir(mode=0o755)
            ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
            shutil.copytree(str(root / "hv_v8"), str(capsule / "hv_v8"), ignore=ignore)
            frozen_module = capsule / "hv_v8/p1_native_audit.py"
            if original_path.read_bytes() != base or frozen_module.read_bytes() != base:
                raise GuardError("live P1 source changed DURING snapshot: refuse mixed version")
            print("P1_SNAPSHOT_SOURCE_CONSISTENCY_PASS")

            staged_source, state, changes = reconcile(base.decode("utf-8"), map_helper, route_helper)
            staged_bytes = staged_source.encode("utf-8")
            frozen_module.write_bytes(staged_bytes)
            if sha_file(frozen_module) != sha(staged_bytes):
                raise GuardError("frozen source hash mismatch after reconciliation")
            print("FROZEN_EXECUTABLE_SHA256={}".format(sha(staged_bytes)))
            log_json("CODE_CONTRACT", state)
            log_json("CODE_RECONCILIATION", changes)
            diff_path = capsule / "p1_runtime_changes.diff"
            diff_path.write_text("".join(difflib.unified_diff(
                base.decode("utf-8").splitlines(True), staged_source.splitlines(True),
                fromfile="live/p1_native_audit.py", tofile="frozen/p1_native_audit.py",
            )), encoding="utf-8")
            metadata = {
                "root": str(root),
                "active_module": str(original_path),
                "live_sha256_before": base_hash,
                "frozen_sha256": sha(staged_bytes),
                "patches_applied_to": "FROZEN_CAPSULE_ONLY",
                "reconciliation_steps": changes,
                "contracts": state,
                "inputs": {key: {"path": str(path), "sha256": sha_file(path)} for key, path in files.items()},
                "backup_forensics": forensic,
                "diff": str(diff_path),
                "result_directory": str(result_dir),
            }
            (capsule / "p1_source_provenance.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print("SOURCE_PROVENANCE={}".format(capsule / "p1_source_provenance.json"))
            print("SOURCE_DIFF={}".format(diff_path))

            api = root / "carla/PythonAPI/carla"
            egg = api / "dist/carla-0.9.15-py3.7-linux-x86_64.egg"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                str(p) for p in [capsule, root, root / "RIFT", root / "scenario-dreamer",
                                 root / "carla/PythonAPI", api] + ([egg] if egg.exists() else [])
            ) + os.pathsep + env.get("PYTHONPATH", "")
            # cwd=capsule assures python -m loads frozen hv_v8, not live project's module.
            command_prefix = [str(py), "-m", P1_MODULE]
            print("=== EXECUTABLE MODULE PREFLIGHT ===")
            identify_code = (
                "import hv_v8.p1_native_audit as m; "
                "import pathlib; print(pathlib.Path(m.__file__).resolve())"
            )
            proc = subprocess.run([str(py), "-c", identify_code], cwd=str(capsule), env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                print(proc.stdout)
                raise GuardError("frozen module import failed (rc={})".format(proc.returncode))
            module_path = proc.stdout.strip().splitlines()[-1]
            if Path(module_path).resolve() != frozen_module.resolve():
                print(proc.stdout)
                raise GuardError("module did not resolve to the FROZEN capsule: {}".format(module_path))
            print("FROZEN_MODULE_PATH={}".format(module_path))
            help_proc = subprocess.run(command_prefix + ["--help"], cwd=str(capsule), env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if help_proc.returncode != 0 or "--map-identity-manifest" not in help_proc.stdout:
                print(help_proc.stdout)
                raise GuardError("frozen module CLI preflight failed")
            print("P1_FROZEN_MAINLINE_PREFLIGHT_PASS")

            if no_carla:
                print("P1_CARLA_SKIPPED_BY_EXPLICIT_NO_CARLA")
                return 0

            args = [
                "--manifest", str(files["p1_manifest"]),
                "--rift-root", str(root / "RIFT"),
                "--rift-recognition-config", str(files["rift_config"]),
                "--rift-mode", "eval",
                "--carla-pythonapi", str(api),
                "--map-identity-manifest", str(files["map_identity_manifest"]),
                "--host", "127.0.0.1",
                "--port", "31120",
                "--startup-timeout", "600",
                "--startup-rpc-timeout", "5",
                "--startup-poll-interval", "2",
                "--rpc-timeout", "120",
                "--expected-carla-version", "0.9.15",
                "--limit", "1",
                "--output", str(result_dir),
            ]
            if result_dir.exists():
                raise GuardError("result directory already exists; refuse overwrite")
            print("=== REAL CARLA P1 SMOKE: one-record, no fallback ===")
            frozen_hash_before = sha_file(frozen_module)
            rc = run_process(command_prefix + args, cwd=capsule, env=env)
            print("P1_PROCESS_EXIT={}".format(rc))
            print_result(result_dir)
            if rc == 0:
                summary_file = result_dir / "summary.json"
                if not summary_file.is_file():
                    rc = 31
                else:
                    try:
                        audited = json.loads(summary_file.read_text(encoding="utf-8"))
                        if audited.get("p1_pass") is not True or audited.get("records") != 1:
                            rc = 31
                    except (ValueError, OSError):
                        rc = 31
                if rc == 31:
                    print("P1_PROCESS_RETURNED_ZERO_BUT_SCIENTIFIC_ACCEPTANCE_NOT_MET")
            if sha_file(frozen_module) != frozen_hash_before:
                raise GuardError("frozen executable was mutated during P1 run")
            print("FROZEN_SOURCE_UNCHANGED_DURING_RUN_PASS")
            live_after = sha_file(original_path)
            print("LIVE_SOURCE_SHA256_AFTER={}".format(live_after))
            if live_after != base_hash:
                print("P1_LIVE_SOURCE_DRIFT_DETECTED: live module changed during frozen run")
            else:
                print("P1_LIVE_SOURCE_UNCHANGED_DURING_RUN_PASS")
            return rc
        except (GuardError, AssertionError, SyntaxError) as exc:
            print("P1_STOP_GUARD={}".format(exc))
            return 42
        except Exception as exc:
            traceback.print_exc()
            print("P1_STOP_EXCEPTION={}".format(exc))
            return 43
        finally:
            print("RUN_LOG={}".format(log_path))
            print("CAPSULE={}".format(capsule))
            print("P1_OUTPUT={}".format(result_dir))
            sys.stdout, sys.stderr = orig_stdout, orig_stderr


def self_test():
    import inspect
    # Fixture reproduces the concrete line layouts supplied by user, without CARLA.
    original = '''from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any
class ContractError(RuntimeError): pass
DEFAULT = {"max_ego_route_len": 50}
def route_waypoints(
    carla_map: Any,
    scene: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]]]:
    expected = carla.Transform(carla.Location(0, 0, 0))
    return [], []
def audit_record(
    client: Any,
    carla: Any,
    grp_cls: Any,
    row: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    scene = row["scene"]
    town = "Town12"
    xodr_sha = "0" * 64
    expected_xodr = scene.get(
        "opendrive_sha256"
    )
    opendrive_status = "unknown"
    map_identity = {
        "opendrive_sha256_expected": expected_xodr,
        "opendrive_status": opendrive_status,
    }
    route, route_checks = route_waypoints(
        carla_map,
        scene,
    )
    return map_identity
def apply_rift_config(*a, **k): pass
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output")
    args = p.parse_args()
    try:
        apply_rift_config(
            "test"
        )
        records = []
        for row in []:
            records.append(
                audit_record(
                    client,
                    carla,
                    grp,
                    row,
                    out,
                )
            )
    except Exception:
        raise
if __name__ == "__main__":
    main()
'''
    import tempfile
    # map patch relies on one exact source segment; our fixture tests the combined
    # reconciler without modifying a real file.
    with tempfile.TemporaryDirectory(prefix="p1_mainline_selftest_") as d:
        rt = Path(d)
        map_path = rt / "p1_map_identity_manifest_runtime_v2_patch.py"
        route_path = rt / "p1_route_carla_scope_fix_v1.py"
        # Derive installed patch locations from the --root directory of this file.
        here = Path(__file__).resolve().parent
        candidates = [here, here / "hv_v8", here.parent / "hv_v8"]
        for candidate in candidates:
            if (candidate / map_path.name).exists() and (candidate / route_path.name).exists():
                map_path = candidate / map_path.name
                route_path = candidate / route_path.name
                break
        if not map_path.is_file() or not route_path.is_file():
            raise GuardError("self-test requires both patch modules next to runner or under project/hv_v8")
        mh = load_patch_module(map_path, "p1mapselftest")
        rh = load_patch_module(route_path, "p1routeselftest")
        before = contract_state(original)
        assert not before["route_ok"] and not before["map_ok"]
        final, state, trace = reconcile(original, mh, rh)
        assert state["route_ok"] and state["map_ok"]
        second, second_state, second_trace = reconcile(final, mh, rh)
        assert second == final
        assert second_state["route_ok"] and second_state["map_ok"]
        assert len(trace) == 2
        print("P1_FROZEN_MAINLINE_SELF_TEST_PASS")
        print("combined map and route patch => pass")
        print("AST code contract => pass")
        print("in-memory patch => pass")
        print("idempotent frozen source => pass")
        print("local fixture only; CARLA not run")


def main():
    parser = argparse.ArgumentParser(description="Freeze and verify P1 source before real CARLA audit")
    parser.add_argument("--root", type=Path, help="Project root (default: parent of script directory)")
    parser.add_argument("--python", help="P1 Python binary")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-carla", action="store_true", help="Only prepare the frozen capsule; skip actual CARLA audit")
    parser.add_argument("--no-pause", action="store_true", help="Do not prompt after the run")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = args.root or Path(__file__).resolve().parent.parent
    py = args.python or "/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python"
    rc = run(root, py, args.no_carla)
    print("P1_FROZEN_RUNNER_EXIT={}".format(rc), flush=True)
    if not args.no_pause and sys.stdin.isatty():
        try:
            input("Press Enter to return to the SAME shell... ")
        except (KeyboardInterrupt, EOFError):
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
