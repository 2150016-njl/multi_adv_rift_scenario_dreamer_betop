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




# Provenance: derived verbatim from three previously supplied, historically
# exercised P1 endpoint-audit/None-guard/pass-contract fixes. Integrated here
# to avoid another copy/patch/rollback cycle on the active source.
_EP_OLD_XODR_BLOCK = '        xodr_wp = carla_map.get_waypoint_xodr(\n            int(r["road_id"]),\n            int(r["lane_id"]),\n            float(r["s"]),\n        )\n\n        require(\n            xodr_wp is not None,\n            "ego_route[%d] "\n            "get_waypoint_xodr returned None"\n            % i,\n        )\n\n        expected = carla.Transform(\n'
_EP_NEW_XODR_BLOCK = '        xodr_wp = carla_map.get_waypoint_xodr(\n            int(r["road_id"]),\n            int(r["lane_id"]),\n            float(r["s"]),\n        )\n\n        # Construct the serialized transform before classifying an XODR miss.\n        # Do not modify r["s"] or substitute a nearby waypoint.\n        expected = carla.Transform(\n'
_EP_IDENTITY_ANCHOR = '        require(\n            (\n                int(exact_wp.road_id)\n                == int(r["road_id"])\n                and int(exact_wp.section_id)\n                == int(r["section_id"])\n                and int(exact_wp.lane_id)\n                == int(r["lane_id"])\n            ),\n            (\n                f"ego_route[{i}] exact lane "\n                "identity mismatch"\n            ),\n        )\n\n        wp = exact_wp\n'
_EP_IDENTITY_REPLACEMENT = '        require(\n            (\n                int(exact_wp.road_id)\n                == int(r["road_id"])\n                and int(exact_wp.section_id)\n                == int(r["section_id"])\n                and int(exact_wp.lane_id)\n                == int(r["lane_id"])\n            ),\n            (\n                f"ego_route[{i}] exact lane "\n                "identity mismatch"\n            ),\n        )\n\n        # Classify only the proven road-end boundary case. The map lookup\n        # remains visibly failed in the audit output; this is not an XODR pass.\n        xodr_road_length_m = _p1_road_lengths.get(int(r["road_id"]))\n        xodr_boundary_verified = False\n        xodr_endpoint_delta_m = None\n        exact_s_error_m = abs(float(exact_wp.s) - float(r["s"]))\n        exact_position_error_m = exact_wp.transform.location.distance(\n            expected.location\n        )\n        exact_yaw_error_deg = abs((\n            float(exact_wp.transform.rotation.yaw)\n            - float(expected.rotation.yaw) + 180.0\n        ) % 360.0 - 180.0)\n\n        if xodr_wp is None:\n            if xodr_road_length_m is not None:\n                xodr_endpoint_delta_m = (\n                    float(r["s"]) - float(xodr_road_length_m)\n                )\n\n            require(\n                xodr_road_length_m is not None\n                and abs(xodr_endpoint_delta_m) <= 1e-4\n                and exact_s_error_m <= 1e-4\n                and exact_position_error_m <= 0.05\n                and exact_yaw_error_deg <= 0.5,\n                (\n                    f"ego_route[{i}] get_waypoint_xodr returned None; "\n                    "not a verified road-end boundary: "\n                    f"road_length={xodr_road_length_m}, "\n                    f"s_minus_road_end={xodr_endpoint_delta_m}, "\n                    f"exact_s_error={exact_s_error_m}, "\n                    f"exact_position_error_m={exact_position_error_m}, "\n                    f"exact_yaw_error_deg={exact_yaw_error_deg}"\n                ),\n            )\n            xodr_boundary_verified = True\n\n        wp = exact_wp\n'
_EP_ROWS_ANCHOR = '    rows = scene.get("ego_route", [])\n\n    require(\n'
_EP_ROWS_REPLACEMENT = '    rows = scene.get("ego_route", [])\n\n    # Read road lengths from the same runtime OpenDRIVE used by CARLA. This\n    # is used only to classify an exact road-end XODR lookup miss.\n    import xml.etree.ElementTree as _p1_ET\n\n    _p1_road_lengths = {}\n    _p1_opendrive_error = None\n    try:\n        _p1_root = _p1_ET.fromstring(carla_map.to_opendrive())\n        for _p1_road in _p1_root.findall("road"):\n            _p1_road_lengths[int(_p1_road.attrib["id"])] = float(\n                _p1_road.attrib["length"]\n            )\n    except Exception as _p1_exc:\n        _p1_opendrive_error = repr(_p1_exc)\n\n    require(\n'
_EP_CHECK_ANCHOR = '        check["route_index"] = i\n\n        check["expected_road_id"] = int(\n'
_EP_CHECK_REPLACEMENT = '        check["route_index"] = i\n\n        check["xodr_found"] = xodr_wp is not None\n        check["xodr_lookup_status"] = (\n            "ok" if xodr_wp is not None\n            else "verified_road_end_boundary"\n        )\n        check["xodr_endpoint_exception_verified"] = xodr_boundary_verified\n        check["xodr_road_length_m"] = xodr_road_length_m\n        check["xodr_s_minus_road_end_m"] = xodr_endpoint_delta_m\n        check["xodr_exact_s_error_m"] = exact_s_error_m\n        check["xodr_exact_position_error_m"] = exact_position_error_m\n        check["xodr_exact_yaw_error_deg"] = exact_yaw_error_deg\n        check["xodr_opendrive_parse_error"] = _p1_opendrive_error\n\n        check["expected_road_id"] = int(\n'
_EP_POSITION_OLD = '        check[\n            "xodr_exact_position_error_m"\n        ] = distance2(\n            xodr_wp.transform.location,\n            exact_wp.transform.location,\n        )\n'
_EP_POSITION_NEW = '        if xodr_wp is not None:\n            check[\n                "xodr_exact_position_error_m"\n            ] = distance2(\n                xodr_wp.transform.location,\n                exact_wp.transform.location,\n            )\n        else:\n            # XODR lookup failed; keep the comparison explicitly unavailable.\n            check["xodr_exact_position_error_m"] = None\n'
_EP_YAW_OLD = '        check[\n            "xodr_exact_yaw_error_deg"\n        ] = abs(\n            (\n                (\n                    float(\n                        xodr_wp.transform.rotation.yaw\n                    )\n                    - float(\n                        exact_wp.transform.rotation.yaw\n                    )\n                    + 180.0\n                )\n                % 360.0\n            )\n            - 180.0\n        )\n'
_EP_YAW_NEW = '        if xodr_wp is not None:\n            check[\n                "xodr_exact_yaw_error_deg"\n            ] = abs(\n                (\n                    (\n                        float(\n                            xodr_wp.transform.rotation.yaw\n                        )\n                        - float(\n                            exact_wp.transform.rotation.yaw\n                        )\n                        + 180.0\n                    )\n                    % 360.0\n                )\n                - 180.0\n            )\n        else:\n            # XODR lookup failed; keep the comparison explicitly unavailable.\n            check["xodr_exact_yaw_error_deg"] = None\n'
_EP_PASS_OLD = '        check["pass"] = (\n            int(wp.road_id)\n            == int(r["road_id"])\n            and int(wp.section_id)\n            == int(r["section_id"])\n            and int(wp.lane_id)\n            == int(r["lane_id"])\n            and check["s_abs_error_m"] <= 0.05\n            and check["position_error_m"] <= 0.30\n            and check["yaw_error_deg"] <= 2.0\n            and check[\n                "xodr_exact_position_error_m"\n            ] <= 0.30\n            and check[\n                "xodr_exact_yaw_error_deg"\n            ] <= 2.0\n        )\n'
_EP_PASS_NEW = '        # Separate core route identity checks from the XODR cross-check.\n        # A verified road-end exception does not make the XODR lookup succeed.\n        route_identity_pass = (\n            int(wp.road_id)\n            == int(r["road_id"])\n            and int(wp.section_id)\n            == int(r["section_id"])\n            and int(wp.lane_id)\n            == int(r["lane_id"])\n            and check["s_abs_error_m"] <= 0.05\n            and check["position_error_m"] <= 0.30\n            and check["yaw_error_deg"] <= 2.0\n        )\n\n        check["route_identity_pass"] = bool(route_identity_pass)\n\n        if xodr_wp is not None:\n            xodr_position_error = check["xodr_exact_position_error_m"]\n            xodr_yaw_error = check["xodr_exact_yaw_error_deg"]\n            xodr_comparison_pass = (\n                xodr_position_error is not None\n                and xodr_position_error <= 0.30\n                and xodr_yaw_error is not None\n                and xodr_yaw_error <= 2.0\n            )\n            check["xodr_exact_comparison_status"] = (\n                "pass" if xodr_comparison_pass else "fail"\n            )\n        else:\n            # The earlier guard must have verified the road-end exception.\n            # Never compare None to a number or report XODR as found.\n            xodr_comparison_pass = bool(xodr_boundary_verified)\n            check["xodr_exact_comparison_status"] = (\n                "not_evaluable_verified_endpoint_exception"\n                if xodr_boundary_verified\n                else "unknown"\n            )\n\n        check["pass"] = bool(\n            route_identity_pass and xodr_comparison_pass\n        )\n'


def _ep_replace_once(source, old, new, label):
    n = source.count(old)
    if n != 1:
        raise GuardError(
            "endpoint reconciliation expected exactly one {} block, got {}; "
            "no active source was modified".format(label, n)
        )
    return source.replace(old, new, 1)


def endpoint_contract_state(source):
    tree = ast.parse(source)
    route = one_function(tree, 'route_waypoints')
    lines = source.splitlines(keepends=True)
    start = route.lineno - 1
    end = route.end_lineno
    content = ''.join(lines[start:end])
    found = (
        'xodr_endpoint_exception_verified' in content
        and 'xodr_exact_comparison_status' in content
        and 'not_evaluable_verified_endpoint_exception' in content
        and 'route_identity_pass = (' in content
        and 'xodr_boundary_verified' in content
        and 'xodr_road_length_m' in content
        and 'xodr_found' in content
        and 'xodr_wp is None' in content
        and 'exact_s_error_m <= 1e-4' in content
        and 'exact_position_error_m <= 0.05' in content
        and 'exact_yaw_error_deg <= 0.5' in content
        and 'abs(xodr_endpoint_delta_m) <= 1e-4' in content
        and 'route_identity_pass and xodr_comparison_pass' in content
        and 'if xodr_wp is not None:' in content
        and '_p1_ET.fromstring(carla_map.to_opendrive())' in content
    )
    # Fatal immediate require must not prevent the strict endpoint evaluation.
    old_guard_present = _EP_OLD_XODR_BLOCK in content
    return bool(found and not old_guard_present)


def reconcile_endpoint(source):
    """Restore verified road-end audit in the FROZEN copy only, fail-closed."""
    if endpoint_contract_state(source):
        return source, ['verified_road_end_contract_already_valid']
    if any(x in source for x in (
        'xodr_endpoint_exception_verified',
        'not_evaluable_verified_endpoint_exception',
        'route_identity_pass = (',
    )):
        raise GuardError(
            'P1 endpoint fix is partially present or its invariant changed; '
            'refuse implicit rewrite, original source remains untouched'
        )
    ops = [
        (_EP_ROWS_ANCHOR, _EP_ROWS_REPLACEMENT, 'runtime OpenDRIVE road lengths'),
        (_EP_OLD_XODR_BLOCK, _EP_NEW_XODR_BLOCK, 'premature XODR None raise'),
        (_EP_IDENTITY_ANCHOR, _EP_IDENTITY_REPLACEMENT, 'exact identity and endpoint bounds'),
        (_EP_CHECK_ANCHOR, _EP_CHECK_REPLACEMENT, 'endpoint evidence in route_checks'),
        (_EP_POSITION_OLD, _EP_POSITION_NEW, 'XODR None position dereference'),
        (_EP_YAW_OLD, _EP_YAW_NEW, 'XODR None yaw dereference'),
        (_EP_PASS_OLD, _EP_PASS_NEW, 'route pass predicate'),
    ]
    for old, new, label in ops:
        source = _ep_replace_once(source, old, new, label)
    compile(source, 'frozen/hv_v8/p1_native_audit.py', 'exec')
    if not endpoint_contract_state(source):
        raise GuardError('endpoint source invariant failed after fix; no CARLA run')
    return source, ['verified_road_end_contract_applied_in_memory']


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


def verify_route37_38_and_map_evidence(result_dir):
    """Independently inspect real P1 record: verified endpoint != XODR success."""
    path = result_dir / "record_0000.json"
    if not path.is_file():
        print("P1_ROUTE_ENDPOINT_RECORD_UNAVAILABLE: record_0000.json was not generated")
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    checks = record.get("route_checks")
    if not isinstance(checks, list):
        raise GuardError("P1 route_checks missing from generated record")
    by_index = {row.get("route_index"): row for row in checks if isinstance(row, dict)}
    if len(checks) != 50:
        raise GuardError("P1 Town12 first record expected 50 route checks, got {}".format(len(checks)))
    for idx in (37, 38):
        row = by_index.get(idx)
        if not isinstance(row, dict):
            raise GuardError("P1 route index {} missing".format(idx))
        expect = {
            "expected_road_id": 630,
            "expected_lane_id": 1,
            "expected_section_id": 0,
            "xodr_found": False,
            "xodr_endpoint_exception_verified": True,
            "xodr_lookup_status": "verified_road_end_boundary",
            "xodr_exact_comparison_status": "not_evaluable_verified_endpoint_exception",
            "route_identity_pass": True,
            "pass": True,
        }
        wrong = {k: (row.get(k), v) for k, v in expect.items() if row.get(k) != v}
        if wrong:
            raise GuardError("P1 verified endpoint #{} evidence mismatch: {}".format(idx, wrong))
        delta = row.get("xodr_s_minus_road_end_m")
        if not isinstance(delta, (int, float)) or abs(delta) > 1e-4:
            raise GuardError("P1 endpoint #{} does not match runtime OpenDRIVE road end".format(idx))
        if row.get("xodr_exact_position_error_m") is not None:
            raise GuardError("P1 endpoint #{} fabricated an XODR-vs-exact position value".format(idx))
    identity = record.get("map_identity", {})
    if not isinstance(identity, dict):
        raise GuardError("P1 record is missing map_identity dict")
    print("P1_VERIFIED_ENDPOINT_37_38_EVIDENCE_PASS")
    log_json("P1_ROUTE_ENDPOINT_EVIDENCE", [
        {"route_index": i, "xodr_found": by_index[i]["xodr_found"],
         "verified": by_index[i]["xodr_endpoint_exception_verified"],
         "s_minus_road_end_m": by_index[i]["xodr_s_minus_road_end_m"]}
        for i in (37, 38)
    ])
    log_json("P1_REAL_MAP_IDENTITY", identity)
    return identity


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
            print("P1_FROZEN_MAINLINE_V2")
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
            staged_source, endpoint_changes = reconcile_endpoint(staged_source)
            changes.extend(endpoint_changes)
            state["verified_road_end_contract_ok"] = endpoint_contract_state(staged_source)
            if not state["verified_road_end_contract_ok"]:
                raise GuardError("frozen route endpoint invariant missing")
            print("P1_VERIFIED_XODR_ENDPOINT_CONTRACT_PASS")
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
            print("=== REAL CARLA P1 SMOKE: strict verified endpoint, no pose/model fallback ===")
            frozen_hash_before = sha_file(frozen_module)
            rc = run_process(command_prefix + args, cwd=capsule, env=env)
            print("P1_PROCESS_EXIT={}".format(rc))
            print_result(result_dir)
            if (result_dir / "record_0000.json").is_file():
                verify_route37_38_and_map_evidence(result_dir)
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




def endpoint_self_test():
    "Exercise strict real-audit body with fake waypoint API; no CARLA needed."
    # The function layout below mirrors the server's multiline route audit
    # structure, including all seven original patch anchors.
    fixture = '''from typing import Any

DEFAULT = {"max_ego_route_len": 50}

def require(cond, message):
    if not cond:
        raise ValueError(message)

def distance2(a, b):
    return a.distance(b)

def route_waypoints(
    carla: Any,
    carla_map: Any,
    scene: dict,
):
    rows = scene.get("ego_route", [])

    require(
        rows,
        "missing route",
    )
    out = []
    checks = []
    for i, r in enumerate(rows[:50]):
        xodr_wp = carla_map.get_waypoint_xodr(
            int(r["road_id"]),
            int(r["lane_id"]),
            float(r["s"]),
        )

        require(
            xodr_wp is not None,
            "ego_route[%d] "
            "get_waypoint_xodr returned None"
            % i,
        )

        expected = carla.Transform(
            carla.Location(float(r["x"]), float(r["y"]), float(r["z"])),
            carla.Rotation(yaw=float(r["yaw_deg"])),
        )
        exact_wp = carla_map.get_waypoint(
            expected.location,
            project_to_road=False,
            lane_type=carla.LaneType.Driving,
        )
        require(
            exact_wp is not None,
            "route exact lookup failed",
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

        wp = exact_wp
        check = {}
        check["route_index"] = i

        check["expected_road_id"] = int(
            r["road_id"]
        )
        check["s_abs_error_m"] = abs(float(wp.s) - float(r["s"]))
        check["position_error_m"] = distance2(wp.transform.location, expected.location)
        check["yaw_error_deg"] = abs((float(wp.transform.rotation.yaw)-float(expected.rotation.yaw)+180)%360-180)
        check[
            "xodr_exact_position_error_m"
        ] = distance2(
            xodr_wp.transform.location,
            exact_wp.transform.location,
        )
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
        check["pass"] = (
            int(wp.road_id)
            == int(r["road_id"])
            and int(wp.section_id)
            == int(r["section_id"])
            and int(wp.lane_id)
            == int(r["lane_id"])
            and check["s_abs_error_m"] <= 0.05
            and check["position_error_m"] <= 0.30
            and check["yaw_error_deg"] <= 2.0
            and check[
                "xodr_exact_position_error_m"
            ] <= 0.30
            and check[
                "xodr_exact_yaw_error_deg"
            ] <= 2.0
        )
        checks.append(check)
        require(check["pass"], "route identity mismatch")
        out.append(wp)
    return out, checks
'''

    patched, changes = reconcile_endpoint(fixture)
    assert changes == ['verified_road_end_contract_applied_in_memory'], changes
    assert endpoint_contract_state(patched)
    unchanged, again = reconcile_endpoint(patched)
    assert unchanged == patched
    assert again == ['verified_road_end_contract_already_valid']

    ns = {}
    exec(compile(patched, '<mock_p1_route>', 'exec'), ns)
    import copy

    class Location:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = float(x), float(y), float(z)
        def distance(self, b):
            return ((self.x-b.x)**2+(self.y-b.y)**2+(self.z-b.z)**2)**0.5

    class Rotation:
        def __init__(self, yaw=0):
            self.yaw = float(yaw)

    class Transform:
        def __init__(self, location, rotation):
            self.location, self.rotation = location, rotation

    class Carla:
        class LaneType:
            Driving = 1
    Carla.Transform = Transform
    Carla.Location = Location
    Carla.Rotation = Rotation

    class Waypoint:
        def __init__(self, x, s):
            self.road_id, self.section_id, self.lane_id = 630, 0, 1
            self.s = s
            self.transform = Transform(Location(x, 0, 0), Rotation(0))

    road_end = 237.9375573196645
    endpoint_s = 237.93755731966448
    class CarlaMap:
        def __init__(self):
            self.force_miss = False
            self.omit_road = False
            self.first = Waypoint(0, 12)
            self.last = Waypoint(1, endpoint_s)
        def to_opendrive(self):
            return ('<OpenDRIVE><road id="630" length="237.9375573196645"/></OpenDRIVE>'
                    if not self.omit_road else '<OpenDRIVE><road id="631" length="22"/></OpenDRIVE>')
        def get_waypoint_xodr(self, road, lane, s):
            if self.force_miss or s > 12.5:
                return None
            return self.first
        def get_waypoint(self, location, project_to_road=False, lane_type=None):
            return self.first if location.x < 0.5 else self.last

    scene = {'ego_route': [
        {'x': 0., 'y': 0., 'z': 0., 'yaw_deg': 0., 's': 12., 'road_id': 630, 'section_id': 0, 'lane_id': 1},
        {'x': 1., 'y': 0., 'z': 0., 'yaw_deg': 0., 's': endpoint_s, 'road_id': 630, 'section_id': 0, 'lane_id': 1},
    ]}
    fake_map = CarlaMap()
    route, checks = ns['route_waypoints'](Carla, fake_map, scene)
    assert len(route) == len(checks) == 2
    assert checks[0]['pass'] and checks[0]['xodr_found']
    assert checks[1]['pass'] and not checks[1]['xodr_found']
    assert checks[1]['xodr_endpoint_exception_verified'] is True
    assert checks[1]['xodr_lookup_status'] == 'verified_road_end_boundary'
    assert checks[1]['xodr_exact_comparison_status'] == 'not_evaluable_verified_endpoint_exception'
    assert checks[1]['xodr_exact_position_error_m'] is None
    assert abs(checks[1]['xodr_s_minus_road_end_m']) < 1e-12

    def should_reject(bad_scene, bad_map=None):
        try:
            ns['route_waypoints'](Carla, bad_map or CarlaMap(), bad_scene)
        except ValueError:
            return
        raise AssertionError('invalid route accepted despite fail-closed contract')

    bad = copy.deepcopy(scene)
    bad['ego_route'][1]['s'] = road_end + 0.005
    should_reject(bad)
    bad = copy.deepcopy(scene)
    bad['ego_route'][1]['lane_id'] = -1
    should_reject(bad)
    bad = copy.deepcopy(scene)
    bad['ego_route'][1]['yaw_deg'] = 1.0
    should_reject(bad)
    bad = copy.deepcopy(scene)
    bad['ego_route'][1]['x'] = 1.1
    should_reject(bad)
    bad_map = CarlaMap()
    bad_map.omit_road = True
    should_reject(scene, bad_map)
    bad_map = CarlaMap()
    bad_map.force_miss = True
    should_reject(scene, bad_map)

    print('P1_ENDPOINT_MOCK_CARLA_REGRESSION_PASS')
    print('valid road end: xodr_found=False, exception verified=True, route identity=True')
    print('wrong s/lane/yaw/position, missing road metadata, interior XODR miss: FAIL CLOSED')
    print('original route coordinates were not modified')
    print('no live CARLA server was contacted')


def main():
    parser = argparse.ArgumentParser(description="Freeze and verify P1 source before real CARLA audit")
    parser.add_argument("--root", type=Path, help="Project root (default: parent of script directory)")
    parser.add_argument("--python", help="P1 Python binary")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--endpoint-self-test", action="store_true", help="Mock CARLA strict road-end audit, no server access")
    parser.add_argument("--no-carla", action="store_true", help="Only prepare the frozen capsule; skip actual CARLA audit")
    parser.add_argument("--no-pause", action="store_true", help="Do not prompt after the run")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.endpoint_self_test:
        endpoint_self_test()
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
