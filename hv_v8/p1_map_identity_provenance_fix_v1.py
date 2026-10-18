#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Idempotent, fail-closed P1 OpenDRIVE map-identity provenance patcher.
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PATCH_MARKER = "P1_MAP_IDENTITY_PROVENANCE_FIX_V1"

HELPER_CODE = r'''
# P1_MAP_IDENTITY_PROVENANCE_FIX_V1
P1_MAP_IDENTITY_MANIFEST = None
P1_MAP_IDENTITY_MANIFEST_PATH = None


def _p1_valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def _p1_town_key(name):
    return normalize_map_name(str(name)).strip().casefold()


def load_p1_map_identity_manifest(path):
    # Required schema: p1_map_identity_manifest_v1 with maps/records and
    # town, opendrive_sha256, map_input_kind, source_sha256,
    # calibrated_map_sha256. Hashes must come from approved calibration
    # provenance, never be copied from the runtime result.
    if not path:
        return None
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(
            "cannot read --map-identity-manifest {!r}: {}".format(path, exc)
        )
    if not isinstance(raw, dict):
        raise ContractError("map identity manifest must be a JSON object")
    if raw.get("schema_version") != "p1_map_identity_manifest_v1":
        raise ContractError(
            "map identity manifest schema_version must be "
            "'p1_map_identity_manifest_v1'"
        )

    rows = raw.get("maps", raw.get("records"))
    if isinstance(rows, dict):
        normalized_rows = []
        for key, value in rows.items():
            if not isinstance(value, dict):
                raise ContractError("maps[{}] must be an object".format(key))
            item = dict(value)
            item.setdefault("town", key)
            normalized_rows.append(item)
        rows = normalized_rows
    if not isinstance(rows, list) or not rows:
        raise ContractError("manifest needs a non-empty maps object/list or records list")

    result = {}
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ContractError("manifest row {} must be an object".format(index))
        town = item.get("town") or item.get("map_name")
        if not isinstance(town, str) or not town.strip():
            raise ContractError("manifest row {} is missing town".format(index))
        provenance = item.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}

        kind = item.get("map_input_kind") or provenance.get("map_input_kind")
        source_sha = (
            item.get("source_sha256") or item.get("source_map_sha256")
            or provenance.get("source_sha256")
        )
        calibrated_sha = (
            item.get("calibrated_map_sha256") or item.get("derived_map_sha256")
            or provenance.get("calibrated_map_sha256")
        )
        xodr_sha = item.get("opendrive_sha256") or provenance.get("opendrive_sha256")

        if kind != "calibrated_carla_map":
            raise ContractError(
                "manifest row {} must declare map_input_kind='calibrated_carla_map'".format(index)
            )
        for field, value in (
            ("source_sha256", source_sha),
            ("calibrated_map_sha256", calibrated_sha),
            ("opendrive_sha256", xodr_sha),
        ):
            if not _p1_valid_sha256(value):
                raise ContractError(
                    "manifest row {} has missing/invalid {}".format(index, field)
                )

        key = _p1_town_key(town)
        if key in result:
            raise ContractError("duplicate normalized town in manifest: {!r}".format(town))
        result[key] = {
            "town": town,
            "opendrive_sha256": xodr_sha.lower(),
            "provenance": {
                "map_input_kind": kind,
                "source_sha256": source_sha.lower(),
                "calibrated_map_sha256": calibrated_sha.lower(),
            },
        }
    return result


def resolve_expected_opendrive_sha256(scene, town):
    # An explicitly supplied manifest is authoritative; a missing town must
    # not silently fall back to a snapshot value.
    if P1_MAP_IDENTITY_MANIFEST_PATH:
        if not P1_MAP_IDENTITY_MANIFEST:
            return None, "map_identity_manifest", "manifest_not_loaded"
        entry = P1_MAP_IDENTITY_MANIFEST.get(_p1_town_key(town))
        if entry is None:
            return None, "map_identity_manifest", "town_not_found_in_map_identity_manifest"
        return entry["opendrive_sha256"], "map_identity_manifest", None

    snapshot_hash = scene.get("opendrive_sha256")
    if _p1_valid_sha256(snapshot_hash):
        return snapshot_hash.lower(), "snapshot", None
    return None, "missing", "no_valid_snapshot_hash; provide --map-identity-manifest"


def p1_opendrive_status(actual_sha256, expected_sha256):
    if not _p1_valid_sha256(expected_sha256):
        return "unknown"
    if not _p1_valid_sha256(actual_sha256):
        return "unknown"
    return "pass" if actual_sha256.lower() == expected_sha256.lower() else "fail"
'''


def replace_once(text, pattern, replacement, description, flags=0):
    compiled = re.compile(pattern, flags)
    matches = list(compiled.finditer(text))
    if len(matches) != 1:
        raise RuntimeError("{}: expected exactly one match, found {}".format(description, len(matches)))
    return compiled.sub(replacement, text, count=1)


def patch_source_text(source):
    if PATCH_MARKER in source:
        required = (
            "resolve_expected_opendrive_sha256(scene, town)",
            "--map-identity-manifest",
            '"opendrive_expected_source": expected_xodr_source',
            "p1_opendrive_status(xodr_sha, expected_xodr)",
        )
        missing = [item for item in required if item not in source]
        if missing:
            raise RuntimeError("patch marker exists but patch is incomplete: " + ", ".join(missing))
        compile(source, "<p1_native_audit.py>", "exec")
        return source, "already_patched"

    if "--map-identity-manifest" in source:
        raise RuntimeError("CLI option exists without patch marker; inspect source first")

    source = replace_once(
        source, r"(?m)^def vec2\(",
        HELPER_CODE + "\n\n\ndef vec2(",
        "helper insertion point def vec2",
    )

    multi = (
        r'(?m)^(?P<indent>[ \t]*)expected_xodr[ \t]*=[ \t]*scene\.get\([ \t]*\r?\n'
        r'[ \t]*["\']opendrive_sha256["\'][ \t]*\r?\n[ \t]*\)[ \t]*$'
    )
    single = (
        r'(?m)^(?P<indent>[ \t]*)expected_xodr[ \t]*=[ \t]*scene\.get\([ \t]*'
        r'["\']opendrive_sha256["\'][ \t]*\)'
        r'(?:[ \t]*or[ \t]*scene\.get\([ \t]*["\']opendrive_sha256["\'][ \t]*\))?'
        r'[ \t]*$'
    )
    mm, sm = list(re.finditer(multi, source)), list(re.finditer(single, source))
    if len(mm) + len(sm) != 1:
        raise RuntimeError(
            "expected_xodr assignment: expected one supported scene-only assignment, found {}".format(
                len(mm) + len(sm)
            )
        )
    m = (mm or sm)[0]
    replacement = (
        m.group("indent")
        + "expected_xodr, expected_xodr_source, expected_xodr_reason = "
        "resolve_expected_opendrive_sha256(scene, town)"
    )
    source = source[:m.start()] + replacement + source[m.end():]

    status_pattern = (
        r'(?ms)^(?P<indent>[ \t]*)if not expected_xodr:[ \t]*\r?\n'
        r'[ \t]*opendrive_status[ \t]*=[ \t]*["\']unknown["\'][ \t]*\r?\n'
        r'(?:[ \t]*\r?\n)*'
        r'[ \t]*elif expected_xodr == xodr_sha:[ \t]*\r?\n'
        r'[ \t]*opendrive_status[ \t]*=[ \t]*["\']pass["\'][ \t]*\r?\n'
        r'(?:[ \t]*\r?\n)*'
        r'[ \t]*else:[ \t]*\r?\n'
        r'[ \t]*opendrive_status[ \t]*=[ \t]*["\']fail[" \t]*'
    )
    # Correct the final branch pattern if the source has the normal quoted literal.
    status_pattern = status_pattern.replace(r'["\']fail[" \t]*', r'["\']fail["\'][ \t]*')
    source = replace_once(
        source, status_pattern,
        r'\g<indent>opendrive_status = p1_opendrive_status(xodr_sha, expected_xodr)',
        "OpenDRIVE status block",
    )

    identity_pattern = (
        r'(?m)^(?P<indent>[ \t]*)"opendrive_sha256_expected"[ \t]*:[ \t]*'
        r'expected_xodr,[ \t]*$'
    )
    identity_replacement = (
        r'\g<indent>"opendrive_sha256_expected": expected_xodr,'
        "\n"
        r'\g<indent>"opendrive_expected_source": expected_xodr_source,'
        "\n"
        r'\g<indent>"opendrive_expected_reason": expected_xodr_reason,'
    )
    source = replace_once(source, identity_pattern, identity_replacement, "map_identity expected hash field")

    parse_pattern = r'(?m)^(?P<indent>[ \t]*)args = p\.parse_args\(\)[ \t]*$'
    pm = list(re.finditer(parse_pattern, source))
    if len(pm) != 1:
        raise RuntimeError("expected one 'args = p.parse_args()', found {}".format(len(pm)))
    m = pm[0]
    indent = m.group("indent")
    cli = (
        indent + "p.add_argument(\n"
        + indent + "    \"--map-identity-manifest\",\n"
        + indent + "    default=None,\n"
        + indent + "    help=(\"Trusted calibrated-map provenance JSON; missing town/hash \"\n"
        + indent + "          \"remains UNKNOWN, never PASS.\"),\n"
        + indent + ")\n"
        + indent + "args = p.parse_args()"
    )
    source = source[:m.start()] + cli + source[m.end():]

    try_pattern = (
        r'(?m)^(?P<indent>[ \t]*)try:[ \t]*\r?\n'
        r'(?P<body_indent>[ \t]+)apply_rift_config\('
    )
    tm = list(re.finditer(try_pattern, source))
    if len(tm) != 1:
        raise RuntimeError("expected one main try/apply_rift_config block, found {}".format(len(tm)))
    m = tm[0]
    ti, bi = m.group("indent"), m.group("body_indent")
    init = (
        ti + "try:\n"
        + bi + "global P1_MAP_IDENTITY_MANIFEST, P1_MAP_IDENTITY_MANIFEST_PATH\n"
        + bi + "P1_MAP_IDENTITY_MANIFEST_PATH = args.map_identity_manifest\n"
        + bi + "P1_MAP_IDENTITY_MANIFEST = load_p1_map_identity_manifest(\n"
        + bi + "    args.map_identity_manifest\n"
        + bi + ") if args.map_identity_manifest else None\n"
        + bi + "apply_rift_config("
    )
    source = source[:m.start()] + init + source[m.end():]

    compile(source, "<p1_native_audit.py>", "exec")
    return source, "patched"


def self_test():
    class ContractError(Exception):
        pass

    namespace = {
        "json": json,
        "Path": Path,
        "ContractError": ContractError,
        "normalize_map_name": lambda name: str(name).replace("\\", "/").rstrip("/").split("/")[-1],
    }
    exec(HELPER_CODE, namespace)

    manifest = {
        "schema_version": "p1_map_identity_manifest_v1",
        "maps": [{
            "town": "Carla/Maps/Town12/Town12",
            "opendrive_sha256": "a" * 64,
            "map_input_kind": "calibrated_carla_map",
            "source_sha256": "b" * 64,
            "calibrated_map_sha256": "c" * 64,
        }],
    }
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        good_path = Path(td) / "good.json"
        good_path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = namespace["load_p1_map_identity_manifest"](str(good_path))
        namespace["P1_MAP_IDENTITY_MANIFEST"] = loaded
        namespace["P1_MAP_IDENTITY_MANIFEST_PATH"] = str(good_path)
        expected, source, reason = namespace["resolve_expected_opendrive_sha256"](
            {"opendrive_sha256": "d" * 64}, "Town12"
        )
        assert expected == "a" * 64 and source == "map_identity_manifest" and reason is None
        assert namespace["p1_opendrive_status"]("a" * 64, expected) == "pass"
        assert namespace["p1_opendrive_status"]("e" * 64, expected) == "fail"
        assert namespace["p1_opendrive_status"]("a" * 64, None) == "unknown"
        missing, source, reason = namespace["resolve_expected_opendrive_sha256"](
            {"opendrive_sha256": "d" * 64}, "Town13"
        )
        assert missing is None and source == "map_identity_manifest"
        assert reason == "town_not_found_in_map_identity_manifest"

        bad = json.loads(json.dumps(manifest))
        bad["maps"][0]["map_input_kind"] = "runtime_world"
        bad_path = Path(td) / "bad.json"
        bad_path.write_text(json.dumps(bad), encoding="utf-8")
        try:
            namespace["load_p1_map_identity_manifest"](str(bad_path))
        except ContractError:
            pass
        else:
            raise AssertionError("untrusted provenance must be rejected")

    fixture = '''import json
from pathlib import Path
class ContractError(Exception):
    pass
def normalize_map_name(name):
    return str(name).replace("\\\\\\\\", "/").rstrip("/").split("/")[-1]
def vec2(loc):
    return float(loc.x), float(loc.y)
def audit_record(scene, town, xodr_sha):
    expected_xodr = scene.get(
        "opendrive_sha256"
    )
    if not expected_xodr:
        opendrive_status = "unknown"

    elif expected_xodr == xodr_sha:
        opendrive_status = "pass"

    else:
        opendrive_status = "fail"
    map_identity = {
        "opendrive_sha256_expected": expected_xodr,
    }
    return map_identity
def main():
    p = argparse.ArgumentParser()
    args = p.parse_args()
    try:
        apply_rift_config()
    except ContractError:
        raise
'''
    patched, state = patch_source_text(fixture)
    assert state == "patched"
    compile(patched, "<fixture>", "exec")
    assert "resolve_expected_opendrive_sha256(scene, town)" in patched
    assert "--map-identity-manifest" in patched
    assert '"opendrive_expected_source": expected_xodr_source' in patched
    patched2, state2 = patch_source_text(patched)
    assert state2 == "already_patched" if PATCH_MARKER in patched else False
    assert patched2 == patched
    print("P1_MAP_IDENTITY_PROVENANCE_SELF_TEST_PASS")
    print("manifest match => pass")
    print("hash mismatch => fail")
    print("missing town/hash => unknown")
    print("untrusted provenance => rejected")
    print("patch application => idempotent")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="Path to hv_v8/p1_native_audit.py")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.target:
        parser.error("--target is required unless --self-test is used")

    target = Path(args.target).resolve()
    if not target.is_file():
        raise SystemExit("P1_PATCH_FAIL target file not found: {}".format(target))
    original = target.read_text(encoding="utf-8")
    patched, state = patch_source_text(original)
    if state == "already_patched":
        print("P1_MAP_IDENTITY_PROVENANCE_ALREADY_PATCHED")
        print("file = {}".format(target))
        return 0

    compile(patched, str(target), "exec")
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + ".bak_map_identity_" + stamp)
    n = 1
    while backup.exists():
        backup = target.with_name(target.name + ".bak_map_identity_" + stamp + "_" + str(n))
        n += 1
    shutil.copy2(str(target), str(backup))
    tmp = target.with_name(target.name + ".tmp_map_identity")
    tmp.write_text(patched, encoding="utf-8")
    tmp.replace(target)
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print("P1_MAP_IDENTITY_PROVENANCE_PATCH_APPLIED")
    print("file = {}".format(target))
    print("backup = {}".format(backup))
    print("patch_marker = {}".format(PATCH_MARKER))
    print("missing expected hash remains UNKNOWN / P1 FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
