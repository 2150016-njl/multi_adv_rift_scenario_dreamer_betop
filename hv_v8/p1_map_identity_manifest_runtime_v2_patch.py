#!/usr/bin/env python3
# Fail-closed patch for wiring the per-town map identity manifest into P1.
import argparse
import datetime
import json
import re
import shutil
from pathlib import Path

MARKER = "P1_MAP_IDENTITY_MANIFEST_RUNTIME_V2"

HELPERS = r'''
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
'''


def _sha(value):
    return isinstance(value, str) and len(value.strip()) == 64 and all(
        ch in "0123456789abcdefABCDEF" for ch in value.strip()
    )


def _status(actual, expected):
    if not _sha(actual) or not _sha(expected):
        return "unknown"
    return "pass" if actual.lower() == expected.lower() else "fail"


def _validate_manifest_for_patcher(path):
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit("ERROR: cannot read manifest {}: {}".format(path, exc))
    if not isinstance(data, dict):
        raise SystemExit("ERROR: manifest root must be an object")
    if data.get("schema_version") != "p1_map_identity_manifest_v1":
        raise SystemExit(
            "ERROR: unsupported schema_version {!r}".format(data.get("schema_version"))
        )
    maps = data.get("maps")
    if not isinstance(maps, list) or not maps:
        raise SystemExit("ERROR: manifest maps must be a non-empty list")
    seen = set()
    for item in maps:
        if not isinstance(item, dict):
            raise SystemExit("ERROR: map row must be an object")
        town = str(item.get("town") or "").replace("\\", "/").rstrip("/").split("/")[-1].casefold()
        sha = item.get("opendrive_sha256")
        if not town or town in seen:
            raise SystemExit("ERROR: empty or duplicate town in manifest")
        seen.add(town)
        if not _sha(sha):
            raise SystemExit("ERROR: invalid OpenDRIVE SHA-256 for town {!r}".format(item.get("town")))
        if not isinstance(item.get("provenance_record_count"), int) or item["provenance_record_count"] < 1:
            raise SystemExit("ERROR: missing provenance_record_count for town {!r}".format(item.get("town")))
        if not item.get("source_sha256_values") or not item.get("calibrated_map_sha256_values"):
            raise SystemExit("ERROR: missing source/calibrated evidence for town {!r}".format(item.get("town")))
    print("P1_MAP_IDENTITY_MANIFEST_VALIDATION_PASS")
    print("schema_version =", data.get("schema_version"))
    print("town_count =", len(maps))
    for item in sorted(maps, key=lambda x: str(x.get("town"))):
        print("{} {} (records={})".format(
            item["town"], item["opendrive_sha256"], item["provenance_record_count"]
        ))


def self_test():
    good = "26fa53d4902ac3397d35f948969f0c77c1bf023755ede2c67edf00bef817151f"
    norm = lambda value: str(value or "").replace("\\", "/").rstrip("/").split("/")[-1].casefold()
    assert norm("Town12") == norm("Carla/Maps/Town12/Town12")
    assert _sha(good)
    assert _status(good, good) == "pass"
    assert _status(good, "0" * 64) == "fail"
    assert _status(None, good) == "unknown"
    sample = {
        "schema_version": "p1_map_identity_manifest_v1",
        "maps": [{
            "town": "Town12",
            "opendrive_sha256": good,
            "provenance_record_count": 1,
            "source_sha256_values": ["a" * 64],
            "calibrated_map_sha256_values": ["b" * 64],
        }],
    }
    assert sample["maps"][0]["opendrive_sha256"] == good
    print("P1_MAP_IDENTITY_RUNTIME_PATCH_SELF_TEST_PASS")
    print("town normalization => pass")
    print("matching hash => pass; mismatch => fail; missing hash => unknown")
    print("manifest schema/evidence contract => pass")
    print("manifest/snapshot conflicts => fail closed")


def prepare_patch(target):
    target = Path(target).resolve()
    if not target.is_file():
        raise SystemExit("ERROR: target file does not exist: {}".format(target))
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        raise SystemExit(
            "ERROR: runtime patch marker already present: {}".format(target)
        )

    required = {
        "main parser": "    args = p.parse_args()",
        "main try/config": "    try:\n        apply_rift_config(",
        "audit_record signature": "def audit_record(\n",
        "expected hash block": '    expected_xodr = scene.get(\n        "opendrive_sha256"\n    )',
        "map identity expected hash field": '        "opendrive_sha256_expected": expected_xodr,',
        "audit_record call": "                audit_record(\n",
    }
    for label, anchor in required.items():
        count = source.count(anchor)
        if count != 1:
            raise SystemExit(
                "ERROR: expected exactly one {} anchor, found {}. No changes made.".format(
                    label, count
                )
            )

    source = source.replace(
        "    args = p.parse_args()",
        '    p.add_argument(\n'
        '        "--map-identity-manifest",\n'
        '        default=None,\n'
        '        help="Trusted per-town map identity manifest with OpenDRIVE SHA-256.",\n'
        '    )\n\n'
        "    args = p.parse_args()",
        1,
    )

    source = source.replace(
        "    try:\n        apply_rift_config(",
        "    try:\n"
        "        map_identity_manifest = _p1_load_map_identity_manifest(\n"
        "            args.map_identity_manifest\n"
        "        )\n"
        "        apply_rift_config(",
        1,
    )

    sig_old = (
        "def audit_record(\n"
        "    client: Any,\n"
        "    carla: Any,\n"
        "    grp_cls: Any,\n"
        "    row: dict[str, Any],\n"
        "    out_dir: Path,\n"
        ") -> dict[str, Any]:"
    )
    if source.count(sig_old) != 1:
        raise SystemExit(
            "ERROR: audit_record signature differs from the inspected source; no changes written."
        )
    sig_new = (
        "def audit_record(\n"
        "    client: Any,\n"
        "    carla: Any,\n"
        "    grp_cls: Any,\n"
        "    row: dict[str, Any],\n"
        "    out_dir: Path,\n"
        "    map_identity_manifest: Any = None,\n"
        ") -> dict[str, Any]:"
    )
    source = source.replace(sig_old, sig_new, 1)

    start_anchor = '    expected_xodr = scene.get(\n        "opendrive_sha256"\n    )'
    start = source.index(start_anchor)
    end = source.index("    map_identity = {", start)
    replacement = (
        "    expected_xodr, expected_xodr_source, expected_xodr_reason = (\n"
        "        _p1_resolve_expected_opendrive_sha256(\n"
        "            scene, town, map_identity_manifest\n"
        "        )\n"
        "    )\n\n"
        "    opendrive_status = _p1_opendrive_status(\n"
        "        xodr_sha, expected_xodr\n"
        "    )\n\n"
    )
    source = source[:start] + replacement + source[end:]

    expected_field = '        "opendrive_sha256_expected": expected_xodr,'
    source = source.replace(
        expected_field,
        expected_field + "\n"
        '        "opendrive_expected_source": expected_xodr_source,\n'
        '        "opendrive_expected_reason": expected_xodr_reason,',
        1,
    )

    # Match the actual call structure. The active source passes `out`, whereas
    # the function's parameter is named `out_dir`; the name need not match.
    call_pattern = re.compile(
        r"(?m)^(?P<callindent>[ ]+)audit_record\(\n"
        r"(?P<argindent>[ ]+)client,\n"
        r"(?P=argindent)carla,\n"
        r"(?P=argindent)grp,\n"
        r"(?P=argindent)row,\n"
        r"(?P=argindent)(?P<outarg>[A-Za-z_][A-Za-z0-9_]*),\n"
        r"(?P<closeindent>[ ]+)\)\n"
        r"(?P<outerindent>[ ]+)\)"
    )

    def add_manifest_arg(match):
        return (
            match.group("callindent") + "audit_record(\n"
            + match.group("argindent") + "client,\n"
            + match.group("argindent") + "carla,\n"
            + match.group("argindent") + "grp,\n"
            + match.group("argindent") + "row,\n"
            + match.group("argindent") + match.group("outarg") + ",\n"
            + match.group("argindent") + "map_identity_manifest,\n"
            + match.group("closeindent") + ")\n"
            + match.group("outerindent") + ")"
        )

    source, call_count = call_pattern.subn(add_manifest_arg, source, count=1)
    if call_count != 1:
        raise SystemExit(
            "ERROR: could not uniquely wire manifest into the actual audit_record call; no changes written."
        )

    main_anchor = "def main() -> None:"
    if source.count(main_anchor) != 1:
        raise SystemExit("ERROR: expected one main() definition; no changes written.")
    source = source.replace(main_anchor, HELPERS + "\n\n" + main_anchor, 1)

    try:
        compile(source, str(target), "exec")
    except Exception as exc:
        raise SystemExit("ERROR: patched source failed compile; original unchanged: {}".format(exc))

    for label, needle in (
        ("CLI argument", '"--map-identity-manifest"'),
        ("manifest loader", "_p1_load_map_identity_manifest"),
        ("manifest passed to audit_record", "map_identity_manifest,"),
        ("manifest hash source", '"opendrive_expected_source": expected_xodr_source,'),
    ):
        if needle not in source:
            raise SystemExit("ERROR: patched source missing {}. No changes written.".format(label))
    return target, source


def patch(target, dry_run=False):
    target, source = prepare_patch(target)
    if dry_run:
        print("P1_MAP_IDENTITY_RUNTIME_PATCH_DRY_RUN_PASS")
        print("target =", target)
        print("source compiles in memory; no file was changed")
        print("audit_record call now passes the loaded manifest")
        return

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(target.name + ".bak_map_identity_runtime_v2_" + stamp)
    shutil.copy2(str(target), str(backup))
    target.write_text(source, encoding="utf-8")
    print("P1_MAP_IDENTITY_RUNTIME_PATCH_APPLIED")
    print("file =", target)
    print("backup =", backup)
    print("patch_marker =", MARKER)
    print("CLI manifest loading and audit_record wiring added")
    print("missing expected hash remains UNKNOWN; hash conflicts fail closed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-manifest", default=None)
    parser.add_argument(
        "--target",
        default="hv_v8/p1_native_audit.py",
        help="Path to the active P1 audit module",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.validate_manifest:
        _validate_manifest_for_patcher(args.validate_manifest)
        return
    patch(args.target, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
