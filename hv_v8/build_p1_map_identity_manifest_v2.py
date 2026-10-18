#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a fail-closed P1 map-identity manifest from calibrated provenance JSON/JSONL.

Accepts nested row["provenance"] records such as conditions.jsonl.
Repeated records for one Town are allowed only when all of their OpenDRIVE
hashes agree. Source/calibrated artifact hashes may vary by source record and
are retained as evidence; they are not used to choose between OpenDRIVE hashes.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


def valid_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdefABCDEF" for ch in value)
    )


def town_key(name):
    return str(name).replace("\\", "/").rstrip("/").split("/")[-1].casefold()


def load_json_or_jsonl(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "{}:{} is not valid JSON: {}".format(path, line_number, exc)
                )
        if not rows:
            raise ValueError("{} is empty".format(path))
        return rows


def get_first(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def extract_provenance_records(payload, source_path):
    found = []

    def walk(node, json_path="$"):
        if isinstance(node, dict):
            prov = node.get("provenance")
            if isinstance(prov, dict):
                town = get_first(
                    prov.get("town"), prov.get("map_name"),
                    node.get("town"), node.get("map_name"),
                    node.get("town_expected")
                )
                kind = get_first(prov.get("map_input_kind"), node.get("map_input_kind"))
                # Only an explicitly calibrated CARLA map is trusted.
                if kind == "calibrated_carla_map":
                    row = {
                        "town": town,
                        "map_input_kind": kind,
                        "source_sha256": get_first(
                            prov.get("source_sha256"), prov.get("source_map_sha256"),
                            node.get("source_sha256"), node.get("source_map_sha256")
                        ),
                        "calibrated_map_sha256": get_first(
                            prov.get("calibrated_map_sha256"), prov.get("derived_map_sha256"),
                            node.get("calibrated_map_sha256"), node.get("derived_map_sha256")
                        ),
                        "opendrive_sha256": get_first(
                            prov.get("opendrive_sha256"), node.get("opendrive_sha256")
                        ),
                        "source_file": str(source_path),
                        "record_path": json_path,
                        "record_id": get_first(node.get("record_id"), node.get("scene_id")),
                    }
                    if not isinstance(town, str) or not town.strip():
                        raise ValueError(
                            "{} {} declares calibrated_carla_map but has no town".format(
                                source_path, json_path
                            )
                        )
                    for field in ("source_sha256", "calibrated_map_sha256", "opendrive_sha256"):
                        if not valid_sha256(row[field]):
                            raise ValueError(
                                "{} {} has missing/invalid {}".format(
                                    source_path, json_path, field
                                )
                            )
                    for field in ("source_sha256", "calibrated_map_sha256", "opendrive_sha256"):
                        row[field] = row[field].lower()
                    found.append(row)

            # Avoid walking into provenance again: its parent row was already processed.
            for key, value in node.items():
                if key == "provenance":
                    continue
                if isinstance(value, (dict, list)):
                    walk(value, "{}.{}".format(json_path, key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                if isinstance(value, (dict, list)):
                    walk(value, "{}[{}]".format(json_path, index))

    walk(payload)
    return found


def build_manifest(source_paths):
    by_town = defaultdict(list)
    source_provenance = []

    for path_text in source_paths:
        path = Path(path_text).resolve()
        payload = load_json_or_jsonl(path)
        source_provenance.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
        for row in extract_provenance_records(payload, path):
            by_town[town_key(row["town"])].append(row)

    if not by_town:
        raise ValueError(
            "no trusted calibrated_carla_map provenance rows found. "
            "Use the approved conditions.jsonl or another file containing nested "
            "provenance with town, map_input_kind, source_sha256, "
            "calibrated_map_sha256, and opendrive_sha256."
        )

    maps = []
    conflicts = []
    for key in sorted(by_town):
        rows = by_town[key]
        town_names = sorted(set(row["town"] for row in rows))
        xodr_hashes = sorted(set(row["opendrive_sha256"] for row in rows))
        if len(xodr_hashes) != 1:
            conflicts.append({
                "town_key": key,
                "town_names": town_names,
                "opendrive_sha256_values": xodr_hashes,
                "record_count": len(rows),
                "records": [
                    {
                        "source_file": row["source_file"],
                        "record_path": row["record_path"],
                        "record_id": row["record_id"],
                        "opendrive_sha256": row["opendrive_sha256"],
                        "source_sha256": row["source_sha256"],
                        "calibrated_map_sha256": row["calibrated_map_sha256"],
                    }
                    for row in rows
                ],
            })
            continue

        # Different per-record source/calibrated hashes are retained as evidence.
        # They do not create a conflict when the town's OpenDRIVE hash agrees.
        representative = rows[0]
        maps.append({
            "town": town_names[0],
            "opendrive_sha256": xodr_hashes[0],
            "map_input_kind": "calibrated_carla_map",
            "source_sha256": representative["source_sha256"],
            "calibrated_map_sha256": representative["calibrated_map_sha256"],
            "provenance_record_count": len(rows),
            "provenance_source_files": sorted(set(row["source_file"] for row in rows)),
            "provenance_record_ids": sorted(set(
                str(row["record_id"]) for row in rows if row["record_id"] is not None
            )),
            "source_sha256_values": sorted(set(row["source_sha256"] for row in rows)),
            "calibrated_map_sha256_values": sorted(set(
                row["calibrated_map_sha256"] for row in rows
            )),
        })

    if conflicts:
        details = json.dumps(conflicts, ensure_ascii=False, indent=2)
        raise ValueError(
            "conflicting OpenDRIVE hashes for the same normalized Town; "
            "refusing to select one automatically:\n{}".format(details)
        )

    return {
        "schema_version": "p1_map_identity_manifest_v1",
        "provenance_sources": source_provenance,
        "maps": maps,
    }


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        valid = td / "conditions.jsonl"
        rows = [
            {
                "record_id": "a",
                "provenance": {
                    "town": "Town12",
                    "map_input_kind": "calibrated_carla_map",
                    "source_sha256": "1" * 64,
                    "calibrated_map_sha256": "2" * 64,
                    "opendrive_sha256": "a" * 64,
                },
            },
            {
                "record_id": "b",
                "provenance": {
                    "town": "Carla/Maps/Town12/Town12",
                    "map_input_kind": "calibrated_carla_map",
                    "source_sha256": "3" * 64,
                    "calibrated_map_sha256": "4" * 64,
                    "opendrive_sha256": "a" * 64,
                },
            },
            {
                "record_id": "c",
                "provenance": {
                    "town": "Town15",
                    "map_input_kind": "calibrated_carla_map",
                    "source_sha256": "5" * 64,
                    "calibrated_map_sha256": "6" * 64,
                    "opendrive_sha256": "b" * 64,
                },
            },
        ]
        valid.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        manifest = build_manifest([str(valid)])
        assert len(manifest["maps"]) == 2
        town12 = next(row for row in manifest["maps"] if town_key(row["town"]) == "town12")
        assert town12["opendrive_sha256"] == "a" * 64
        assert town12["provenance_record_count"] == 2
        assert len(town12["source_sha256_values"]) == 2

        conflict = td / "conflict.jsonl"
        conflict.write_text("\n".join(json.dumps(row) for row in [
            rows[0],
            {
                "record_id": "conflict",
                "provenance": {
                    "town": "Town12",
                    "map_input_kind": "calibrated_carla_map",
                    "source_sha256": "7" * 64,
                    "calibrated_map_sha256": "8" * 64,
                    "opendrive_sha256": "c" * 64,
                },
            },
        ]) + "\n", encoding="utf-8")
        try:
            build_manifest([str(conflict)])
        except ValueError as exc:
            assert "conflicting OpenDRIVE hashes" in str(exc)
        else:
            raise AssertionError("conflicting town hashes must fail closed")

        untrusted = td / "untrusted.jsonl"
        untrusted.write_text(json.dumps({
            "provenance": {
                "town": "Town12",
                "map_input_kind": "runtime_world",
                "source_sha256": "1" * 64,
                "calibrated_map_sha256": "2" * 64,
                "opendrive_sha256": "a" * 64,
            }
        }) + "\n", encoding="utf-8")
        try:
            build_manifest([str(untrusted)])
        except ValueError as exc:
            assert "no trusted calibrated_carla_map" in str(exc)
        else:
            raise AssertionError("runtime provenance must not be accepted")
    print("P1_MAP_IDENTITY_MANIFEST_BUILDER_V2_SELF_TEST_PASS")
    print("nested provenance town => supported")
    print("same-town same-OpenDRIVE hash with varying source hashes => accepted with evidence")
    print("same-town conflicting OpenDRIVE hashes => fail closed")
    print("untrusted runtime provenance => rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", help="Approved provenance JSON/JSONL; repeatable")
    parser.add_argument("--output", help="Output manifest JSON path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.source or not args.output:
        parser.error("--source (repeatable) and --output are required")

    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit("REFUSE_OVERWRITE: {}".format(output))
    try:
        manifest = build_manifest(args.source)
    except (OSError, ValueError) as exc:
        raise SystemExit("P1_MANIFEST_BUILD_FAIL: {}".format(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("P1_MAP_IDENTITY_MANIFEST_BUILD_PASS")
    print("output = {}".format(output))
    print("towns = {}".format(len(manifest["maps"])))
    for row in manifest["maps"]:
        print("{} {} (records={})".format(
            row["town"], row["opendrive_sha256"], row["provenance_record_count"]
        ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
