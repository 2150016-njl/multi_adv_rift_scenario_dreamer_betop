#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a fail-closed P1 map identity manifest from approved provenance JSON/JSONL."""

import argparse
import hashlib
import json
import re
import sys
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


def first_value(obj, prov, names):
    for name in names:
        value = obj.get(name)
        if value not in (None, ""):
            return value
    for name in names:
        value = prov.get(name)
        if value not in (None, ""):
            return value
    return None


def extract_rows(payload):
    found = []

    def walk(node, inherited_town=None):
        if isinstance(node, dict):
            town = (
                node.get("town")
                or node.get("map_name")
                or node.get("town_expected")
                or inherited_town
            )
            prov = node.get("provenance")
            if not isinstance(prov, dict):
                prov = {}

            kind = first_value(node, prov, ("map_input_kind",))
            source_sha = first_value(
                node, prov, ("source_sha256", "source_map_sha256")
            )
            calibrated_sha = first_value(
                node, prov, ("calibrated_map_sha256", "derived_map_sha256")
            )
            xodr_sha = first_value(node, prov, ("opendrive_sha256",))

            if (
                isinstance(town, str)
                and town.strip()
                and kind == "calibrated_carla_map"
                and valid_sha256(source_sha)
                and valid_sha256(calibrated_sha)
                and valid_sha256(xodr_sha)
            ):
                found.append({
                    "town": town,
                    "opendrive_sha256": xodr_sha.lower(),
                    "map_input_kind": kind,
                    "source_sha256": source_sha.lower(),
                    "calibrated_map_sha256": calibrated_sha.lower(),
                })

            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value, town)
        elif isinstance(node, list):
            for value in node:
                walk(value, inherited_town)

    walk(payload)
    return found


def build_manifest(source_paths):
    per_town = {}
    source_provenance = []
    for path_text in source_paths:
        path = Path(path_text).resolve()
        payload = load_json_or_jsonl(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_provenance.append({"path": str(path), "sha256": digest})
        for row in extract_rows(payload):
            key = town_key(row["town"])
            previous = per_town.get(key)
            if previous is not None and previous != row:
                raise ValueError(
                    "ambiguous provenance for town {}: rows disagree; do not choose "
                    "a hash automatically".format(row["town"])
                )
            per_town[key] = row

    if not per_town:
        raise ValueError(
            "no provenance-backed map rows found. Required fields: town, "
            "opendrive_sha256, map_input_kind='calibrated_carla_map', "
            "source_sha256, calibrated_map_sha256. Do not substitute a runtime hash."
        )

    return {
        "schema_version": "p1_map_identity_manifest_v1",
        "provenance_sources": source_provenance,
        "maps": [per_town[key] for key in sorted(per_town)],
    }


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        source = Path(td) / "approved.json"
        source.write_text(json.dumps({
            "town": "Carla/Maps/Town12/Town12",
            "provenance": {
                "map_input_kind": "calibrated_carla_map",
                "source_sha256": "b" * 64,
                "calibrated_map_sha256": "c" * 64,
                "opendrive_sha256": "a" * 64,
            },
        }), encoding="utf-8")
        result = build_manifest([str(source)])
        assert result["schema_version"] == "p1_map_identity_manifest_v1"
        assert len(result["maps"]) == 1
        assert result["maps"][0]["town"].endswith("Town12")
        assert result["maps"][0]["opendrive_sha256"] == "a" * 64

        bad = Path(td) / "untrusted.json"
        bad.write_text(json.dumps({
            "town": "Town12",
            "provenance": {
                "map_input_kind": "runtime_world",
                "source_sha256": "b" * 64,
                "calibrated_map_sha256": "c" * 64,
                "opendrive_sha256": "a" * 64,
            },
        }), encoding="utf-8")
        try:
            build_manifest([str(bad)])
        except ValueError:
            pass
        else:
            raise AssertionError("runtime_world provenance must not be accepted")
    print("P1_MAP_IDENTITY_MANIFEST_BUILDER_SELF_TEST_PASS")


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
        print("{} {}".format(row["town"], row["opendrive_sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
