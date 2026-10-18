#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch P1 target actor selection to honor manifest target_actor_ids.

The patch is fail-closed:
- manifest target_actor_ids are primary when present;
- snapshot target_actor_ids are the fallback;
- actor.is_target is the final fallback;
- actor IDs are normalized to strings;
- missing/conflicting IDs are recorded and block p1_pass.
"""
import ast
import datetime
import re
import shutil
import sys
from pathlib import Path


HELPER = r"""
def select_target_actors(row, scene, actors):
    # Select actor rows using the manifest/snapshot target contract.
    if not isinstance(actors, list):
        actors = []

    manifest_ids = row.get("target_actor_ids") or []
    snapshot_ids = scene.get("target_actor_ids") or []

    manifest_keys = {
        str(value) for value in manifest_ids if value is not None
    }
    snapshot_keys = {
        str(value) for value in snapshot_ids if value is not None
    }

    if manifest_keys:
        selected_ids = manifest_keys
        selection_source = "manifest.target_actor_ids"
    elif snapshot_keys:
        selected_ids = snapshot_keys
        selection_source = "snapshot.target_actor_ids"
    else:
        selected_ids = {
            str(actor.get("actor_id"))
            for actor in actors
            if isinstance(actor, dict)
            and actor.get("is_target")
            and actor.get("actor_id") is not None
        }
        selection_source = "snapshot.actors.is_target"

    actor_rows = [
        actor for actor in actors
        if isinstance(actor, dict)
    ]
    target_actors = [
        dict(actor)
        for actor in actor_rows
        if actor.get("actor_id") is not None
        and str(actor.get("actor_id")) in selected_ids
    ]

    matched_ids = {
        str(actor.get("actor_id"))
        for actor in target_actors
        if actor.get("actor_id") is not None
    }
    missing_ids = sorted(selected_ids - matched_ids)

    manifest_snapshot_ids_match = None
    if manifest_keys and snapshot_keys:
        manifest_snapshot_ids_match = (
            manifest_keys == snapshot_keys
        )

    source_conflict = manifest_snapshot_ids_match is False
    selection_status = (
        "pass"
        if selected_ids and not missing_ids and not source_conflict
        else "fail"
    )

    selection = {
        "status": selection_status,
        "source": selection_source,
        "manifest_target_actor_ids": sorted(manifest_keys),
        "snapshot_target_actor_ids": sorted(snapshot_keys),
        "selected_target_actor_ids": sorted(selected_ids),
        "matched_target_actor_ids": sorted(matched_ids),
        "missing_target_actor_ids": missing_ids,
        "manifest_snapshot_ids_match": manifest_snapshot_ids_match,
        "actor_rows": len(actor_rows),
        "target_actor_rows": len(target_actors),
    }
    return target_actors, selection

"""


SELECTION_RE = re.compile(
    r'(?ms)^    actors = scene\.get\(\s*"actors",\s*\[\],\s*\)\s*'
    r'\n\s*selected_ids = set\(\s*scene\.get\(\s*"target_actor_ids",'
    r'\s*\[\],\s*\)\s*\)\s*\n\s*if not selected_ids:\s*\n'
    r'\s*selected_ids = \{\s*a\.get\("actor_id"\)\s*for a in actors'
    r'\s*if a\.get\("is_target"\)\s*\}\s*\n\s*target_actors = \[\s*'
    r'dict\(a\)\s*for a in actors\s*if a\.get\("actor_id"\)\s*'
    r'in selected_ids\s*\]'
)


def patch_source(source):
    if "def select_target_actors(" in source:
        raise RuntimeError(
            "select_target_actors already exists; inspect source manually, no changes made"
        )

    matches = list(SELECTION_RE.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            "expected exactly one target-selection block, found %d; no changes made"
            % len(matches)
        )

    audit_anchor = "def audit_record("
    if source.count(audit_anchor) != 1:
        raise RuntimeError(
            "expected exactly one def audit_record( anchor; no changes made"
        )

    record_anchor = '        "native_rows": native_rows,\n'
    if source.count(record_anchor) != 1:
        raise RuntimeError(
            "expected exactly one native_rows record field; no changes made"
        )

    summary_init_anchor = "    parity_statuses = []\n"
    if source.count(summary_init_anchor) != 1:
        raise RuntimeError(
            "expected exactly one parity_statuses summary initializer; no changes made"
        )

    loop_anchor = "    for r in records:\n        map_ok.append(\n"
    if source.count(loop_anchor) != 1:
        raise RuntimeError(
            "expected exactly one summary record loop anchor; no changes made"
        )

    summary_return_anchor = (
        '    return {\n'
        '        "schema_version": (\n'
        '            "hv_v8_p1_summary_v2"\n'
    )
    if source.count(summary_return_anchor) != 1:
        raise RuntimeError(
            "expected exactly one hv_v8_p1_summary_v2 return; no changes made"
        )

    pass_gate_anchor = (
        "            and all(map_ok)\n"
        "            and bool(coord)\n"
    )
    if source.count(pass_gate_anchor) != 1:
        raise RuntimeError(
            "expected exactly one P1 pass gate anchor; no changes made"
        )

    match = matches[0]
    replacement = (
        '    actors = scene.get(\n'
        '        "actors",\n'
        '        [],\n'
        '    )\n'
        '    target_actors, target_selection = select_target_actors(\n'
        '        row,\n'
        '        scene,\n'
        '        actors,\n'
        '    )'
    )
    patched = source[:match.start()] + replacement + source[match.end():]
    patched = patched.replace(audit_anchor, HELPER + audit_anchor, 1)

    patched = patched.replace(
        record_anchor,
        record_anchor + '        "target_selection": target_selection,\n',
        1,
    )

    patched = patched.replace(
        summary_init_anchor,
        summary_init_anchor + "    target_selection_statuses = []\n",
        1,
    )
    patched = patched.replace(
        loop_anchor,
        '    for r in records:\n'
        '        target_selection_statuses.append(\n'
        '            r.get("target_selection", {}).get("status", "unknown")\n'
        '        )\n'
        '        map_ok.append(\n',
        1,
    )

    patched = patched.replace(
        summary_return_anchor,
        '    target_selection_all_pass = (\n'
        '        bool(target_selection_statuses)\n'
        '        and all(status == "pass" for status in target_selection_statuses)\n'
        '    )\n\n' + summary_return_anchor,
        1,
    )
    patched = patched.replace(
        '        "records": len(records),\n',
        '        "records": len(records),\n'
        '        "target_selection_all_pass": target_selection_all_pass,\n'
        '        "target_selection_pass_count": sum(\n'
        '            status == "pass" for status in target_selection_statuses\n'
        '        ),\n'
        '        "target_selection_fail_count": sum(\n'
        '            status == "fail" for status in target_selection_statuses\n'
        '        ),\n'
        '        "target_selection_unknown_count": sum(\n'
        '            status not in ("pass", "fail")\n'
        '            for status in target_selection_statuses\n'
        '        ),\n',
        1,
    )
    patched = patched.replace(
        pass_gate_anchor,
        '            and all(map_ok)\n'
        '            and target_selection_all_pass\n'
        '            and bool(coord)\n',
        1,
    )

    try:
        ast.parse(patched)
        compile(patched, "<p1_native_audit.py patched>", "exec")
    except Exception as exc:
        raise RuntimeError("patched source failed syntax/compile check: %s" % exc)

    return patched, True


def self_test():
    sample = """from __future__ import annotations
from typing import Any

def audit_record(client: Any, carla: Any, grp_cls: Any, row: dict[str, Any], out_dir: Any):
    scene = {}
    actors = scene.get(
        "actors",
        [],
    )

    selected_ids = set(
        scene.get(
            "target_actor_ids",
            [],
        )
    )
    if not selected_ids:
        selected_ids = {
            a.get("actor_id")
            for a in actors
            if a.get("is_target")
        }

    target_actors = [
        dict(a)
        for a in actors
        if a.get("actor_id")
        in selected_ids
    ]

    native_rows = []
    record = {
        "native_rows": native_rows,
    }
    return record

def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    coord = []
    native_statuses = []
    sedan = []
    map_ok = []
    parity_statuses = []

    interaction_error_count = 0

    for r in records:
        map_ok.append(
            r["map_identity"]["town_pass"]
            and r["map_identity"].get("opendrive_status") == "pass"
        )

    return {
        "schema_version": (
            "hv_v8_p1_summary_v2"
        ),
        "records": len(records),
        "p1_pass": (
            bool(map_ok)
            and all(map_ok)
            and bool(coord)
        ),
    }
"""
    patched, changed = patch_source(sample)
    assert changed
    ns = {}
    exec(compile(patched, "<self-test>", "exec"), ns)
    select = ns["select_target_actors"]

    actors = [
        {"actor_id": 101, "is_target": False},
        {"actor_id": 102, "is_target": False},
        {"actor_id": 999, "is_target": False},
    ]
    selected, info = select(
        {"target_actor_ids": ["101", 102]},
        {"actors": actors},
        actors,
    )
    assert [a["actor_id"] for a in selected] == [101, 102]
    assert info["status"] == "pass"
    assert info["source"] == "manifest.target_actor_ids"

    selected, info = select(
        {"target_actor_ids": [404]},
        {"actors": actors},
        actors,
    )
    assert selected == []
    assert info["status"] == "fail"
    assert info["missing_target_actor_ids"] == ["404"]

    selected, info = select(
        {"target_actor_ids": [101]},
        {"target_actor_ids": [102], "actors": actors},
        actors,
    )
    assert info["status"] == "fail"
    assert info["manifest_snapshot_ids_match"] is False

    print("P1_TARGET_ACTOR_SELECTION_PATCH_SELF_TEST_PASS")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
        return 0

    if len(sys.argv) != 2:
        print(
            "Usage: %s /absolute/path/to/hv_v8/p1_native_audit.py"
            % sys.argv[0],
            file=sys.stderr,
        )
        return 2

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        print("ERROR: target file not found: %s" % target, file=sys.stderr)
        return 2

    source = target.read_text(encoding="utf-8")
    try:
        patched, changed = patch_source(source)
    except RuntimeError as exc:
        print("ERROR: %s. No changes made." % exc, file=sys.stderr)
        return 2

    if not changed:
        print("P1_TARGET_ACTOR_SELECTION_PATCH_NOT_NEEDED")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(
        target.name + ".bak_target_selection_%s" % stamp
    )
    shutil.copy2(str(target), str(backup))
    target.write_text(patched, encoding="utf-8")

    written = target.read_text(encoding="utf-8")
    compile(written, str(target), "exec")

    print("P1_TARGET_ACTOR_SELECTION_PATCH_APPLIED")
    print("file = %s" % target)
    print("backup = %s" % backup)
    print("manifest target_actor_ids are now primary")
    print("actor IDs normalized to strings")
    print("missing/conflicting target IDs are recorded and fail closed")
    print("P1_PYCOMPILE_SOURCE_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
