#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import datetime
from pathlib import Path

NEW_FUNCTION = r'''def select_target_actors(row, scene, actors):
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
'''

OLD_CALL = '''    actors = scene.get(
        "actors",
        [],
    )
    target_actors, target_selection = select_target_actors(
        row,
        scene,
        actors,
    )
'''

NEW_CALL = '''    # Pose/state rows use the actual hv_scene_dataset_v3 schema.
    # Target identity is resolved only from rift_labels.is_interactive.
    actors = scene.get("agents", [])
    target_actors, target_selection = select_target_actors(
        row,
        scene,
        actors,
    )
'''


def run_self_test():
    namespace = {}
    compile(NEW_FUNCTION, "<new_select_target_actors>", "exec")
    exec(NEW_FUNCTION, namespace)
    selector = namespace["select_target_actors"]

    ids = list(range(100, 118))
    agents = [{"actor_id": i, "x": float(i), "y": 0.0} for i in ids]
    positives = {101, 104, 109, 117}
    labels = [
        {"actor_id": i, "is_interactive": i in positives}
        for i in ids
    ]
    scene = {
        "agents": agents,
        "rift_labels": labels,
        "diagnostics": {
            "rift_labeler": {
                "rift_candidate_actor_ids": sorted(positives)
            }
        },
    }

    targets, selection = selector(
        {"target_actor_ids": None}, scene, scene["agents"]
    )
    assert selection["status"] == "pass", selection
    assert selection["source"] == "snapshot.rift_labels.is_interactive"
    assert len(targets) == 4, selection
    assert set(selection["selected_target_actor_ids"]) == {
        str(x) for x in positives
    }
    assert selection["agents_label_ids_match"] is True
    assert selection["diagnostic_candidate_ids_match_labels"] is True

    _, conflict = selector(
        {"target_actor_ids": [999]}, scene, scene["agents"]
    )
    assert conflict["status"] == "fail"
    assert any("manifest.target_actor_ids conflicts" in x
               for x in conflict["validation_errors"])

    conflict_scene = dict(scene)
    conflict_scene["diagnostics"] = {
        "rift_labeler": {"rift_candidate_actor_ids": [101]}
    }
    _, diagnostic_conflict = selector(
        {}, conflict_scene, conflict_scene["agents"]
    )
    assert diagnostic_conflict["status"] == "fail"
    assert diagnostic_conflict["diagnostic_candidate_ids_match_labels"] is False

    missing_agents = [a for a in agents if a["actor_id"] != 104]
    _, missing = selector({}, scene, missing_agents)
    assert missing["status"] == "fail"
    assert "104" in missing["missing_target_actor_ids"]

    legacy_scene = {"actors": [{"actor_id": 999, "is_target": True}]}
    _, no_fallback = selector({}, legacy_scene, [])
    assert no_fallback["status"] == "fail"
    assert no_fallback["selected_target_actor_ids"] == []
    assert no_fallback["source"] == "snapshot.rift_labels.is_interactive"

    print("P1_SCHEMA_CONTRACT_SELF_TEST_PASS")
    print("fixture_agents=18; positive_targets=4")
    print("manifest_conflict_fails_closed=PASS")
    print("diagnostic_candidate_conflict_fails_closed=PASS")
    print("missing_agent_fails_closed=PASS")
    print("no_legacy_actor_fallback=PASS")


def apply_patch(path):
    source = path.read_text(encoding="utf-8")

    if (
        '"snapshot.rift_labels.is_interactive"' in source
        and 'actors = scene.get("agents", [])' in source
        and '"snapshot.actors.is_target"' not in source
    ):
        print("P1_SCHEMA_CONTRACT_PATCH_ALREADY_PRESENT")
        print("file =", path)
        return

    start = source.find("def select_target_actors(row, scene, actors):")
    if start < 0:
        raise SystemExit(
            "ERROR: select_target_actors definition not found; no changes made"
        )
    end = source.find("\ndef audit_record(", start)
    if end < 0:
        raise SystemExit(
            "ERROR: audit_record boundary not found; no changes made"
        )

    if source.count(OLD_CALL) != 1:
        raise SystemExit(
            "ERROR: expected exactly one legacy actors callsite; found "
            + str(source.count(OLD_CALL))
            + ". No changes made."
        )

    prefix = source[:start]
    suffix = source[end + 1:]
    patched = prefix + NEW_FUNCTION.rstrip() + "\n\n" + suffix
    patched = patched.replace(OLD_CALL, NEW_CALL, 1)
    compile(patched, str(path), "exec")

    backup = path.with_name(
        path.name + ".bak_schema_contract_v2_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    backup.write_text(source, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")

    print("P1_SCHEMA_CONTRACT_PATCH_APPLIED")
    print("file =", path)
    print("backup =", backup)
    print("target source = snapshot.rift_labels.is_interactive")
    print("actor state source = snapshot.agents")
    print("manifest/snapshot IDs are constraints, not fallbacks")
    print("diagnostic candidate IDs are cross-checked, not fallback targets")
    print("snapshot contents and route coordinates unchanged")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        default="hv_v8/p1_native_audit.py",
        help="Path to p1_native_audit.py",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run selector-contract tests without editing files",
    )
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
        return

    path = Path(args.file).resolve()
    if not path.is_file():
        raise SystemExit("ERROR: source file does not exist: " + str(path))

    run_self_test()
    apply_patch(path)


if __name__ == "__main__":
    main()
