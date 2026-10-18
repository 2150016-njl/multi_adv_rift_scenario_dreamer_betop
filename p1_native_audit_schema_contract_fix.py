#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely patch the existing server p1_native_audit.py in place.

This preserves the rest of the server's accumulated fixes; it does not replace
an unavailable 2,300-line server file with a reconstructed copy.
"""
import argparse
import ast
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

HELPER_CODE = r'''
def _p1_select_targets_from_snapshot(scene, row):
    """Resolve RIFT-positive targets using the actual snapshot schema.

    Vehicle state: scene['agents'].
    Per-agent RIFT label: scene['rift_labels'][...]['is_interactive'].
    Candidate/targetable cross-checks: diagnostics.rift_labeler.
    Any contradiction fails closed and returns no target actors.
    """
    errors = []
    agents = scene.get("agents")
    labels = scene.get("rift_labels")
    diagnostics = scene.get("diagnostics") or {}
    labeler = diagnostics.get("rift_labeler") or {}

    if not isinstance(agents, list):
        agents = []
        errors.append("snapshot agents is missing or is not a list")
    if not isinstance(labels, list):
        labels = []
        errors.append("snapshot rift_labels is missing or is not a list")

    agent_by_id = {}
    for i, agent in enumerate(agents):
        if not isinstance(agent, dict) or agent.get("actor_id") is None:
            errors.append("agents[%d] is missing actor_id" % i)
            continue
        aid = str(agent["actor_id"])
        if aid in agent_by_id:
            errors.append("duplicate agent actor_id: %s" % aid)
            continue
        agent_by_id[aid] = agent

    label_by_id = {}
    for i, label in enumerate(labels):
        if not isinstance(label, dict) or label.get("actor_id") is None:
            errors.append("rift_labels[%d] is missing actor_id" % i)
            continue
        aid = str(label["actor_id"])
        if aid in label_by_id:
            errors.append("duplicate rift label actor_id: %s" % aid)
            continue
        label_by_id[aid] = label

    agent_ids = set(agent_by_id)
    label_ids = set(label_by_id)
    if agent_ids != label_ids:
        errors.append(
            "agent/label ID coverage mismatch: agent_only=%s label_only=%s"
            % (sorted(agent_ids - label_ids), sorted(label_ids - agent_ids))
        )

    positive_ids = {
        aid for aid, label in label_by_id.items()
        if label.get("is_interactive") is True
    }

    candidate_list = labeler.get("rift_candidate_actor_ids")
    if not isinstance(candidate_list, list):
        candidate_list = []
        errors.append("diagnostics.rift_labeler.rift_candidate_actor_ids missing")
    candidate_ids = {str(value) for value in candidate_list}
    if len(candidate_ids) != len(candidate_list):
        errors.append("duplicate rift_candidate_actor_ids")
    if positive_ids != candidate_ids:
        errors.append(
            "RIFT label/candidate mismatch: positive_labels=%s candidates=%s"
            % (sorted(positive_ids), sorted(candidate_ids))
        )

    targetable_list = labeler.get("targetable_actor_ids")
    if not isinstance(targetable_list, list):
        targetable_list = []
        errors.append("diagnostics.rift_labeler.targetable_actor_ids missing")
    targetable_ids = {str(value) for value in targetable_list}
    if len(targetable_ids) != len(targetable_list):
        errors.append("duplicate targetable_actor_ids")
    if not candidate_ids.issubset(targetable_ids):
        errors.append("RIFT candidates not targetable: %s" % sorted(candidate_ids - targetable_ids))

    unlabeled = labeler.get("unlabeled_rift_candidate_actor_ids")
    if not isinstance(unlabeled, list):
        errors.append("diagnostics.rift_labeler.unlabeled_rift_candidate_actor_ids missing")
    elif unlabeled:
        errors.append("unlabeled RIFT candidates: %s" % unlabeled)

    positive_count = diagnostics.get("num_rift_positive_agents")
    if positive_count is None:
        errors.append("diagnostics.num_rift_positive_agents missing")
    else:
        try:
            if int(positive_count) != len(positive_ids):
                errors.append("RIFT positive count mismatch: diagnostic=%s labels=%d" % (positive_count, len(positive_ids)))
        except (TypeError, ValueError):
            errors.append("diagnostics.num_rift_positive_agents is not an integer")

    if not positive_ids.issubset(agent_ids):
        errors.append("RIFT-positive IDs missing from agents: %s" % sorted(positive_ids - agent_ids))

    declared = row.get("target_actor_ids") if isinstance(row, dict) else None
    if declared is None:
        declared = scene.get("target_actor_ids")
    manifest_match = None
    declared_ids = set()
    if declared is not None:
        if not isinstance(declared, list):
            errors.append("declared target_actor_ids is not a list")
        else:
            declared_ids = {str(value) for value in declared}
            if len(declared_ids) != len(declared):
                errors.append("duplicate declared target_actor_ids")
            manifest_match = declared_ids == positive_ids
            if not manifest_match:
                errors.append("manifest/snapshot target-ID mismatch: manifest=%s snapshot=%s" % (sorted(declared_ids), sorted(positive_ids)))

    status = "pass" if not errors else "fail"
    targets = []
    if status == "pass":
        for agent in agents:
            aid = str(agent["actor_id"])
            if aid in positive_ids:
                resolved = dict(agent)
                resolved["_rift_label"] = dict(label_by_id[aid])
                targets.append(resolved)

    selected_ids = set(positive_ids) if status == "pass" else set()
    selection = {
        "actor_rows": len(agents),
        "manifest_snapshot_ids_match": manifest_match,
        "manifest_target_actor_ids": sorted(declared_ids),
        "matched_target_actor_ids": sorted(positive_ids & agent_ids),
        "missing_target_actor_ids": sorted(positive_ids - agent_ids),
        "selected_target_actor_ids": sorted(selected_ids),
        "snapshot_target_actor_ids": sorted(positive_ids),
        "candidate_actor_ids": sorted(candidate_ids),
        "source": "snapshot.rift_labels.is_interactive",
        "status": status,
        "target_actor_rows": len(targets),
        "validation_errors": errors,
    }
    return agents, selected_ids, targets, selection
'''


def _run_helper_self_test():
    ns = {}
    exec(compile(HELPER_CODE, "<p1-schema-helper>", "exec"), ns)
    resolve = ns["_p1_select_targets_from_snapshot"]
    ids = list(range(6120, 6136)) + [6137, 6138]
    positives = {6121, 6125, 6137, 6138}
    scene = {
        "agents": [{"actor_id": x, "type_id": "vehicle.tesla.model3", "x": float(x)} for x in ids],
        "rift_labels": [{"actor_id": x, "is_interactive": x in positives, "score": 0.5 if x in positives else 0.0} for x in ids],
        "diagnostics": {
            "num_rift_positive_agents": 4,
            "rift_labeler": {
                "rift_candidate_actor_ids": sorted(positives),
                "targetable_actor_ids": ids,
                "unlabeled_rift_candidate_actor_ids": [],
            },
        },
    }
    _, selected, targets, report = resolve(scene, {})
    assert report["status"] == "pass", report
    assert len(targets) == 4 and selected == {str(x) for x in positives}
    _, selected, targets, report = resolve(scene, {"target_actor_ids": [6121]})
    assert report["status"] == "fail" and not selected and not targets
    assert any("manifest/snapshot target-ID mismatch" in x for x in report["validation_errors"])
    bad = json.loads(json.dumps(scene))
    bad["diagnostics"]["rift_labeler"]["rift_candidate_actor_ids"] = [6121, 6125]
    _, selected, targets, report = resolve(bad, {})
    assert report["status"] == "fail" and not selected and not targets
    assert any("RIFT label/candidate mismatch" in x for x in report["validation_errors"])
    print("P1_SCHEMA_CONTRACT_SELF_TEST_PASS")
    print("fixture_agents=18; positive_targets=4")
    print("manifest_conflict_fails_closed=PASS")
    print("label_candidate_conflict_fails_closed=PASS")


def preflight_manifest(manifest_path, record_index=0):
    ns = {}
    exec(compile(HELPER_CODE, "<p1-schema-preflight>", "exec"), ns)
    resolve = ns["_p1_select_targets_from_snapshot"]
    manifest = Path(manifest_path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 0 <= record_index < len(rows):
        raise SystemExit("record index %d out of range; records=%d" % (record_index, len(rows)))
    row = rows[record_index]
    snapshot = Path(row["snapshot_path"])
    raw = snapshot.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    expected_sha = row.get("snapshot_sha256")
    if expected_sha and actual_sha != expected_sha:
        raise SystemExit("SNAPSHOT_SHA256_MISMATCH expected=%s actual=%s" % (expected_sha, actual_sha))
    scene = json.loads(raw)
    _, _, targets, selection = resolve(scene, row)
    report = {
        "manifest": str(manifest), "manifest_records": len(rows), "record_index": record_index,
        "scene_id": row.get("scene_id"), "snapshot_path": str(snapshot),
        "snapshot_sha256_matches_manifest": actual_sha == expected_sha if expected_sha else None,
        "agents_count": len(scene.get("agents", [])) if isinstance(scene.get("agents"), list) else None,
        "rift_labels_count": len(scene.get("rift_labels", [])) if isinstance(scene.get("rift_labels"), list) else None,
        "target_selection": selection,
        "target_actor_ids_resolved": sorted(str(x.get("actor_id")) for x in targets),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if selection["status"] != "pass":
        raise SystemExit(3)
    print("P1_SCHEMA_PREFLIGHT_PASS")


def _function_nodes(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]


def patch_file(target):
    path = Path(target).resolve()
    if not path.is_file():
        raise SystemExit("ERROR: target file not found: %s" % path)
    source = path.read_text(encoding="utf-8")
    if "def _p1_select_targets_from_snapshot(" in source:
        raise SystemExit("ERROR: schema helper already exists; refusing to patch twice. Inspect current file.")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise SystemExit("ERROR: source already has syntax errors: %s" % exc)
    audit_nodes = _function_nodes(tree, "audit_record")
    if len(audit_nodes) != 1:
        raise SystemExit("ERROR: expected one audit_record function, found %d" % len(audit_nodes))
    node = audit_nodes[0]
    lines = source.splitlines(True)
    start, end = node.lineno - 1, node.end_lineno
    starts = [i for i in range(start, end) if re.match(r'^\s*actors\s*=\s*scene\.get\(\s*["\']actors["\']\s*,\s*\[\]\s*\)\s*$', lines[i].rstrip())]
    if len(starts) != 1:
        context = ["%d:%s" % (i + 1, lines[i].rstrip()) for i in range(start, end) if "actors" in lines[i] or "target_actors" in lines[i]][:35]
        raise SystemExit("ERROR: expected one legacy actors assignment in audit_record, found %d. No changes made.\n%s" % (len(starts), "\n".join(context)))
    begin = starts[0]
    loops = [i for i in range(begin + 1, end) if re.match(r'^\s*for\s+a\s+in\s+target_actors\s*:\s*$', lines[i].rstrip())]
    if not loops:
        raise SystemExit("ERROR: target actor loop not found after legacy assignment. No changes made.")
    indent = re.match(r'^(\s*)', lines[begin]).group(1)
    replacement = indent + "actors, selected_ids, target_actors, target_selection = _p1_select_targets_from_snapshot(scene, row)\n"
    new_source = "".join(lines[:begin] + [replacement] + lines[loops[0]:])

    # Update the scene occupancy/background function only.
    tree2 = ast.parse(new_source, filename=str(path))
    bg_nodes = []
    for fn in ast.walk(tree2):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(new_source, fn) or ""
            if "background = [ego_row]" in seg:
                bg_nodes.append((fn, seg))
    if len(bg_nodes) != 1:
        raise SystemExit("ERROR: expected one background = [ego_row] function, found %d. No changes written." % len(bg_nodes))
    bg_node, bg = bg_nodes[0]
    patterns = [
        (r'(?m)^([ \t]*)target_ids\s*=\s*\{\s*a\.get\(["\']actor_id["\']\)\s+for\s+a\s+in\s+target_actors\s*\}\s*$', r'\1target_ids = {str(a.get("actor_id")) for a in target_actors}', "target ID normalization"),
        (r'a\s+for\s+a\s+in\s+scene\.get\(\s*["\']actors["\']\s*,\s*\[\]\s*\)', 'a for a in scene.get("agents", [])', "background agents schema"),
        (r'if\s+a\.get\(["\']actor_id["\']\)\s+not\s+in\s+target_ids', 'if str(a.get("actor_id")) not in target_ids', "background actor ID comparison"),
    ]
    for pattern, repl, label in patterns:
        count = len(re.findall(pattern, bg))
        if count != 1:
            raise SystemExit("ERROR: expected one %s match, found %d. No changes written." % (label, count))
        bg = re.sub(pattern, repl, bg, count=1)
    new_lines = new_source.splitlines(True)
    new_source = "".join(new_lines[:bg_node.lineno - 1] + bg.splitlines(True) + new_lines[bg_node.end_lineno:])

    match = re.search(r'(?m)^def\s+audit_record\s*\(', new_source)
    if not match:
        raise SystemExit("ERROR: cannot locate audit_record for helper insertion. No changes written.")
    new_source = new_source[:match.start()] + HELPER_CODE.strip() + "\n\n\n" + new_source[match.start():]
    try:
        compile(new_source, str(path), "exec")
    except SyntaxError as exc:
        raise SystemExit("ERROR: patched source compile failed at line %s: %s. No changes written." % (exc.lineno, exc.msg))
    if 'scene.get("actors", [])' in new_source:
        raise SystemExit('ERROR: legacy scene.get("actors", []) remains. No changes written.')

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + ".bak_schema_contract_" + stamp)
    shutil.copy2(str(path), str(backup))
    temp = path.with_name(path.name + ".tmp_schema_contract")
    temp.write_text(new_source, encoding="utf-8")
    temp.replace(path)
    print("P1_SCHEMA_CONTRACT_PATCH_APPLIED")
    print("file =", path)
    print("backup =", backup)
    print("target source = snapshot.rift_labels[].is_interactive")
    print("vehicle state = snapshot.agents")
    print("cross-check = diagnostics.rift_labeler.rift_candidate_actor_ids")
    print("background occupancy = snapshot.agents")
    print("manifest target IDs, if supplied, must match; conflicts fail closed")
    print("compile check = PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", nargs="?", help="existing hv_v8/p1_native_audit.py")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight-manifest")
    parser.add_argument("--record-index", type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        _run_helper_self_test()
    if args.preflight_manifest:
        preflight_manifest(args.preflight_manifest, args.record_index)
    if args.target:
        patch_file(args.target)
    if not (args.self_test or args.preflight_manifest or args.target):
        parser.error("provide target, --self-test, or --preflight-manifest")


if __name__ == "__main__":
    main()
