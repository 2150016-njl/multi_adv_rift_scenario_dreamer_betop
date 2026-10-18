#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Safely ensure hv_v8/p1_native_audit.py has a module-level `import json`.

This is a narrow runtime fix for the observed NameError at json.dumps().
It does not alter route validation, thresholds, manifests, or pass/fail logic.
"""
import ast
import datetime
import re
import shutil
import sys
from pathlib import Path


def has_module_level_json_binding(tree):
    """Return True only for a direct module-level import binding the name json."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound == "json":
                    return True
        elif isinstance(node, ast.ImportFrom):
            # `from json import ...` does not bind the name `json`.
            continue
    return False


def insertion_line(source_lines, tree):
    """Return a zero-based list insertion index preserving docstring/future imports."""
    after_line = 0

    # Preserve a module docstring as the first statement.
    if tree.body:
        first = tree.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), (ast.Str,))
        ):
            after_line = max(after_line, getattr(first, "end_lineno", first.lineno))

    # Future imports must remain before ordinary imports.
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            after_line = max(after_line, getattr(node, "end_lineno", node.lineno))

    # If there is no docstring/future import, preserve a shebang and encoding cookie.
    if after_line == 0:
        for i, line in enumerate(source_lines[:2]):
            stripped = line.strip()
            if i == 0 and stripped.startswith("#!"):
                after_line = i + 1
            elif re.search(r"coding[:=]\s*[-\w.]+", stripped):
                after_line = max(after_line, i + 1)

    return after_line


def patch_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise RuntimeError("target source cannot be parsed before patch: %s" % exc)

    if has_module_level_json_binding(tree):
        return source, False

    lines = source.splitlines(keepends=True)
    idx = insertion_line(lines, tree)

    # Keep the insertion robust for files that do not end in a newline.
    if idx > 0 and lines and not lines[idx - 1].endswith(("\n", "\r")):
        lines[idx - 1] += "\n"

    lines.insert(idx, "import json\n")

    patched = "".join(lines)
    try:
        compile(patched, "<p1_native_audit.py patched>", "exec")
    except Exception as exc:
        raise RuntimeError("patched source failed compile check: %s" % exc)

    return patched, True


def self_test():
    samples = [
        '"""module docstring"""\nfrom __future__ import annotations\n\nimport os\n',
        '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\nfrom __future__ import annotations\n',
        'import os\n\ndef f():\n    return json.dumps({})\n',
    ]
    for i, sample in enumerate(samples):
        patched, changed = patch_source(sample)
        compile(patched, "<self-test-%d>" % i, "exec")
        assert "import json\n" in patched
        assert changed
    already = '"""doc"""\nimport json\n'
    patched, changed = patch_source(already)
    assert not changed and patched == already
    print("P1_JSON_IMPORT_PATCH_SELF_TEST_PASS")


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
        print("P1_JSON_IMPORT_ALREADY_PRESENT")
        print("file = %s" % target)
        print("No changes made.")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = target.with_name(
        target.name + ".bak_json_import_%s" % stamp
    )
    shutil.copy2(str(target), str(backup))
    target.write_text(patched, encoding="utf-8")

    # Verify the actual written file, not only the in-memory string.
    written = target.read_text(encoding="utf-8")
    compile(written, str(target), "exec")

    print("P1_JSON_IMPORT_PATCH_APPLIED")
    print("file = %s" % target)
    print("backup = %s" % backup)
    print("only added a module-level import json")
    print("route logic and P1 acceptance criteria unchanged")
    print("P1_PYCOMPILE_SOURCE_CHECK_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
