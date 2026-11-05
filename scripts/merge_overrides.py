#!/usr/bin/env python3
"""Build the LAB_OVR string for a run.

The single-file builds read option overrides from the LAB_OVR environment
variable, and LAB_OVR REPLACES the build's whole default override set (the
JSON string written into the build after `os.environ.get('LAB_OVR', ...)`).
Passing a small override file on its own would silently drop every default.

This script reads the build's default set, merges an override file on top
(keys starting with "_" are comments and are skipped) and prints the merged
JSON, ready for `export LAB_OVR=...`.

Usage:
    merge_overrides.py BUILD [OVERRIDES.json]       # prints merged JSON
    merge_overrides.py BUILD OVERRIDES.json --diff  # prints what changes

BUILD is the platypus package directory (defaults in platypus/presets.py,
DEFAULT_LAB_OVR) or a single-file build.
"""
import ast
import json
import re
import sys


def build_defaults(build_path):
    import os
    if os.path.isdir(build_path):
        tree = ast.parse(open(os.path.join(build_path, "presets.py"), encoding="utf-8").read())
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "DEFAULT_LAB_OVR" for t in n.targets):
                return json.loads(ast.literal_eval(n.value))
        sys.exit(f"merge_overrides: no DEFAULT_LAB_OVR in {build_path}/presets.py")
    src = open(build_path, encoding="utf-8").read()
    m = re.search(r"os\.environ\.get\(\s*['\"]LAB_OVR['\"]\s*,\s*('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")\s*\)", src)
    if not m:
        sys.exit(f"merge_overrides: no LAB_OVR default found in {build_path}")
    return json.loads(ast.literal_eval(m.group(1)))


def load_overrides(path):
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    base = build_defaults(argv[1])
    ovr = load_overrides(argv[2] if len(argv) > 2 and not argv[2].startswith("--") else None)
    if "--diff" in argv:
        for k, v in ovr.items():
            print(f"{k}: {base.get(k, '(not in defaults)')} -> {v}")
        return
    merged = dict(base)
    merged.update(ovr)
    print(json.dumps(merged, separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv)
