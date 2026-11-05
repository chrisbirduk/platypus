#!/usr/bin/env bash
# Assemble a release folder for a new GPU build and check it before it is shared.
#
#   make_release.sh BUILD README.md COMMIT.txt [OUT_DIR]
#
# BUILD is the platypus package directory or a single-file build. A package is
# released as a .tar.gz of its .py files plus its single-file export.
#
# Copies the files into releases/<build tag>/, writes SHA256SUMS, checks the
# build compiles, checks NAME strings contain no apostrophes (a standing rule),
# and runs the secret scan. Stops at the first failure.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:?usage: make_release.sh BUILD README.md COMMIT.txt [OUT_DIR]}"; BUILD="${BUILD%/}"
README="${2:?}"; COMMIT="${3:?}"
if [ -d "$BUILD" ]; then
  tag="$(python3 - "$BUILD" <<'PY'
import ast, re, sys
t = ast.parse(open(sys.argv[1] + "/presets.py", encoding="utf-8").read())
v = [n.value.value for n in t.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "BUILD_NAME"][0]
m = re.search(r"\+ (cg2[0-9][a-z_]*) = ", v)
print(m.group(1) if m else "platypus")
PY
)"
else
  tag="$(basename "$BUILD" .py | sed -E 's/^threshold_lean_stagec9_7//; s/_conduit$//')"
fi
OUT="${4:-releases/$tag}"
mkdir -p "$OUT"

if [ -d "$BUILD" ]; then python3 -m compileall -q "$BUILD" >/dev/null && echo "compiles: ok"; else python3 -m py_compile "$BUILD" && echo "compiles: ok"; fi

python3 - "$BUILD" <<'PY'
import ast, os, sys
p = sys.argv[1]
files = [os.path.join(p, "presets.py"), os.path.join(p, "__main__.py")] if os.path.isdir(p) else [p]
NAMES = ("NAME", "BUILD_NAME", "BASE_NAME")
bad = []
for f in files:
    tree = ast.parse(open(f, encoding="utf-8").read())
    bad += [f"{os.path.basename(f)}:{n.lineno}" for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in n.targets)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) and "'" in n.value.value]
if bad:
    sys.exit(f"NAME string contains an apostrophe at {bad}")
print("NAME strings: ok")
PY

if [ -d "$BUILD" ]; then
  tar -czf "$OUT/platypus_$tag.tar.gz" --exclude='__pycache__' -C "$(dirname "$BUILD")" "$(basename "$BUILD")"
  python3 "$HERE/export_single_file.py" "$OUT/platypus_${tag}_single.py"
  cp "$README" "$COMMIT" "$OUT/"
else
  cp "$BUILD" "$README" "$COMMIT" "$OUT/"
fi
( cd "$OUT" && sha256sum -- * > SHA256SUMS )
python3 "$HERE/scan_secrets.py" "$OUT"
echo "release ready: $OUT"
