#!/usr/bin/env bash
# Assemble a release folder for a new GPU build and check it before it is shared.
#
#   make_release.sh BUILD.py README.md COMMIT.txt [OUT_DIR]
#
# Copies the files into releases/<build tag>/, writes SHA256SUMS, checks the
# build compiles, checks NAME strings contain no apostrophes (a standing rule),
# and runs the secret scan. Stops at the first failure.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${1:?usage: make_release.sh BUILD.py README.md COMMIT.txt [OUT_DIR]}"
README="${2:?}"; COMMIT="${3:?}"
tag="$(basename "$BUILD" .py | sed -E 's/^threshold_lean_stagec9_7//; s/_conduit$//')"
OUT="${4:-releases/$tag}"
mkdir -p "$OUT"

python3 -m py_compile "$BUILD" && echo "compiles: ok"

python3 - "$BUILD" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
bad = [n.lineno for n in ast.walk(tree)
       if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "NAME" for t in n.targets)
       and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str) and "'" in n.value.value]
if bad:
    sys.exit(f"NAME string contains an apostrophe at line(s) {bad}")
print("NAME strings: ok")
PY

cp "$BUILD" "$README" "$COMMIT" "$OUT/"
( cd "$OUT" && sha256sum -- * > SHA256SUMS )
python3 "$HERE/scan_secrets.py" "$OUT"
echo "release ready: $OUT"
