"""Undefined names in the package, as pyflakes reports them, minus the names a module declares in its _LATE table (wired in by
platypus/__init__.py after import). compileall cannot see these; a NameError inside a print-and-continue handler shows only as
'unavailable'. Usage: python scripts/check_undefined.py [package dir]. Exit 1 if any are found. Needs pyflakes."""
import ast
import re
import subprocess
import sys


def late_names(path):
    names = set()
    for n in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_LATE" for t in n.targets) and isinstance(n.value, ast.Dict):
            names |= {k.value for k in n.value.keys if isinstance(k, ast.Constant)}
    return names


def main(pkg="platypus"):
    out = subprocess.run([sys.executable, "-m", "pyflakes", pkg], capture_output=True, text=True).stdout
    late, found = {}, set()
    for line in out.splitlines():
        m = re.match(r"(.+?):(\d+):\d+:? undefined name '(.+)'", line)
        if not m: continue
        path, ln, name = m.groups()
        if path not in late: late[path] = late_names(path)
        if name not in late[path]: found.add((path, int(ln), name))
    for path, ln, name in sorted(found): print(f"{path}:{ln}: undefined name '{name}'")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
