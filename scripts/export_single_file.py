"""Write the platypus package out as one self-contained script (for environments that take a single file).
usage: python3 scripts/export_single_file.py [OUT.py]   (default: dist/platypus_single.py)
The modules are concatenated in load order with the package imports removed, so every module-level name lives in
one namespace again; presets and the run of __main__.py follow under `if __name__ == "__main__":`."""
import ast, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, 'platypus')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'dist', 'platypus_single.py')

def module_order():
    t = ast.parse(open(os.path.join(PKG, '__init__.py')).read())
    for n in t.body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'MODULES': return [c.value for c in n.value.elts]
    raise SystemExit('MODULES not found in platypus/__init__.py')

def strip(path):
    """the module's code without its docstring, its imports and its _LATE table; returns (std_imports, code)"""
    src = open(path).read(); t = ast.parse(src); L = src.split('\n'); drop = set(); std = []
    for i, n in enumerate(t.body):
        rng = range(n.lineno - 1, n.end_lineno)
        if i == 0 and isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str): drop.update(rng)
        elif isinstance(n, ast.ImportFrom) and (n.module or '').startswith('platypus'): drop.update(rng)
        elif isinstance(n, ast.Import) and any(a.name == 'platypus' for a in n.names): drop.update(rng)
        elif isinstance(n, ast.Import) and n.col_offset == 0 and i < 40 and all(ast.unparse(m) for m in [n]):
            std.append(ast.unparse(n)); drop.update(rng)
        elif isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == '_LATE': drop.update(rng)
    code = '\n'.join(l for j, l in enumerate(L) if j not in drop).strip('\n')
    # the comment line introducing _LATE
    code = re.sub(r"\n# names this module uses at run time from modules that load after it;[^\n]*", '', code)
    return std, code

order = module_order()
init_src = open(os.path.join(PKG, '__init__.py')).read()
header = '\n'.join(l for l in init_src.split('\n') if l.startswith('#'))
parts, std_all = [], []
for m in order:
    std, code = strip(os.path.join(PKG, *m.split('.')) + '.py')
    for s in std:
        if s not in std_all: std_all.append(s)
    parts.append(f"# {'=' * 20} platypus.{m} {'=' * 20}\n{code}")
_, presets = strip(os.path.join(PKG, 'presets.py'))
_, main = strip(os.path.join(PKG, '__main__.py'))
main = '\n'.join(('    ' + l if l.strip() else l) for l in main.split('\n'))
os_first = [s for s in std_all if s == 'import os as _os0']
rest = [s for s in std_all if s != 'import os as _os0']
out = (header + '\n# Single-file export of the platypus package (scripts/export_single_file.py). Edit the package, not this file.\n'
       + '\n'.join(os_first) + "\n_os0.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')\n"
       + '\n'.join(rest) + '\n\n' + '\n\n'.join(parts).replace("\nimport os as _os0\n_os0.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')   # less fragmentation (only takes effect before CUDA starts)\n", '\n')
       + '\n\n_label_regions()\n\n' + presets + '\n\nif __name__ == "__main__":\n' + main + '\n')
os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
open(OUT, 'w').write(out)
compile(out, OUT, 'exec')
print(f'wrote {OUT}: {len(out.splitlines())} lines from {len(order)} modules')
