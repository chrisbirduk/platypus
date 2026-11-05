"""Check that python -m platypus builds the same cfg and prints the same banner as the single file's main block.
python3 main_equiv.py ORIG.py PKGDIR"""
import sys, os, io, ast, textwrap, contextlib, importlib, importlib.util as iu, runpy, json
orig, pkgdir = sys.argv[1], sys.argv[2]
FAKE = {'bpc': 1.0, 'forget': 0.0, 'post': [1.0], 'passes': [], 'domains': ['a'], 'timing': {}}
CASES = [{}, {'PLATYPUS_DREAMS': 'random'}, {'PLATYPUS_DREAMS': 'noreplay'}, {'PLATYPUS_DREAMS': '0'}, {'PLATYPUS_CF': '0'},
         {'PLATYPUS_SLEEP': 'fixed'}, {'PLATYPUS_PASSES': '1'}, {'PLATYPUS_RULEFREE': '0'}, {'PLATYPUS_ROUTEFIX': '0'},
         {'LAB_OVR': json.dumps({'n_passes': 2, 'inf_cons': 'mem', 'seed': 3, 'inf_name_strict': 1})}, {'PLATYPUS_DREAMS': 'rem'}, {'PLATYPUS_DREAMS': 'nrem'}]
def snap(cfg): return {k: repr(getattr(cfg, k)) for k in sorted(set(dir(cfg))) if not k.startswith('__')}
spec = iu.spec_from_file_location('o', orig); O = iu.module_from_spec(spec); spec.loader.exec_module(O)
src = open(orig).read(); t = ast.parse(src); mb = [n for n in t.body if isinstance(n, ast.If)][-1]
mcode = textwrap.dedent('\n'.join(src.split('\n')[mb.body[0].lineno - 1:mb.end_lineno]))
sys.path.insert(0, pkgdir); import platypus; R = sys.modules['platypus.runner']
bad = 0
for env in CASES:
    for k in [k for k in os.environ if k.startswith('PLATYPUS_') or k == 'LAB_OVR']: del os.environ[k]
    os.environ.update(env)
    got = {}
    def stub(cfg, tag):
        got[tag] = snap(cfg); return FAKE
    b1 = io.StringIO()
    ns = dict(vars(O)); ns['run'] = lambda c: stub(c, 'o')
    with contextlib.redirect_stdout(b1): exec(mcode, ns)
    b2 = io.StringIO(); R.run, r0 = (lambda c: stub(c, 'p')), R.run
    try:
        with contextlib.redirect_stdout(b2): runpy.run_path(os.path.join(pkgdir, 'platypus', '__main__.py'), run_name='__main__')
    finally: R.run = r0
    same = b1.getvalue() == b2.getvalue() and got['o'] == got['p']
    bad += not same
    print('OK ' if same else 'DIFF', env)
    if not same:
        print([k for k in got['o'] if got['o'].get(k) != got['p'].get(k)][:20]); print(b1.getvalue()[:300]); print(b2.getvalue()[:300])
print('ALL SAME' if not bad else f'{bad} cases differ')
