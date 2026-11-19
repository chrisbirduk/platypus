import sys, types; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import platypus
from platypus.inference.namemem import _nm_material, _NM_REL
print('relation regex matches:', bool(_NM_REL.match('is a child of')))
it = [dict(d=i // 20) for i in range(100)]
def S(objs): return {'it': it, 'conc': [], 'rel': {(b'Zorquila', b' is a child of '): objs}}
cases = {'UPDATE then replay re-sights the old parent': {b'Mirvolo': [1, 2, 3, 50, 60], b'Talquiro': [40, 41]},
         'NOISE: the new parent seen once': {b'Mirvolo': [1, 2, 3, 50, 60], b'Talquiro': [55]},
         'NO CHANGE': {b'Mirvolo': [1, 2, 3, 50, 60]},
         'UPDATE, no replay': {b'Mirvolo': [1, 2, 3], b'Talquiro': [40, 41]}}
model = types.SimpleNamespace()
for fix in (0, 1):
    cfg = types.SimpleNamespace(inf_nm_update_first=fix, inf_nm_min_seen=2, inf_nm_max=384, inf_cf_wordings=1)
    for nm, objs in cases.items():
        out = _nm_material(model, cfg, S(objs), nmax=50)
        print(f"fix {fix} | {nm:46s} -> {[e['ans'] for e in out] or 'nothing'}")
