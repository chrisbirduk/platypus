"""THE JUDGE'S LIST FILTER (16 Nov 2026; tools/judge_list.py) - offline, GPU write path, distant_50k. The relational judge (inf_rret_sig 'rel')
counts later records linking the pair by words, not lists; its list test misses 'X , Y and Z' (one comma + 'and': 'Microsoft Windows , OS X
and Android'). For each round (writes at visits 2 / 4 / 6, 400 by the rule: the batch 0-200 and the next tier 200-400), the judge's AUC
right vs wrong (relational records of the pair in the two visits after the write) with the current filter and with the wider one (also
lists: a span with ', ' before an 'and' / 'or' between the names, or 'and' / 'or' as the only word). Distant labels score only.
Usage: python3 tools/judge_list.py <distant_50k.json.gz>"""
import sys, types, collections, re
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=400, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
def list2(r):
    if pm._rret_listlike(r): return True
    w = ' ' + r[3] + ' '
    return bool(re.search(r' , .*\b(and|or)\b', w)) or bool(re.search(r'\b(and|or)\b.* , ', w)) or r[3].strip() in ('and', 'or', ', and', ', or', 'as well as', 'and the')
import math as _m
WDF = collections.Counter(r[3] for r in R if 1 <= r[5] <= 8 and not list2(r))   # 16 Nov: how common each wording is (label-free)
NW = sum(WDF.values())
for v in (1, 3, 5):
    pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v); W = dict(pm._PM['rti'])
    want = {frozenset(k): k for k in W}; c1 = collections.Counter(); c2 = collections.Counter(); c3 = collections.Counter(); c4 = collections.Counter(); nl = [0, 0]
    for r in R:
        if h(v) <= r[0] < h(v + 2) and 1 <= r[5] <= 8:
            k = want.get(frozenset((r[1], r[2])))
            if k is None: continue
            if not pm._rret_listlike(r): c1[k] += 1
            if not list2(r): c2[k] += 1; c3[k] += _m.log(NW / (1 + WDF[r[3]])); c4[k] = max(c4[k], _m.log(NW / (1 + WDF[r[3]])))
            nl[0] += not pm._rret_listlike(r); nl[1] += pm._rret_listlike(r) is False and list2(r)
    out = []
    for tr, lo, hi in (('batch 0-200', 0, 200), ('next tier 200-400', 200, 400)):
        ks = [k for k, w in W.items() if lo <= w['rank'] < hi]; y = np.array([W[k]['truth'] for k in ks])
        for nm, c in (('current filter', c1), ('wider filter', c2), ('wider + IDF-weighted count', c3), ('wider + most specific wording', c4)):
            s = np.array([c[k] for k in ks], float); out.append(f"{tr}, {nm}: AUC {_cg_auc(s[y], s[~y]):.3f}, kept (any record) right {y[s > 0].mean() if (s > 0).any() else float('nan'):.3f} ({int((s > 0).sum())}), unconfirmed right {y[s == 0].mean() if (s == 0).any() else float('nan'):.3f}")
        out.append(f"{tr}: right {y.mean():.3f}")
    print(f"ROUND visit {v + 1}: relational records {nl[0]}, of them newly called lists {nl[1]} || " + " | ".join(out), flush=True)
