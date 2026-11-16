"""ARE THE 'WRONG' INFERENCES WRONG? (16 Nov 2026; tools/label_check.py) - the GPU path's 200 writes at visit 2 (distant_50k); for writes the
distant labels call WRONG but a later RELATIONAL record states (the judge would keep them), print the records, for reading by hand.
Usage: python3 tools/label_check.py <distant.json.gz>"""
import sys, types, collections
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; PRD = pm._PM['PRD']
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(1)], 1); W = pm._PM['rti']
isrel = lambda r: 1 <= r[5] <= 8 and not pm._rret_listlike(r)
want = {frozenset(k): k for k, w in W.items() if not w['truth']}; out = collections.defaultdict(list)
for r in R:
    if h(1) <= r[0] < h(3) and isrel(r):
        k = want.get(frozenset((r[1], r[2])))
        if k: out[k].append(r[7])
print(f"wrong by the distant labels: {sum(1 for w in W.values() if not w['truth'])} of {len(W)}; of them stated later in a relational record: {len(out)}")
for k, sp in list(out.items())[:40]:
    print(f"  {k[0]} ~ {k[1]} | Wikidata both ways: {sorted(PRD.get(k, set()) | PRD.get((k[1], k[0]), set()))} | " + " || ".join(collections.Counter(sp).most_common(2)[i][0] for i in range(min(2, len(set(sp))))))
