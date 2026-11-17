"""ADAPTIVE WRITE BUDGET IN THE PACKAGE (16 Nov 2026; tools/adapt_check.py) - offline, the GPU write path through the package's own code
(premises._rcode_step + _rret_judge), distant_50k: four write rounds (visits 2 / 4 / 6 / 8), each batch judged two visits later, with
inf_rti_adapt 0 (200 a round) vs 1 (the next write's size from the last judged batch: blocks of 100 by rank, one block past the last block
with at least AC_STOP of its inferences confirmed by a later relational record, at most 400). Readouts: written and right per round, all
written, judged-kept and their precision. Usage: python3 tools/adapt_check.py <distant_50k.json.gz>   env AC_STOP (0.3)"""
import os, sys, types
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
for ad in (0, 1):
    cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa', inf_rcode_asym=1.0,
                                inf_rcode_selreport=0, inf_rti_max=200, inf_rret=1, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1, inf_rret_sig='rel',
                                inf_rret_list2=1, inf_rti_adapt=ad, inf_rti_adapt_stop=float(os.environ.get('AC_STOP', 0.3)), inf_rti_adapt_max=400)
    pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
    pm._PM['rti'] = {}; pm._PM.pop('rti_budget', None); res = []
    for v in (1, 3, 5, 7):
        before = set(pm._PM['rti']); pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v)
        new = [k for k in pm._PM['rti'] if k not in before]
        for k in new: pm._PM['rti'][k]['ndv'] = h(v); pm._PM['rti'][k]['vis'] = v
        res.append(f"visit {v + 1}: wrote {len(new)} right {np.mean([pm._PM['rti'][k]['truth'] for k in new]) if new else float('nan'):.3f}")
        if v < 7: pm._rret_judge(cfg, [r for r in R if r[0] < h(v + 2)], h(v + 2), v + 2)
    W = pm._PM['rti']; kept = [w for w in W.values() if w.get('judged') is not None and not w.get('flag')]
    print(f"ADAPT {ad}: " + " | ".join(res) + f" || all written {len(W)} right {np.mean([w['truth'] for w in W.values()]):.3f}; judged-kept {len(kept)} right "
          f"{np.mean([w['truth'] for w in kept]):.3f}", flush=True)
