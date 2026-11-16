"""A JUDGE FROM TYPED CONFIRMATIONS (16 Nov 2026; tools/typed_judge.py) - offline, GPU write path, distant_50k. Queue item 31.
On the GPU (cg27a_kty), the memory-feedback readout's label-free signal 'confirmations - contradictions' separated right from wrong typed
inferences at AUC 0.707 (visit 4) and 0.626 (visit 6, the next-tier batch), where the relational judge gave 0.688 and 0.519. That readout
counts later spans 'HEAD <words> X' (HEAD = the inference's typed head): CONFIRM when X is the written tail, CONTRADICT when X is another
name the memory's address would answer for (the address carries the wording key, so only compatible relations count). Offline here, on the
distant records of the two visits after each write (writes at visits 2 / 4 / 6 as on the GPU, 400 a round: the batch 0-200 and the next tier
200-400), per typed inference, over relational records (not lists) whose FIRST name is the head:
  JUDGE        the current relational judge: records naming the pair, either order (the count)
  CC-ANY       (confirm - contradict) / (confirm + contradict), any wording
  CC-WORD      the same, only records with the inference's own typed wording
  CC-CLASS     the same, only records whose second name has the tail's class (a compatible relation by type)
  JUDGE+CC     judge count (ranked) + CC-CLASS (ranked), summed
AUC right vs wrong per tier, and how many inferences each signal covers (any record). Distant labels score only.
Usage: python3 tools/typed_judge.py <distant_50k.json.gz>"""
import sys, types, collections, re
import numpy as np
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
from scipy.stats import rankdata
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4, inf_rcode_sel='code+aa',
                            inf_rcode_asym=1.0, inf_rcode_selreport=0, inf_rti_max=400, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5, inf_rti_typed=1)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']; TY = pm._PM.get('TY', {})
h = lambda v: int(round(ND * min(1.0, (v + 1) / 10)))
def list2(r):
    if pm._rret_listlike(r): return True
    w = ' ' + r[3] + ' '
    return bool(re.search(r' , .*\b(and|or)\b', w)) or bool(re.search(r'\b(and|or)\b.* , ', w)) or r[3].strip() in ('and', 'or', ', and', ', or', 'as well as', 'and the')
norm = lambda w: ' '.join(t for t in w.split() if t not in ('the', 'a', 'an'))
for v in (1, 3, 5):
    pm._PM['rti'] = {}; pm._rcode_step(None, cfg, lambda d: 0, [1], [r for r in R if r[0] < h(v)], v); W = dict(pm._PM['rti'])
    pair = {frozenset(k): k for k in W}; heads = collections.defaultdict(list)
    ST = {frozenset((r[1], r[2])) for r in R if r[0] < h(v) and 1 <= r[5] <= 8}   # 16 Nov (queue 38): pairs the store already links - containment-aware
    for k, w in W.items():
        if w.get('tw'): heads[w['th']].append(k)
    jc = collections.Counter(); cc = {nm: collections.defaultdict(lambda: [0, 0]) for nm in ('ANY', 'WORD', 'CLASS', 'ANY-H', 'WORD-H', 'CLASS-H')}
    for r in R:
        if not (h(v) <= r[0] < h(v + 2) and 1 <= r[5] <= 8) or list2(r): continue
        k = pair.get(frozenset((r[1], r[2])))
        if k is not None: jc[k] += 1
        for k2 in heads.get(r[1], ()):
            w = W[k2]; hit = r[2] == w['tt']
            lk = (not hit) and frozenset((r[2], w['tt'])) in ST                #   'X in Z' with Z linked to the tail (Y in Z / Z in Y) is not a contradiction
            for nm, ok in (('ANY', True), ('WORD', norm(r[3]) == norm(w['tw'])), ('CLASS', TY.get(r[2], 'X') == TY.get(w['tt'], 'X'))):
                if ok: cc[nm][k2][0 if hit else 1] += 1
                if ok and not lk: cc[nm + '-H'][k2][0 if hit else 1] += 1
    out = []
    for tr, lo, hi in (('batch 0-200', 0, 200), ('next tier 200-400', 200, 400)):
        ks = [k for k, w in W.items() if lo <= w['rank'] < hi and w.get('tw')]; y = np.array([W[k]['truth'] for k in ks])
        sc = {'JUDGE': np.array([jc[k] for k in ks], float)}
        for nm in ('ANY', 'WORD', 'CLASS', 'ANY-H', 'WORD-H', 'CLASS-H'):
            sc['CC-' + nm] = np.array([(cc[nm][k][0] - cc[nm][k][1]) / max(1, sum(cc[nm][k])) for k in ks], float)
        sc['JUDGE+CC'] = rankdata(sc['JUDGE']) + rankdata(sc['CC-CLASS']); sc['JUDGE+CC-H'] = rankdata(sc['JUDGE']) + rankdata(sc['CC-ANY-H'])
        cov = {'JUDGE': np.mean([jc[k] > 0 for k in ks])}
        for nm in ('ANY', 'WORD', 'CLASS', 'ANY-H', 'WORD-H', 'CLASS-H'): cov['CC-' + nm] = np.mean([sum(cc[nm][k]) > 0 for k in ks])
        cov['JUDGE+CC'] = np.mean([jc[k] > 0 or sum(cc['CLASS'][k]) > 0 for k in ks]); cov['JUDGE+CC-H'] = np.mean([jc[k] > 0 or sum(cc['ANY-H'][k]) > 0 for k in ks])
        out.append(f"{tr} (right {y.mean():.3f}, n {len(ks)}): " + " | ".join(f"{nm} AUC {_cg_auc(s[y], s[~y]):.3f} (covers {cov[nm]:.2f})" for nm, s in sc.items()))
    print(f"ROUND visit {v + 1}: " + " || ".join(out), flush=True)
