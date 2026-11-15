"""CHOOSE WITH ONE SIGNAL, JUDGE WITH ANOTHER (15 Nov 2026; tools/select_judge.py) - offline, the GPU path's writes.
tools/human_trigger.py: on Re-DocRED writes, restatement and resource-allocation path support separate right from wrong at AUC 0.70-0.75
(human labels), but on the GPU path's own writes only ~0.64 - the path already chose them (code + Adamic-Adar). Here the GPU path's write
step (premises._rcode_step, the asymmetry filter) chooses its 200 writes at document 2/10 by each rule - the code alone, code + AA, code +
RA - and the later reading (documents 2/10 -> 4/10) judges them:
  RESTATEMENT   later documents stating the pair; records with 3 or fewer words between
  PATH          Adamic-Adar / resource allocation on the graph of everything read by 4/10, records stating ANY written pair left out
  COMBINED      rank sums: restatement + short-gap + RA
Per rule: precision at write (distant labels), each judge's AUC, and END TO END - the precision of what reaches the trunk once the
lowest 20% are flagged (and what an oracle would pass).
Usage: SJ_STAGE=write SJ_RULES=<rule> python3 tools/select_judge.py <train_distant.json.gz> (once per rule), then SJ_STAGE=judge
       env SJ_RULES (code,code+aa,code+ra), SJ_OUT (file prefix), TR_ASYM (1.0)"""
import sys, os, types, time, collections, gc
import numpy as np
from scipy.sparse import csr_matrix
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
mk = lambda sel: types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10, inf_rcode_hold=0.1, inf_rcode_k=512, seed=4,
                                       inf_rcode_sel=sel, inf_rcode_asym=float(os.environ.get('TR_ASYM', 1.0)), inf_rcode_selreport=0,
                                       inf_rti_max=200, inf_rret=0, inf_rti_ctl=0.333, inf_rret_q=0.2, inf_rret_ctl=0.5)
cfg0 = mk('code'); pm._load_distant(cfg0); ND = pm._PM['ND']; R = pm._PM['R']; LAB = pm._PM['LAB']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4)); R1 = [r for r in R if r[0] < n1]
import json
SJ = os.environ.get('SJ_OUT', 'select_judge_writes'); RULES = os.environ.get('SJ_RULES', 'code,code+aa,code+ra').split(',')
if os.environ.get('SJ_STAGE', 'write') == 'write':                              # one rule per process (memory: the 512-dim table peaks ~3 GB)
    sel = RULES[0]; pm._PM['rti'] = {}; pm._rcode_step(None, mk(sel), lambda d: 0, [1], R1, 1); W = pm._PM['rti']
    L = [(list(k), bool(W[k]['truth'])) for k in W]; json.dump(L, open(f"{SJ}_{sel}.json", 'w'))
    P(f"rule {sel}: {len(L)} writes at document {n1}, right {np.mean([t for _, t in L]):.3f}  ({time.time() - T0:.0f}s)"); sys.exit(0)
WR = {sel: [(tuple(k), t) for k, t in json.load(open(f"{SJ}_{sel}.json"))] for sel in RULES}
for sel, L in WR.items(): P(f"rule {sel}: {len(L)} writes at document {n1}, right {np.mean([t for _, t in L]):.3f}")
del R1
ALLP = {frozenset(k) for L in WR.values() for k, _ in L}
docs = collections.defaultdict(set); short = collections.Counter(); NI = {}; ia, ib = [], []
for r in R:
    if r[0] >= n2 or r[1] == r[2]: continue
    fk = frozenset((r[1], r[2]))
    if fk in ALLP:
        if r[0] >= n1: docs[fk].add(r[0]); short[fk] += int(r[5] <= 3)
        continue
    ia.append(NI.setdefault(r[1], len(NI))); ib.append(NI.setdefault(r[2], len(NI)))
NN = len(NI); G = csr_matrix((np.ones(len(ia), np.float32), (np.array(ia), np.array(ib))), shape=(NN, NN)); G = G + G.T; G.sum_duplicates(); G.data[:] = 1.0
deg = np.asarray(G.sum(1)).ravel(); iw = np.where(deg > 1, 1.0 / np.log(np.maximum(deg, 2)), 0.0).astype(np.float32); iw1 = (1.0 / np.maximum(deg, 1)).astype(np.float32)
del ia, ib; P(f"later reading to document {n2}: graph {NN} names  ({time.time() - T0:.0f}s)")
rk = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), -np.inf, x))
for sel, L in WR.items():
    y = np.array([t for _, t in L]); fk = [frozenset(k) for k, _ in L]
    nd = np.array([len(docs.get(f, ())) for f in fk], float); ns = np.array([short.get(f, 0) for f in fk], float)
    V = np.full((len(L), 2), np.nan)
    for i, (k, _) in enumerate(L):
        a, c = k
        if a in NI and c in NI:
            m = G[NI[a]].multiply(G[NI[c]]); V[i] = [float(m.multiply(iw).sum()), float(m.multiply(iw1).sum())]
    P(f"RULE {sel}: written right {y.mean():.3f} (n {len(y)}); restated later {np.mean(nd > 0):.2f} (right {np.mean(nd[y] > 0):.2f} / wrong {np.mean(nd[~y] > 0):.2f})")
    for name, s in (('restatement: later documents', nd), ('restatement: short-gap records', ns), ('path: Adamic-Adar (indirect)', V[:, 0]),
                    ('path: resource allocation (indirect)', V[:, 1]), ('COMBINED restatement + short + RA', rk(nd) + rk(ns) + rk(V[:, 1])),
                    ('COMBINED restatement + RA', rk(nd) + rk(V[:, 1]))):
        s2 = np.where(np.isnan(s), -np.inf, s); fl = s2 <= np.quantile(s2, 0.2)
        auc = _cg_auc(s2[y], s2[~y]) if y.any() and (~y).any() else float('nan')
        P(f"   {name:40s} AUC {auc:.3f} | flag lowest 20%: wrong among flagged {1 - y[fl].mean():.2f} | TO THE TRUNK: {(~fl).sum()} kept, right {y[~fl].mean():.3f} "
          f"({y[~fl].sum()} right, {(~y[~fl]).sum()} wrong)")
    P(f"   ORACLE: {y.sum()} right, 0 wrong")
P(f"DONE ({time.time() - T0:.0f}s)")
