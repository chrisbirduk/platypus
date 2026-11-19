"""COMPOSE FROM CONFIRMED PREMISES (19 Nov 2026; tools/confirm_compose.py) - offline, distant set, label-free (labels for evaluation).
Does composing only from CONFIRMED premises make real-text inference cleaner? Documents arrive in blocks (CC_BLOCK 5000). The wording reader
(self-trained from usefulness on block 0) writes the top CC_TOP of each block as TENTATIVE premises; a later block CONFIRMS a tentative pair
when another document states it with a different record text and a relational wording (the reader's top CC_RELQ of that block). At the end,
A-C pairs are composed through a shared name B (A-B and B-C from different documents; hubs in more than 200 documents skipped) from
(1) every tentative premise, (2) CONFIRMED premises only, (3) one confirmed + one tentative; checkable pairs scored by the distant labels.
Usage: python3 tools/confirm_compose.py <distant_50k.json.gz>   env CC_BLOCK, CC_TOP (0.05), CC_RELQ (0.3)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, LAB, ND = pm._PM['R'], pm._PM['LAB'], pm._PM['ND']; BL = int(os.environ.get('CC_BLOCK', 5000)); TOP = float(os.environ.get('CC_TOP', 0.05)); RQ = float(os.environ.get('CC_RELQ', 0.3))
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); blk = doc // BL; NB = ND // BL; key = [frozenset((r[1], r[2])) for r in R]
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
I0 = np.where(blk == 0)[0]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for i in I0: pd[key[i]].add(R[i][0]); nd[R[i][1]].add(R[i][0]); nd[R[i][2]].add(R[i][0])
def u0(i):
    r = R[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(BL / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(BL / (nA + 2)) - (1 - hB) * 0.5)
U = np.array([u0(i) for i in I0]); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[I0[ok & (U >= hi)], I0[ok & (U <= lo)]]], np.r_[np.ones((ok & (U >= hi)).sum()), np.zeros((ok & (U <= lo)).sum())])
S = c.decision_function(X); ndall = collections.defaultdict(set)
for r in R: ndall[r[1]].add(r[0]); ndall[r[2]].add(r[0])
tent, conf = {}, {}
for t in range(1, NB):
    I = np.where(blk == t)[0]; s = S[I]; thr = np.quantile(s, 1 - RQ); byk = collections.defaultdict(list)
    for i, v in zip(I, s): byk[key[i]].append((i, v))
    for k, (i0, t0) in list(tent.items()):
        if k in conf: continue
        if any(R[j][0] != R[i0][0] and R[j][7] != R[i0][7] and v >= thr for j, v in byk.get(k, [])): conf[k] = i0
    for i in I[np.argsort(-s)[:int(len(I) * TOP)]]:
        if key[i] not in tent: tent[key[i]] = (i, t)
P(f"{ND} documents, {NB} blocks: {len(tent)} tentative premises (right {np.mean([y[i] for i, t in tent.values()]):.3f}), {len(conf)} confirmed (right "
  f"{np.mean([y[i] for i in conf.values()]):.3f})  ({time.time() - T0:.0f}s)")
def compose(prem, conf_set=None, mixed=False):
    nb = collections.defaultdict(list); PR = collections.defaultdict(set)
    for k, i in prem.items(): r = R[i]; nb[r[1]].append((r[2], r[0], k)); nb[r[2]].append((r[1], r[0], k))
    for B_, L in nb.items():
        if len(ndall[B_]) > 200: continue
        for x in range(len(L)):
            for z in range(x + 1, len(L)):
                (A_, dA, kA), (C_, dC, kC) = L[x], L[z]
                if dA == dC or A_ == C_: continue
                if mixed and not ((kA in conf_set) ^ (kC in conf_set)): continue
                PR[frozenset((A_, C_))].add(B_)
    PR = {p: b for p, b in PR.items() if len(p) == 2 and p not in prem}
    ks = [p for p in PR if (lambda u, v: bool(ndall[u] & ndall[v]))(*tuple(p))]; g = np.array([p in LAB for p in ks]); b2 = np.array([len(PR[p]) >= 2 for p in ks])
    return f"{len(PR)} A-C composed, {len(ks)} checkable, right {g.mean() if len(g) else float('nan'):.3f}" + (f"; 2+ bridges {int(b2.sum())} right {g[b2].mean():.3f}" if b2.any() else '')
allp = {k: i for k, (i, t) in tent.items()}
P(f"   from EVERY tentative premise: {compose(allp)}")
P(f"   from CONFIRMED premises only: {compose(conf)}")
P(f"   one confirmed + one tentative: {compose(allp, set(conf), True)}")
P(f"DONE ({time.time() - T0:.0f}s)")
