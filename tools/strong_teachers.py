"""STRONG TEACHERS, JOINTLY, IN A FEEDBACK LOOP (19 Nov 2026; tools/strong_teachers.py) - offline, distant set, LABEL-FREE (own-document labels
for evaluation only). The combined-teacher test failed with weak, correlated proxies; here only the signals that were strong on their own:
  USEFUL   usefulness over time (top 20% +1, bottom half -1; never recurs: abstain)                         offline: +1 right 0.45-0.55
  RCONF    READER-SCORED CONFIRMATION: another document states the pair, in a different text, in a wording the CURRENT reader scores in its
           top 30% (+1; never -1: unconfirmed is not false)                                                  offline: 0.60-0.72 on its top picks
  FIT      FIT WITH EVERYTHING ELSE READ (the relational memory's check, as graph structure): the two names share neighbours in OTHER
           documents (Adamic-Adar over the store graph without the record's own document; top 10% +1, none -1)
The loop: after each block the reader (student: logistic regression on the words between) is retrained from the teachers, and the NEW
reader is the one that scores confirmations in the next round - so a better reader gives better confirmations, which teach a better reader.
Scored each round on the NEXT, unread block. Arms: USEFUL ONLY; USEFUL + RCONF (+1 from either; -1 from USEFUL); ALL THREE (+1 from two of
three or USEFUL; -1 from USEFUL or FIT); AGREEMENT ONLY (+1 only where two teachers agree, -1 where USEFUL or FIT says so).
Usage: python3 tools/strong_teachers.py <distant.json.gz>   env ST_BLOCK (2000), ST_RELQ (0.3)"""
import os, sys, time, types, math, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=1))
R, ND = pm._PM['R'], pm._PM['ND']; BL = int(os.environ.get('ST_BLOCK', 2000)); NB = ND // BL; RQ = float(os.environ.get('ST_RELQ', 0.3))
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); blk = doc // BL; key = [frozenset((r[1], r[2])) for r in R]
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
P(f"{ND} documents in {NB} blocks of {BL}, {len(R)} records (related {y.mean():.3f})  ({time.time() - T0:.0f}s)")
def fit(pos, neg):
    c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))]); return c
def build(upto):
    I = np.where(blk < upto)[0]; pd, nd = collections.defaultdict(set), collections.defaultdict(set); nb = collections.defaultdict(lambda: collections.defaultdict(set))
    byk = collections.defaultdict(list)
    for i in I:
        r = R[i]; pd[key[i]].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0]); nb[r[1]][r[2]].add(r[0]); nb[r[2]][r[1]].add(r[0]); byk[key[i]].append(i)
    return I, pd, nd, nb, byk
def static_votes(I, pd, nd, nb, Nd):
    V = np.zeros((len(I), 3), int); U = np.full(len(I), np.nan); AA = np.zeros(len(I))
    for n_, i in enumerate(I):
        r = R[i]; k = key[i]; own = r[0]
        a_ = len(pd[k] - {own}); nA, nB = len(nd[r[1]] - {own}), len(nd[r[2]] - {own})
        if a_:
            hA, hB = a_ / (nA + 2), a_ / (nB + 2); U[n_] = max(hA * np.log(Nd / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(Nd / (nA + 2)) - (1 - hB) * 0.5)
        na, nb_ = nb[r[1]], nb[r[2]]
        if len(na) > len(nb_): na, nb_ = nb_, na
        if len(na) > 300: continue                                              # hub names: skipped (time)
        s = 0.0
        for c_, ds in na.items():
            if c_ in (r[1], r[2]) or c_ not in nb_: continue
            if (ds - {own}) and (nb_[c_] - {own}): s += 1.0 / math.log(2 + len(nb[c_]))
        AA[n_] = s
    ok = ~np.isnan(U)
    if ok.sum() > 10: hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5); V[ok & (U >= hi), 0] = 1; V[ok & (U <= lo), 0] = -1
    pos = AA > 0
    if pos.sum() > 10: V[AA >= np.quantile(AA[pos], 0.9), 2] = 1
    V[AA == 0, 2] = -1
    return V
def rconf(I, byk, sc, thr):
    out = np.zeros(len(I), int)
    if sc is None: return out
    for n_, i in enumerate(I):
        r = R[i]
        if any(R[j][0] != r[0] and R[j][7] != r[7] and sc[j] >= thr for j in byk[key[i]]): out[n_] = 1
    return out
ARMS = ['USEFUL ONLY', 'USEFUL + RCONF', 'ALL THREE', 'AGREEMENT ONLY']; clf = {a: None for a in ARMS}; curves = collections.defaultdict(list); rs = np.random.RandomState(0)
for t in range(1, NB):
    I, pd, nd, nb, byk = build(t); S_ = rs.permutation(len(I))[:int(os.environ.get('ST_SAMPLE', 100000))]; Is = I[S_]; Inext = np.where(blk == t)[0]; line = []
    V0 = static_votes(Is, pd, nd, nb, t * BL)
    for a in ARMS:
        sc = None; thr = 0.0
        if clf[a] is not None and a != 'USEFUL ONLY':                          # the CURRENT reader scores the confirmations (the feedback loop)
            sc = np.full(len(R), -np.inf); sc[I] = clf[a].decision_function(X[I]); thr = np.quantile(sc[I], 1 - RQ)
        V = V0.copy(); V[:, 1] = rconf(Is, byk, sc, thr); u, c, f = V[:, 0], V[:, 1], V[:, 2]
        if a == 'USEFUL ONLY': p, n = u == 1, u == -1
        elif a == 'USEFUL + RCONF': p, n = (u == 1) | (c == 1), (u == -1) & (c != 1)
        elif a == 'ALL THREE': p = (u == 1) | ((c == 1) & (f == 1)); n = ((u == -1) | (f == -1)) & ~p
        else: p = ((u == 1).astype(int) + (c == 1) + (f == 1)) >= 2; n = ((u == -1) | (f == -1)) & (c != 1) & ~p
        clf[a] = fit(Is[p], Is[n]); s = clf[a].decision_function(X[Inext]); au = roc_auc_score(y[Inext], s); curves[a].append(au)
        o = np.argsort(-s); line.append(f"{a}: +{p.sum()} (right {y[Is][p].mean():.2f}) -{n.sum()} (right {(~y[Is][n]).mean():.2f}) -> next AUC {au:.3f} top5% {y[Inext][o[:len(o) // 20]].mean():.2f}")
        if a == 'ALL THREE':
            line.append("teacher precision (+1): " + ", ".join(f"{nm} {y[Is][V[:, j] == 1].mean():.2f} (n {(V[:, j] == 1).sum()})" for j, nm in enumerate(('USEFUL', 'RCONF', 'FIT')) if (V[:, j] == 1).any()))
    P(f"   read {t * BL:5d} docs || " + " || ".join(line) + f"  ({time.time() - T0:.0f}s)")
P("CURVES (AUC on each unread block): " + " || ".join(f"{a}: " + " ".join(f"{x:.3f}" for x in v) for a, v in curves.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
