"""WHICH NEW FACTS TO WRITE FROM THE COMPRESSED FACT TABLE? A SEEDED SELECTOR (20 Nov 2026; tools/rcode_select.py) - offline, distant set.
The package writes new facts (pairs never named together) ranked by the code (cosine of two names' PPMI + SVD codes), or code + resource
allocation (rank sum). With a small labelled seed now accepted for the reader, the same seed can teach a SELECTOR over write-time features.
Documents are split by hash: TRAIN 80% (the graph), VAL 10% (candidate pairs named together there but never in TRAIN - with labels: the seed
for the selector), TEST 10% (the same, never in TRAIN or VAL - evaluation only). Features per candidate: the code from the plain graph and
from the SOFT graph (edge = the seeded reader's probability), Adamic-Adar, resource allocation, common neighbours, both degrees and their
ratio, and the best two-step path (max over shared names of min(reader probability of the two edges)). Selectors: logistic regression and
gradient-boosted trees trained on VAL. Scored on TEST among candidates with both names in the graph: AUC, top 1 / 5 / 10% right; COVERAGE is
reported separately (a pair with a name never read cannot be inferred by any method).
RS_VALN k (0 = all): the selector learns from only k labelled VAL candidates (how many labels it needs).
Usage: python3 tools/rcode_select.py <distant.json.gz>   env RS_DOCS (20000), RS_DIM (128), RS_SEED (20000), RS_VALN (0)"""
import os, sys, time, types, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=1)); CX = pm._PM.get('CTX') or {}
ND = min(pm._PM['ND'], int(os.environ.get('RS_DOCS', 20000))); R = [r for r in pm._PM['R'] if r[0] < ND]; LAB = pm._PM['LAB']; DIM = int(os.environ.get('RS_DIM', 128))
doc = np.array([r[0] for r in R]); hh = (doc * 2654435761) % 1000; part = np.where(hh < 100, 2, np.where(hh < 200, 1, 0))   # 0 train, 1 val, 2 test
key = [frozenset((r[1], r[2])) for r in R]; y = np.array([bool(r[6]) for r in R]); tr = np.where(part == 0)[0]
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split()) for r in R])
SD = int(os.environ.get('RS_SEED', 20000)); sd = tr[np.random.RandomState(0).permutation(len(tr))[:SD]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[sd], y[sd]); pr = 1.0 / (1.0 + np.exp(-c.decision_function(X))); del X
names = {}; ia, ib, wv = [], [], []
for i in tr:
    a = names.setdefault(R[i][1], len(names)); b = names.setdefault(R[i][2], len(names))
    if a != b: ia += [a, b]; ib += [b, a]; wv += [pr[i], pr[i]]
n = len(names); ia, ib, wv = np.array(ia), np.array(ib), np.array(wv)
G = csr_matrix((np.ones(len(ia)), (ia, ib)), shape=(n, n)); G.sum_duplicates(); G.data[:] = 1.0
Gs = csr_matrix((wv, (ia, ib)), shape=(n, n)); Gs.sum_duplicates()                          # soft: summed reader probability per pair
Gm = csr_matrix((wv, (ia, ib)), shape=(n, n)); Gm = Gm.tocoo()                                # max reader probability per pair (for paths)
EM = {}
for a, b, v in zip(Gm.row, Gm.col, Gm.data):
    if v > EM.get((a, b), -1): EM[(a, b)] = v
def codes(M):
    deg = np.asarray(M.sum(1)).ravel(); tot = deg.sum(); C = M.tocoo(); v = np.log(C.data * tot / (deg[C.row] * deg[C.col] + 1e-9)); m = v > 0
    W = csr_matrix((v[m], (C.row[m], C.col[m])), shape=M.shape); Z = TruncatedSVD(min(DIM, n - 1), random_state=0).fit_transform(W)
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
Z, Zs = codes(G), codes(Gs); deg = np.asarray(G.sum(1)).ravel(); adj = [set(G.indices[G.indptr[i]:G.indptr[i + 1]]) for i in range(n)]
P(f"{ND} documents, {len(R)} records; train graph {n} names; seeded reader {SD} records  ({time.time() - T0:.0f}s)")
trp = {key[i] for i in tr}; valp = {key[i] for i in np.where(part == 1)[0] if len(key[i]) == 2} - trp
tep = {key[i] for i in np.where(part == 2)[0] if len(key[i]) == 2} - trp - valp
def feats(cs):
    F_, ok = [], []
    for k in cs:
        a, b = tuple(k)
        if a not in names or b not in names: ok.append(False); continue
        a, b = names[a], names[b]; cm = adj[a] & adj[b]; ok.append(True)
        aa = sum(1.0 / np.log(max(deg[x], 2)) for x in cm); ra = sum(1.0 / deg[x] for x in cm)
        pth = max((min(EM.get((a, x), 0), EM.get((x, b), 0)) for x in cm), default=0.0)
        F_.append([float(Z[a] @ Z[b]), float(Zs[a] @ Zs[b]), aa, ra, len(cm), np.log1p(deg[a] + deg[b]), np.log1p(min(deg[a], deg[b])), abs(np.log((deg[a] + 1) / (deg[b] + 1))), pth])
    return np.array(F_), np.array(ok)
cv, ct = sorted(valp, key=sorted), sorted(tep, key=sorted)
VN = int(os.environ.get('RS_VALN', 0))
if VN: cv = [cv[j] for j in sorted(np.random.RandomState(5).permutation(len(cv))[:int(VN / 0.3)])]   # about VN coded
Fv, okv = feats(cv); Ft, okt = feats(ct); yv = np.array([k in LAB for k in cv])[okv]; yt = np.array([k in LAB for k in ct])[okt]
P(f"VAL candidates {len(cv)} (coded {okv.mean():.2f}, labelled {yv.mean():.3f}) | TEST candidates {len(ct)} (coded {okt.mean():.2f}, labelled among coded {yt.mean():.3f}, "
  f"among all {np.mean([k in LAB for k in ct]):.3f})  ({time.time() - T0:.0f}s)")
rk = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable') / max(1, len(v) - 1)
sc = StandardScaler().fit(Fv); lr = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced').fit(sc.transform(Fv), yv)
gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Fv, yv)
asy_ = Ft[:, 7] > 1.0; aat = rk(Ft[:, 0]) + rk(Ft[:, 2])                          # the package's kra14 rule: code + AA among |log degree ratio| > 1
lrv = lr.decision_function(sc.transform(Ft)); gbv = gb.predict_proba(Ft)[:, 1]
S = {'code (package)': Ft[:, 0], 'code + RA (package rule)': rk(Ft[:, 0]) + rk(Ft[:, 3]), 'code + AA, asym > 1 (kra14)': np.where(asy_, aat, -9.0),
     'SELECTOR, logistic, asym > 1': np.where(asy_, lrv, -1e9), 'SELECTOR, trees, asym > 1': np.where(asy_, gbv, -9.0), 'soft code': Ft[:, 1], 'Adamic-Adar': Ft[:, 2],
     'best two-step path': Ft[:, 8], 'SEEDED SELECTOR, logistic': lrv, 'SEEDED SELECTOR, trees': gbv}
for nm_, s in S.items():
    o = np.argsort(-s, kind='stable'); top = lambda f: yt[o[:max(1, int(len(o) * f))]].mean()
    P(f"   {nm_:30s} AUC {roc_auc_score(yt, s):.3f} | top 1% {top(0.01):.3f}, 5% {top(0.05):.3f}, 10% {top(0.10):.3f} | top 100 / 200 / 400 {yt[o[:100]].mean():.3f} / {yt[o[:200]].mean():.3f} / {yt[o[:400]].mean():.3f}")
P(f"   asymmetric (|log degree ratio| > 1) share of coded TEST candidates {asy_.mean():.2f}, labelled {yt[asy_].mean():.3f}")
P("   logistic weights (standardised): " + ", ".join(f"{a} {b:+.2f}" for a, b in zip(['code', 'soft code', 'AA', 'RA', 'CN', 'log deg sum', 'log deg min', 'deg ratio', 'path'], lr.coef_[0])))
P(f"DONE ({time.time() - T0:.0f}s)")
