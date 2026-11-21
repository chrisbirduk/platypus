"""DOES THE READER FEED INFERENCE? (19 Nov 2026; tools/rcode_reader.py) - offline, distant set, LABEL-FREE (distant labels for evaluation).
The package's real-text inferences (inf_rti_code) come from a code per name compressed from the store's name x name graph of ALL records; the
reader never chooses what goes in. Here the same method (PPMI of the graph, truncated SVD, cosine of two names' codes) on graphs built from:
ALL records | the reader's top 50% | top 25% | top 10% (wording reader self-trained from usefulness). Candidates: pairs named together in
held-out documents (10%) but never in the training graph (the package's set-up); scored by the code; precision of the top 1 / 5 / 10%
against the distant labels, and AUC.
20 Nov: RC_SEED n uses the SEEDED READER instead (n labelled records from training documents, own-document labels, + 3 context words each side,
as kra14), and adds a SOFT graph: every record, edge weight = the reader's probability (summed per pair) instead of a hard cut.
Usage: python3 tools/rcode_reader.py <distant.json.gz>   env RC_DOCS (20000), RC_DIM (128), RC_SEED (0)"""
import os, sys, time, types, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
SEED = int(os.environ.get('RC_SEED', 0)); pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=1 if SEED else 0)); CX = pm._PM.get('CTX') or {}
ND = min(pm._PM['ND'], int(os.environ.get('RC_DOCS', 20000))); R = [r for r in pm._PM['R'] if r[0] < ND]; LAB = pm._PM['LAB']; DIM = int(os.environ.get('RC_DIM', 128))
doc = np.array([r[0] for r in R]); held = (doc * 2654435761 % 1000) < 100; key = [frozenset((r[1], r[2])) for r in R]
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k, h in zip(R, key, held):
    if not h: pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def u_(i):
    r = R[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
tr = np.where(~held)[0]; U = np.array([u_(i) for i in tr]); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split()) for r in R])
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced')
if SEED: sd = tr[np.random.RandomState(0).permutation(len(tr))[:SEED]]; yr = np.array([bool(r[6]) for r in R]); c.fit(X[sd], yr[sd]); P(f"SEEDED READER {SEED} records ({yr[sd].mean():.3f} related), context {'on' if CX else 'off'}")
else: c.fit(X[np.r_[tr[ok & (U >= hi)], tr[ok & (U <= lo)]]], np.r_[np.ones((ok & (U >= hi)).sum()), np.zeros((ok & (U <= lo)).sum())])
s = c.decision_function(X); del X
trainpairs = {key[i] for i in tr}
cand = sorted({key[i] for i in np.where(held)[0] if key[i] not in trainpairs and len(key[i]) == 2}, key=lambda k: sorted(k))
P(f"{ND} documents, {len(R)} records ({held.sum()} in held-out documents); {len(cand)} candidate pairs never in the training graph, labelled {np.mean([k in LAB for k in cand]):.3f}  ({time.time() - T0:.0f}s)")
yc = np.array([k in LAB for k in cand])
pr_ = 1.0 / (1.0 + np.exp(-s))
for nm, q in (('ALL records', 1.0), ('reader top 50%', 0.5), ('reader top 25%', 0.25), ('reader top 10%', 0.10)) + ((('SOFT (reader prob)', -1.0),) if SEED else ()):
    thr = np.quantile(s[tr], 1 - q) if 0 < q < 1 else -np.inf; keep = tr[s[tr] >= thr]
    names = {}; rows, cols, ww = [], [], []
    for i in keep:
        a = names.setdefault(R[i][1], len(names)); b = names.setdefault(R[i][2], len(names)); rows += [a, b]; cols += [b, a]; ww += [pr_[i]] * 2
    n = len(names); M = csr_matrix((np.array(ww) if q < 0 else np.ones(len(rows)), (rows, cols)), shape=(n, n)); M.sum_duplicates()
    if q >= 0: M.data[:] = 1.0
    deg = np.asarray(M.sum(1)).ravel(); tot = deg.sum(); M = M.tocoo()
    pmi = np.log(M.data * tot / (deg[M.row] * deg[M.col] + 1e-9)); m = pmi > 0
    W = csr_matrix((pmi[m], (M.row[m], M.col[m])), shape=(n, n))
    k = min(DIM, n - 1); Z = TruncatedSVD(k, random_state=0).fit_transform(W); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    sc = np.full(len(cand), -9.0); cov = 0
    for j, kk in enumerate(cand):
        a, b = tuple(kk)
        if a in names and b in names: sc[j] = float(Z[names[a]] @ Z[names[b]]); cov += 1
    has = sc > -9; o = np.argsort(-sc); pr = lambda f: yc[o[:max(1, int(len(o) * f))]].mean()
    P(f"   {nm:16s} graph {len(keep):7d} records, {n:6d} names | candidates with both names coded {cov} ({cov / len(cand):.2f}) | AUC (coded) {roc_auc_score(yc[has], sc[has]) if has.any() and yc[has].any() else float('nan'):.3f} | "
      f"precision top 1% {pr(0.01):.3f}, 5% {pr(0.05):.3f}, 10% {pr(0.10):.3f}  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
