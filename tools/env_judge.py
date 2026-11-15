"""INVARIANCE ACROSS ENVIRONMENTS AS THE JUDGE (15 Nov 2026; tools/env_judge.py) - offline, the GPU path's writes.
After Lopez-Paz, 'Against All Likelihood' (arXiv 2611.09385): likelihood is neither necessary nor sufficient for truth - a claim repeated
often becomes probable without becoming true; truth shows as INVARIANCE across environments. Our judges count (documents restating a pair:
94% of the current writes are restated, right or wrong) or resample one distribution (bootstrap ~0.50) or slice time (0.61). Here the
environments are made WITHOUT labels: the documents read by 4/10 are clustered by the names they mention (TF-IDF over names -> SVD 64 ->
k-means, K groups). For each write (A, C), chosen at 2/10 as on the GPU (tools/select_judge.py SJ_STAGE=write):
  ENV RESTATE   distinct environments among the later documents (2/10 -> 4/10) restating A-C
  ENV PATH      environments in which a bridge path A-x-C exists INSIDE the environment (records stating any written pair left out)
  ENV PATH / ENV BOTH   the share of environments mentioning both A and C in which they are linked (invariance: holds wherever both occur)
  ENV SPREAD    distinct environments of any record of A or C with a shared neighbour (support spread)
against the count signals of select_judge (later documents; resource allocation). AUC right vs wrong (distant labels) and END TO END: the
precision of what reaches the trunk with the lowest 20% flagged (exactly 20%, ties broken by write order).
Usage: python3 tools/env_judge.py <train_distant.json.gz> <writes .json> [...]   env EJ_K (8,16,32)"""
import sys, os, json, types, time, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfTransformer
sys.path.insert(0, '.')
import platypus.inference.premises as pm
from platypus.inference.compress import _cg_auc
P = lambda *a: print(*a, flush=True); T0 = time.time()
cfg = types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_premises_visits=10)
pm._load_distant(cfg); ND = pm._PM['ND']; R = pm._PM['R']
n1 = int(round(ND * 0.2)); n2 = int(round(ND * 0.4))
WS = [(p.split('/')[-1], [(tuple(k), bool(t)) for k, t in json.load(open(p))]) for p in sys.argv[2:]]
ALLP = {frozenset(k) for _, L in WS for k, _ in L}; WN = {n for p in ALLP for n in p}
# documents x names (label-free environments)
NI = {}; di, ni = [], []
for r in R:
    if r[0] >= n2: continue
    for nm in (r[1], r[2]): di.append(r[0]); ni.append(NI.setdefault(nm, len(NI)))
M = csr_matrix((np.ones(len(di), np.float32), (np.array(di), np.array(ni))), shape=(n2, len(NI))); M.data[:] = 1.0; M.sum_duplicates(); M.data[:] = 1.0
del di, ni
Z = TruncatedSVD(64, random_state=0).fit_transform(TfidfTransformer().fit_transform(M)); Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
P(f"{n2} documents x {len(NI)} names -> 64-dim document codes  ({time.time() - T0:.0f}s)")
rk = lambda x: (lambda z: np.argsort(np.argsort(z, kind='stable'), kind='stable') / max(1, len(z) - 1))(np.where(np.isnan(x), -np.inf, x))
for K in [int(k) for k in os.environ.get('EJ_K', '8,16,32').split(',')]:
    env = MiniBatchKMeans(K, random_state=0, n_init=3, batch_size=4096).fit_predict(Z)
    nbr = collections.defaultdict(lambda: collections.defaultdict(set)); seen = collections.defaultdict(set); rest = collections.defaultdict(set)
    for r in R:
        if r[0] >= n2 or r[1] == r[2]: continue
        e = int(env[r[0]]); fk = frozenset((r[1], r[2]))
        if fk in ALLP:
            if r[0] >= n1: rest[fk].add(e)
            continue
        for a, b in ((r[1], r[2]), (r[2], r[1])):
            if a in WN: nbr[a][e].add(b); seen[a].add(e)
    sz = np.bincount(env, minlength=K)
    P(f"K = {K} environments (documents per environment: min {sz.min()}, median {int(np.median(sz))}, max {sz.max()})  ({time.time() - T0:.0f}s)")
    for nm, L in WS:
        y = np.array([t for _, t in L]); fk = [frozenset(k) for k, _ in L]
        er = np.array([len(rest.get(f, ())) for f in fk], float)
        ep, eb, es, ra = [], [], [], []
        for (a, c), _ in L:
            both = seen[a] & seen[c]; pe = [e for e in both if nbr[a][e] & nbr[c][e]]
            ep.append(len(pe)); eb.append(len(pe) / len(both) if both else np.nan)
            es.append(len({e for e in both if nbr[a][e] or nbr[c][e]}))
        ep, eb, es = map(lambda v: np.array(v, float), (ep, eb, es))
        P(f"  {nm}: {len(L)} writes, right {y.mean():.3f}")
        sig = [('ENV RESTATE: environments restating A-C later', er), ('ENV PATH: environments with a path A-x-C inside', ep),
               ('ENV PATH / ENV BOTH (invariance share)', eb), ('ENV SPREAD: environments mentioning both', es),
               ('COMBINED env restate + env path', rk(er) + rk(ep)), ('COMBINED env path + invariance share', rk(ep) + rk(eb))]
        for name, s in sig:
            s2 = np.where(np.isnan(s), -1.0, s); m = ~np.isnan(s2)
            auc = _cg_auc(s2[y], s2[~y]) if y.any() and (~y).any() else float('nan')
            order = np.lexsort((np.arange(len(s2)), s2)); fl = np.zeros(len(s2), bool); fl[order[:int(round(0.2 * len(s2)))]] = True
            P(f"     {name:48s} AUC {auc:.3f} | flag lowest 20%: wrong among flagged {1 - y[fl].mean():.2f} | TO THE TRUNK right {y[~fl].mean():.3f} ({(~y[~fl]).sum()} wrong)")
P(f"DONE ({time.time() - T0:.0f}s)")
