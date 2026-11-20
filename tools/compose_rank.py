"""WHICH A-C PROPOSALS TO BELIEVE? (20 Nov 2026; tools/compose_rank.py) - offline, distant set. With the SEEDED READER (n labelled records from
the first 80% of documents, + 3 context words each side), reading is no longer the limit end to end; composition is. This ranks the A-C
proposals made through a shared name B (A-B and B-C in different documents, hub names in more than 200 documents skipped) from the reader's
top KEEP share, by:
  bridges            number of distinct names B joining A and C
  best min premise   max over bridges of min(reader score A-B, reader score B-C) - both premises must look relational
  sum min premise    sum over bridges of sigmoid(min premise score)
  hub-weighted       sum over bridges of sigmoid(min premise) / log(documents naming B)   (Adamic-Adar style)
  sum x hub          the two above combined by rank sum
Scored on CHECKABLE proposals (A and C named together in some document) by the distant labels: AUC and precision of the top 100 / 500 / 2000 /
10000, so a writer that takes only the best few can be judged. Labels are used for evaluation and for the seed only.
Usage: python3 tools/compose_rank.py <distant.json.gz>   env CR_SEED (20000), CR_KEEP ('0.25,0.10'), CR_CTX (1)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
T0 = time.time(); P = lambda s: print(s, flush=True)
CTX = int(os.environ.get('CR_CTX', 1))
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0, inf_premises_seed_ctx=CTX))
R, LAB, ND = pm._PM['R'], pm._PM['LAB'], pm._PM['ND']; CX = pm._PM.get('CTX') or {}
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); tr = doc < int(ND * 0.8)
nd = collections.defaultdict(set)
for r in R: nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
tx = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3} " + ' '.join('C_' + w for w in CX.get(id(r), '').split())
X = hv.transform([tx(r) for r in R]); n_ = int(os.environ.get('CR_SEED', 20000))
sd = np.where(tr)[0][np.random.RandomState(0).permutation(int(tr.sum()))[:n_]]
c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[sd], y[sd]); sc = c.decision_function(X)
P(f"{ND} documents, {len(R)} records (premises labelled {y.mean():.3f}); seeded reader {n_} records, context {'on' if CX else 'off'}; held-out AUC "
  f"{roc_auc_score(y[~tr], sc[~tr]):.3f}  ({time.time() - T0:.0f}s)")
sg = lambda v: 1.0 / (1.0 + np.exp(-v))
rk = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable').astype(float)
for frac in [float(v) for v in os.environ.get('CR_KEEP', '0.25,0.10').split(',')]:
    keep = np.argsort(-sc)[:int(len(R) * frac)]; nb = collections.defaultdict(list)
    for i in keep: nb[R[i][1]].append((R[i][2], R[i][0], sc[i])); nb[R[i][2]].append((R[i][1], R[i][0], sc[i]))
    PR = collections.defaultdict(dict); np_ = 0                                 # PR[{A, C}][B] = best min premise score through B
    for i in keep:
        if np_ > 6000000: break
        for B_, C_ in ((R[i][1], R[i][2]), (R[i][2], R[i][1])):
            if len(nd[B_]) > 200: continue
            for A_, dA, sA in nb[B_]:
                if dA != R[i][0] and A_ != C_:
                    k_ = frozenset((A_, C_)); m_ = min(sA, sc[i])
                    if m_ > PR[k_].get(B_, -1e9): PR[k_][B_] = m_
                    np_ += 1
    ks = [p for p in PR if len(p) == 2 and (lambda u, v: bool(nd[u] & nd[v]))(*tuple(p))]
    g = np.array([p in LAB for p in ks])
    F = {'bridges': np.array([len(PR[p]) + 1e-6 * max(PR[p].values()) for p in ks]),
         'best min premise': np.array([max(PR[p].values()) for p in ks]),
         'sum min premise': np.array([sum(sg(v) for v in PR[p].values()) for p in ks]),
         'hub-weighted': np.array([sum(sg(v) / np.log(len(nd[b]) + 2) for b, v in PR[p].items()) for p in ks])}
    F['sum x hub'] = rk(F['sum min premise']) + rk(F['hub-weighted']) + 0.5 * rk(F['best min premise'])
    P(f"KEEP {frac:.0%}: premises {y[keep].mean():.2f}; A-C {len(PR)} proposed, {len(ks)} checkable, right {g.mean():.3f}  ({time.time() - T0:.0f}s)")
    for nm, s in F.items():
        o = np.argsort(-s, kind='stable')
        P(f"   {nm:18s} AUC {roc_auc_score(g, s):.3f} | top " + " / ".join(f"{n}: {g[o[:n]].mean():.3f}" for n in (100, 500, 2000, 10000) if n <= len(o)))
P(f"DONE ({time.time() - T0:.0f}s)")
