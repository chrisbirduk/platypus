"""END-TO-END REAL TEXT, OFFLINE (19 Nov 2026; tools/e2e_reader.py) - the steps of kra11's readout (inf_premises_e2e) on the distant set, label-free
(labels for evaluation only): READ every record -> the READER keeps its top 25 / 10 / 2% (policies: random, nearer, usefulness, wording reader
self-trained from usefulness, wording + usefulness) -> COMPOSE A-C through a shared name B (A-B and B-C kept in different documents; hub names
in more than 200 documents skipped) -> score: premises kept (own-document label), A-C proposals CHECKABLE (A and C named together in some
document) right by the distant labels, and with 2+ bridges.
20 Nov: E2_SEED '2000,20000' adds SEEDED READER policies (n labelled records from the first 80% of documents, weight E2_W 3, alone and with the
label-free pseudo-labels); the seed's own records are among those kept and scored (a small leak; A-C proposals are judged across documents).
Usage: python3 tools/e2e_reader.py <distant.json.gz>   env E2_DOCS (all), E2_SEED (''), E2_W (3)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, LAB, ND = pm._PM['R'], pm._PM['LAB'], pm._PM['ND']; ND = min(ND, int(os.environ.get('E2_DOCS', ND))); R = [r for r in R if r[0] < ND]
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R])
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r in R: pd[frozenset((r[1], r[2]))].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def use_(r):
    a_ = len(pd[frozenset((r[1], r[2]))]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
U = np.array([use_(r) for r in R]); tr = doc < int(ND * 0.8); sn = np.where(tr & ~np.isnan(U))[0]
hi, lo = np.quantile(U[sn], 0.8), np.quantile(U[sn], 0.5); ps, ng = sn[U[sn] >= hi], sn[U[sn] <= lo]
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[ps, ng]], np.r_[np.ones(len(ps)), np.zeros(len(ng))]); sw = c.decision_function(X)
rk = lambda v: np.argsort(np.argsort(v, kind='stable'), kind='stable').astype(float)
POL = {'random': np.random.RandomState(0).rand(len(R)), 'nearer': -np.array([r[5] for r in R], float) + 1e-3 * np.random.RandomState(1).rand(len(R)),
       'usefulness': np.nan_to_num(U, nan=-9.0), 'WORDING READER': sw, 'WORDING + USEFULNESS': np.where(np.isnan(U), rk(sw) * 0.5, rk(sw) + rk(np.nan_to_num(U, nan=-9.0)))}
for n_ in [int(v) for v in os.environ.get('E2_SEED', '').split(',') if v.strip()]:   # 20 Nov: SEEDED READERS - n labelled records from training documents
    sd = np.where(tr)[0][np.random.RandomState(0).permutation(int(tr.sum()))[:n_]]; W_ = float(os.environ.get('E2_W', 3))   # (weight E2_W) + the label-free
    c2 = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c2.fit(X[sd], y[sd]); POL[f'SEED {n_} ONLY'] = c2.decision_function(X)   # pseudo-labels
    c2 = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c2.fit(X[np.r_[sd, ps, ng]], np.r_[y[sd], np.ones(len(ps)), np.zeros(len(ng))],
                                                                                 sample_weight=np.r_[np.full(len(sd), W_), np.ones(len(ps) + len(ng))]); POL[f'SEED {n_} + LABEL-FREE'] = c2.decision_function(X)
P(f"{ND} documents, {len(R)} records (premises labelled {y.mean():.3f}); reader trained on {len(ps)} / {len(ng)} pseudo-labels  ({time.time() - T0:.0f}s)")
for k, s in POL.items():
    row = []
    for frac in (0.25, 0.10, 0.02):
        keep = np.argsort(-s)[:int(len(R) * frac)]; nb = collections.defaultdict(list); PR = collections.defaultdict(set); np_ = 0
        for i in keep: nb[R[i][1]].append((R[i][2], R[i][0])); nb[R[i][2]].append((R[i][1], R[i][0]))
        for i in keep:
            if np_ > 3000000: break
            for B_, C_ in ((R[i][1], R[i][2]), (R[i][2], R[i][1])):
                if len(nd[B_]) > 200: continue
                for A_, dA in nb[B_]:
                    if dA != R[i][0] and A_ != C_: PR[frozenset((A_, C_))].add(B_); np_ += 1
        ks = [p for p in PR if len(p) == 2 and (lambda u, v: bool(nd[u] & nd[v]))(*tuple(p))]
        g = np.array([p in LAB for p in ks]); b2 = np.array([len(PR[p]) >= 2 for p in ks])
        row.append(f"{frac:.0%}: premises {y[keep].mean():.2f}, A-C {len(PR)} proposed, {len(ks)} checkable right {g.mean() if len(g) else float('nan'):.3f}"
                   + (f", 2+ bridges {int(b2.sum())} right {g[b2].mean():.3f}" if b2.any() else ''))
    P(f"   {k:22s} " + " || ".join(row) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
