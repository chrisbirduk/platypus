"""NAME TYPES WITHOUT LABELS (13 Nov 2026; tools/name_types.py) - offline, no effect on the package.
tools/relation_free.py: the GOLD entity-type pair of two names alone tells which relation a record expresses (AUC 0.83; words between 0.63-0.67).
Blind people infer appearance from category ('a bird, so feathers'); unsupervised relation extraction finds entity types the strongest cue.
Can the store learn name TYPES with no labels, from the contexts names occur in?
  CONTEXTS   every mention of a name (Re-DocRED train + dev, gold mention spans as names; types never used): the 2 words before and after
             (position-tagged), the name's own shape (digits, number of capitalised words, its last word)
  TYPES      names x context features, PPMI, SVD (64), k-means (k = 6, 12, 24, 48); written to NT_OUT as {name: [type per k]}
  SCORED     purity / NMI against the gold types (PER, LOC, ORG, TIME, NUM, MISC); then tools/relation_free.py (RF_TYPES=NT_OUT) uses the
             learned type pair as the relation cue
NT_FEAT (13 Nov 16:05): ctx (contexts), shape, inner (the name's own words and letter triples), comma-separated.
Usage: python3 tools/name_types.py   env NT_OUT (default <scratchpad>/name_types.json)"""
import os
import json
import time
import collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
P = lambda *a: print(*a, flush=True); T0 = time.time()
SCR = '/tmp/claude-0/-home-claude/62aed2de-fc26-5d81-b0ca-dc2522d86ab8/scratchpad'
DIR = os.environ.get('DV_DIR', SCR + '/bench_dl'); OUT = os.environ.get('NT_OUT', SCR + '/name_types.json')
docs = json.load(open(DIR + '/train_revised.json')) + json.load(open(DIR + '/dev_revised.json'))
feat = collections.defaultdict(collections.Counter); gtype = collections.defaultdict(collections.Counter)
for d in docs:
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    for ei, vs in enumerate(d['vertexSet']):
        n = cn[ei]
        for m in vs:
            toks = d['sents'][m['sent_id']]; a, b = m['pos']
            gtype[n][m.get('type', vs[0].get('type', 'X'))] += 1
            if 'ctx' not in os.environ.get('NT_FEAT', 'ctx,shape'): continue
            for k in (1, 2):
                if a - k >= 0: feat[n][f"L{k}:{toks[a - k].lower()}"] += 1
                if b + k - 1 < len(toks): feat[n][f"R{k}:{toks[b + k - 1].lower()}"] += 1
        w = n.split(); FS = os.environ.get('NT_FEAT', 'ctx,shape')
        if 'shape' in FS:
            feat[n]['SHAPE:' + ('num' if any(c.isdigit() for c in n) else f"cap{min(3, sum(t[:1].isupper() for t in w))}")] += 3
            feat[n]['LAST:' + w[-1].lower()] += 2
        if 'inner' in FS:                                                       # the name's own words and letter triples (Smith / -burg / -ton)
            for t in w: feat[n]['W:' + t.lower()] += 2
            for t in w:
                tt = '^' + t.lower() + '$'
                for q in range(len(tt) - 2): feat[n]['C3:' + tt[q:q + 3]] += 1
names = sorted(feat); fi = {}
rows, cols, vals = [], [], []
for i, n in enumerate(names):
    for f, c in feat[n].items(): rows.append(i); cols.append(fi.setdefault(f, len(fi))); vals.append(c)
M = csr_matrix((np.array(vals, float), (rows, cols)), shape=(len(names), len(fi)))
keep = np.asarray((M > 0).sum(0)).ravel() >= 3; M = M[:, keep]
tot = M.sum(); rs_ = np.asarray(M.sum(1)).ravel(); cs_ = np.asarray(M.sum(0)).ravel(); C = M.tocoo()
pm = np.log(C.data * tot / (rs_[C.row] * cs_[C.col])); kp = pm > 0
X = TruncatedSVD(64, random_state=0).fit_transform(csr_matrix((pm[kp], (C.row[kp], C.col[kp])), shape=M.shape))
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
g = np.array([gtype[n].most_common(1)[0][0] for n in names])
P(f"names {len(names)}, context features {M.shape[1]} | gold types {collections.Counter(g).most_common()}  ({time.time() - T0:.0f}s)")
OUTD = {n: [] for n in names}
for k in (6, 12, 24, 48):
    lab = KMeans(k, n_init=4, random_state=0).fit_predict(X)
    pur = sum(collections.Counter(g[lab == c]).most_common(1)[0][1] for c in range(k)) / len(g)
    P(f"   k {k:2d}: purity against gold types {pur:.3f} | NMI {normalized_mutual_info_score(g, lab):.3f}  ({time.time() - T0:.0f}s)")
    for n, l in zip(names, lab): OUTD[n].append(int(l))
json.dump({'k': [6, 12, 24, 48], 'types': OUTD}, open(OUT, 'w'))
P(f"written {OUT}; DONE ({time.time() - T0:.0f}s)")
