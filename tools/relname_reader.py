"""NAMING RELATIONS ON REAL TEXT (19 Nov 2026; tools/relname_reader.py) - offline, distant set, LABEL-FREE (gold relation types for evaluation).
Stage 2 of real-text inference: can the reader group its kept pairs by WHAT relation they state? Records = the wording reader's top slice
(self-trained from usefulness); groups by (1) the last word between the names, (2) the normalised words between (backed off to the last word
under 3 records), (3) k-means on hashed word n-grams of the words between plus the order and the two names' shapes. Scored on the kept records
that have a gold relation (Wikidata property of the ordered pair): purity and NMI of the groups against the commonest gold relation, and the
share of records in groups whose majority relation covers 60%+ of them (usable typed groups).
Usage: python3 tools/relname_reader.py <distant_50k.json.gz>   env RN_TOP (0.10), RN_K (64)"""
import os, sys, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import normalized_mutual_info_score
T0 = time.time(); P = lambda s: print(s, flush=True)
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=0))
R, ND, PRD = pm._PM['R'], pm._PM['ND'], pm._PM['PRD']; TOP = float(os.environ.get('RN_TOP', 0.10)); K = int(os.environ.get('RN_K', 64))
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); key = [frozenset((r[1], r[2])) for r in R]
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k in zip(R, key): pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def u_(i):
    r = R[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
tr = np.where(doc < int(ND * 0.8))[0]; U = np.array([u_(i) for i in tr]); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
hv = HashingVectorizer(n_features=2 ** int(os.environ.get("RN_BITS", 20)), ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[tr[ok & (U >= hi)], tr[ok & (U <= lo)]]], np.r_[np.ones((ok & (U >= hi)).sum()), np.zeros((ok & (U <= lo)).sum())])
s = c.decision_function(X); keep = np.argsort(-s)[:int(len(R) * TOP)]
def gold(r):
    a = PRD.get((r[1], r[2])); b = PRD.get((r[2], r[1]))
    if a: return sorted(a)[0]
    if b: return sorted(b)[0] + '^-1'
    return None
G = [(i, gold(R[i])) for i in keep]; G = [(i, g) for i, g in G if g]; gi = np.array([i for i, g in G]); gl = [g for i, g in G]
P(f"{ND} documents, {len(R)} records; kept top {TOP:.0%} = {len(keep)} (related {y[keep].mean():.3f}); {len(G)} kept records with a gold relation, "
  f"{len(set(gl))} relation types (commonest {collections.Counter(gl).most_common(5)})  ({time.time() - T0:.0f}s)")
def score(nm, lab):
    by = collections.defaultdict(list)
    for l, g in zip(lab, gl): by[l].append(g)
    pur = sum(collections.Counter(v).most_common(1)[0][1] for v in by.values()) / len(gl)
    good = sum(len(v) for v in by.values() if len(v) >= 5 and collections.Counter(v).most_common(1)[0][1] / len(v) >= 0.6) / len(gl)
    top = sorted(((len(v), collections.Counter(v).most_common(1)[0]) for l, v in by.items() if len(v) >= 20), reverse=True)[:4]
    P(f"   {nm:44s} groups {len(by):5d} | purity {pur:.3f} | NMI {normalized_mutual_info_score(gl, lab):.3f} | records in usable groups (5+, majority 60%+) {good:.3f} | "
      f"largest: {'; '.join(f'{n} records -> {g} {m / n:.0%}' for n, (g, m) in top)}")
score('one group (baseline)', ['all'] * len(gl))
score('LAST WORD between the names', [R[i][4] or '-' for i in gi])
fw = collections.Counter(R[i][3] for i in gi)
score('WORDS BETWEEN (backed off to the last word)', [R[i][3] if fw[R[i][3]] >= 3 else '~' + (R[i][4] or '-') for i in gi])
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3, batch_size=4096).fit(X[keep]); lk = dict(zip(keep, km.labels_))
score(f'K-MEANS on the words between (k {K})', [int(lk[i]) for i in gi])
km2 = MiniBatchKMeans(n_clusters=4 * K, random_state=0, n_init=3, batch_size=4096).fit(X[keep]); lk2 = dict(zip(keep, km2.labels_))
score(f'K-MEANS on the words between (k {4 * K})', [int(lk2[i]) for i in gi])
P(f"DONE ({time.time() - T0:.0f}s)")
