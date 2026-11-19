"""HOW MANY LABELS DOES THE READER NEED? (19 Nov 2026; tools/seed_reader.py) - offline, distant set. A SMALL LABELLED SEED (records from
training documents with their own-document labels) is mixed with the label-free teaching (usefulness pseudo-labels), to break the loop the
reader is stuck in (noisy teacher -> weak reader). Seed sizes SD_SIZES; arms:
  SEED ONLY                        the seed's labels alone
  SEED + LABEL-FREE                the seed (weight SD_W each) plus usefulness +1 / -1 records
  SEED + LABEL-FREE + CONTEXT      the same with 3 words of context each side
  ... + TYPES (reference)          the dataset's entity types as features (annotation: an upper bound for learned types)
Scored on held-out documents: AUC, top 10% precision, PHRASE recall in the top 10%, phrase-record AUC.
Usage: python3 tools/seed_reader.py <distant_50k.json.gz>   env SD_DOCS (30000), SD_SIZES (0,500,2000,5000,20000), SD_W (3)"""
import os, sys, gzip, json, time, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
D = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(D), int(os.environ.get('SD_DOCS', 30000))); D = D[:ND]
LW = {',', 'and', 'or', '(', ')', '-', ';', '/', '"'}; recs = []
for di, d in enumerate(D):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    ty = [collections.Counter(m.get('type', 'X') for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {frozenset((l['h'], l['t'])) for l in d['labels']}; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si, ms in bys.items():
        ms = sorted(ms); S = d['sents'][si]
        for x in range(len(ms)):
            for z in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[z]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                mid = [w.lower() for w in S[a1:b0]]
                recs.append((di, cn[ea], cn[eb], ' '.join(mid), mid[-1] if mid else '', len(mid), not (len(mid) <= 1 or all(w in LW for w in mid) or '(' in mid),
                             ' '.join(w.lower() for w in S[max(0, a0 - 3):a0]) + ' ' + ' '.join(w.lower() for w in S[b1:b1 + 3]), ty[ea], ty[eb], frozenset((ea, eb)) in lab))
del D
y = np.array([r[10] for r in recs]); doc = np.array([r[0] for r in recs]); te = doc >= int(ND * 0.8); tr = ~te; ph = np.array([r[6] for r in recs])
key = [frozenset((r[1], r[2])) for r in recs]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for i in np.where(tr)[0]: r = recs[i]; pd[key[i]].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
def u_(i):
    r = recs[i]; a_ = len(pd[key[i]] - {r[0]}); nA, nB = len(nd[r[1]] - {r[0]}), len(nd[r[2]] - {r[0]})
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
U = np.full(len(recs), np.nan); I = np.where(tr)[0]; U[I] = [u_(i) for i in I]; ok = tr & ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
ps, ng = np.where(ok & (U >= hi))[0], np.where(ok & (U <= lo))[0]
P(f"{ND} documents, {len(recs)} records (related {y.mean():.3f}; phrase records {ph.mean():.2f}); usefulness +1 {len(ps)} (right {y[ps].mean():.3f}), -1 {len(ng)}  ({time.time() - T0:.0f}s)")
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
base = lambda r: f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}"
XW = hv.transform([base(r) for r in recs]); XC = hv.transform([base(r) + ' ' + ' '.join('C_' + w for w in r[7].split()) for r in recs])
XT = hv.transform([base(r) + ' ' + ' '.join('C_' + w for w in r[7].split()) + f" TP_{r[8]}_{r[9]} TA_{r[8]} TB_{r[9]}" for r in recs])
W8 = float(os.environ.get('SD_W', 3))
def fit(X, ix, yy, w):
    c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[ix], yy, sample_weight=w); return c.decision_function(X[te])
def rep(nm, s):
    yy = y[te]; o = np.argsort(-s); top = np.zeros(len(s), bool); top[o[:len(s) // 10]] = True; p_ = ph[te]
    return f"{nm}: AUC {roc_auc_score(yy, s):.3f}, top10% {yy[top].mean():.3f}, phrase recall {top[yy & p_].mean():.3f}, phrase AUC {roc_auc_score(yy[p_], s[p_]):.3f}"
rs = np.random.RandomState(0); pool = np.where(tr)[0]
for n in [int(x) for x in os.environ.get('SD_SIZES', '0,500,2000,5000,20000').split(',')]:
    seed = pool[rs.permutation(len(pool))[:n]] if n else np.zeros(0, int); line = []
    if n: line.append(rep('SEED ONLY', fit(XW, seed, y[seed], np.ones(n))))
    lf = np.r_[ps, ng]; ly = np.r_[np.ones(len(ps)), np.zeros(len(ng))]
    ix = np.r_[seed, lf]; yy = np.r_[y[seed], ly]; w = np.r_[np.full(n, W8), np.ones(len(lf))]
    line.append(rep('SEED + LABEL-FREE', fit(XW, ix, yy, w)))
    line.append(rep('+ CONTEXT', fit(XC, ix, yy, w)))
    line.append(rep('+ CONTEXT + TYPES (reference)', fit(XT, ix, yy, w)))
    P(f"   seed {n:6d} (related {y[seed].mean() if n else float('nan'):.3f}) || " + " || ".join(line) + f"  ({time.time() - T0:.0f}s)")
P(f"DONE ({time.time() - T0:.0f}s)")
