"""A CLEANER TEACHER, THEN RICHER FEATURES (19 Nov 2026; tools/clean_teacher.py) - offline, distant set, LABEL-FREE (own-document labels and
the dataset's entity types for evaluation only). Diagnosis (tools/reader_ceiling.py): the teacher (usefulness, +1 votes about 0.4 right)
costs the reader 0.10-0.13 AUC, and richer features (types, context) raise the true-label ceiling 0.74 -> 0.82 but HURT a reader taught by a
noisy teacher. Here the CONFIRMED facts (about 0.7 right) become the teacher, then the features are enriched.
Learned types (label-free): every name is described by the words around its mentions in documents read so far; k-means on a reduced
name x word matrix gives TC_K types (purity against the dataset's types is reported for reference only).
Documents arrive in blocks (TC_BLOCK); after each block every arm is retrained from what was read and scored on the NEXT, unread block:
  USEFUL            usefulness +1 / -1, words between
  KNOWLEDGE         + every record of a confirmed pair positive (confirmed: the reader's top 5% of a block, stated again in a later block in
                    another document, in the reader's top 30%), words between
  CONFIRMED         positives = every record of a confirmed pair only; negatives = usefulness -1 (precision 0.86-0.93); words between
  ... + TYPES + CONTEXT   the same teachers with the learned type pair and 3 words of context each side
Usage: python3 tools/clean_teacher.py <distant_50k.json.gz>   env TC_DOCS (30000), TC_BLOCK (5000), TC_K (24)"""
import os, sys, gzip, json, time, collections
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
T0 = time.time(); P = lambda s: print(s, flush=True)
D = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(D), int(os.environ.get('TC_DOCS', 30000))); D = D[:ND]; BL = int(os.environ.get('TC_BLOCK', 5000)); NB = ND // BL; K = int(os.environ.get('TC_K', 24))
LW = {',', 'and', 'or', '(', ')', '-', ';', '/', '"'}; recs = []; ments = []
for di, d in enumerate(D):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    ty = [collections.Counter(m.get('type', 'X') for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {frozenset((l['h'], l['t'])) for l in d['labels']}; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs:
            bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei)); S = d['sents'][m['sent_id']]
            ments.append((di, cn[ei], ty[ei], ' '.join(w.lower() for w in S[max(0, m['pos'][0] - 3):m['pos'][0]] + S[m['pos'][1]:m['pos'][1] + 3])))
    for si, ms in bys.items():
        ms = sorted(ms); S = d['sents'][si]
        for x in range(len(ms)):
            for z in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[z]
                if ea == eb or a1 > b0 or cn[ea] == cn[eb] or b0 - a1 > 12: continue
                mid = [w.lower() for w in S[a1:b0]]
                recs.append(dict(d=di, A=cn[ea], B=cn[eb], mid=' '.join(mid), last=mid[-1] if mid else '', n=len(mid), txt=' '.join(S[a0:b1]),
                                 phrase=not (len(mid) <= 1 or all(w in LW for w in mid) or '(' in mid),
                                 ctx=' '.join(w.lower() for w in S[max(0, a0 - 3):a0]) + ' ' + ' '.join(w.lower() for w in S[b1:b1 + 3]), y=frozenset((ea, eb)) in lab))
del D
y = np.array([r['y'] for r in recs]); doc = np.array([r['d'] for r in recs]); blk = doc // BL; key = [frozenset((r['A'], r['B'])) for r in recs]; ph = np.array([r['phrase'] for r in recs])
P(f"{ND} documents, {NB} blocks, {len(recs)} records, {len(ments)} mentions  ({time.time() - T0:.0f}s)")
# ---- learned types from the first block's mentions (name x context-word matrix -> SVD -> k-means), extended to later names by nearest centroid
hv14 = HashingVectorizer(n_features=2 ** 14, alternate_sign=False, norm=None, token_pattern=r"[^\s]+")
nm_ctx = collections.defaultdict(list); nm_ty = collections.defaultdict(collections.Counter)
for di, n, t, c in ments: nm_ctx[n].append(c); nm_ty[n][t] += 1
names = list(nm_ctx); NI = {n: i for i, n in enumerate(names)}
M = hv14.transform([' '.join(nm_ctx[n][:30]) for n in names]); M = M.multiply(1.0 / (np.asarray(M.sum(1)) + 1.0)).tocsr()
Z = TruncatedSVD(64, random_state=0).fit_transform(M); Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
km = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=3, batch_size=8192).fit(Z); LT = km.labels_
gt = [nm_ty[n].most_common(1)[0][0] for n in names]; pur = sum(collections.Counter(g for g, l in zip(gt, LT) if l == c).most_common(1)[0][1] for c in set(LT) if (LT == c).any()) / len(names)
P(f"learned types: {K} clusters over {len(names)} names; purity against the dataset's types {pur:.3f} (majority type {collections.Counter(gt).most_common(1)[0][1] / len(gt):.3f}; evaluation only)  ({time.time() - T0:.0f}s)")
del M, Z, ments
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
XW = hv.transform([f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3}" for r in recs])
XR = hv.transform([f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3} TP_{LT[NI[r['A']]]}_{LT[NI[r['B']]]} TA_{LT[NI[r['A']]]} TB_{LT[NI[r['B']]]} "
                   + ' '.join('C_' + w for w in r['ctx'].split()) for r in recs])
P(f"features built  ({time.time() - T0:.0f}s)")
byk = collections.defaultdict(list)
for i, k in enumerate(key): byk[k].append(i)
def useful(t):
    I = np.where(blk < t)[0]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
    for i in I: pd[key[i]].add(recs[i]['d']); nd[recs[i]['A']].add(recs[i]['d']); nd[recs[i]['B']].add(recs[i]['d'])
    U = np.full(len(I), np.nan)
    for n_, i in enumerate(I):
        r = recs[i]; a_ = len(pd[key[i]]) - 1
        if a_:
            nA, nB = len(nd[r['A']]) - 1, len(nd[r['B']]) - 1; hA, hB = a_ / (nA + 2), a_ / (nB + 2)
            U[n_] = max(hA * np.log(t * BL / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(t * BL / (nA + 2)) - (1 - hB) * 0.5)
    ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5); return I[ok & (U >= hi)], I[ok & (U <= lo)]
def fit(X, pos, neg):
    c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[np.r_[pos, neg]], np.r_[np.ones(len(pos)), np.zeros(len(neg))]); return c
ARMS = [('USEFUL', 'W'), ('KNOWLEDGE', 'W'), ('CONFIRMED', 'W'), ('USEFUL', 'R'), ('KNOWLEDGE', 'R'), ('CONFIRMED', 'R')]
st = {a: dict(clf=None, tent={}, conf=set()) for a in ARMS}; curves = collections.defaultdict(list)
for t in range(1, NB):
    pu, nu = useful(t); In = np.where(blk == t)[0]; line = []
    for a in ARMS:
        A = st[a]; X = XW if a[1] == 'W' else XR
        if a[0] != 'USEFUL' and A['clf'] is not None:
            Ib = np.where(blk == t - 1)[0]; sb = A['clf'].decision_function(X[Ib]); thr = np.quantile(sb, 0.7); sc = dict(zip(Ib, sb))
            for k_, (i0, d0) in list(A['tent'].items()):
                if k_ not in A['conf'] and any(blk[j] == t - 1 and doc[j] != d0 and recs[j]['txt'] != recs[i0]['txt'] and sc[j] >= thr for j in byk[k_]): A['conf'].add(k_)
            for i in Ib[np.argsort(-sb)[:len(Ib) // 20]]:
                if key[i] not in A['tent']: A['tent'][key[i]] = (i, doc[i])
        kp = np.array([j for k_ in A['conf'] for j in byk[k_] if blk[j] < t], int)
        if a[0] == 'USEFUL' or (a[0] != 'USEFUL' and len(kp) < 200): pos, neg = pu, nu
        elif a[0] == 'KNOWLEDGE': pos = np.unique(np.r_[pu, kp]); neg = np.setdiff1d(nu, kp)
        else: pos = kp; neg = np.setdiff1d(nu, kp)
        if a[0] != 'USEFUL' and A['clf'] is None: pos, neg = pu, nu
        A['clf'] = fit(X, pos, neg); s = A['clf'].decision_function(X[In]); o = np.argsort(-s); top = np.zeros(len(In), bool); top[o[:len(o) // 10]] = True
        yy = y[In]; au = roc_auc_score(yy, s); curves[f"{a[0]} {'WORDS' if a[1] == 'W' else '+ TYPES + CONTEXT'}"].append(au)
        line.append(f"{a[0]} {'W' if a[1] == 'W' else 'R'}: AUC {au:.3f} top10% {yy[top].mean():.3f} phrase recall {top[yy & ph[In]].mean():.3f}"
                    + (f" (conf {len(A['conf'])}, right {np.mean([y[byk[k_][0]] for k_ in A['conf']]) if A['conf'] else float('nan'):.2f}; pos {len(pos)} right {y[pos].mean():.2f})" if a[0] != 'USEFUL' else f" (pos right {y[pos].mean():.2f})"))
    P(f"   block {t}: " + " || ".join(line) + f"  ({time.time() - T0:.0f}s)")
P("CURVES (AUC on each unread block): " + " || ".join(f"{a}: " + " ".join(f"{x:.3f}" for x in v) for a, v in curves.items()))
P(f"DONE ({time.time() - T0:.0f}s)")
