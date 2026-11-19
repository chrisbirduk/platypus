"""READERS PER HORIZON, COMBINED (19 Nov 2026; tools/horizon_reader.py) - offline, distant set, LABEL-FREE (own-document labels for evaluation).
The user's idea: instead of one reader over every kind of record, one reader per HORIZON - how far apart and how the two names are joined -
each taught by usefulness WITHIN its horizon (so the horizon's form cannot be the shortcut), then combined. Horizons:
  H0 adjacent / comma / bracket / list | H1 short phrase (1-3 words) | H2 longer phrase (4-12) | H3 same sentence 13-30 words apart |
  H4 next sentence | H5 pronoun-led sentence (the article's subject)
Combination (label-free): each horizon's score turned into a probability of 'related' with that horizon's own pseudo-labels (Platt scaling
on the teacher's +1 / -1 records, scaled by the share of +1 votes the teacher gives in the horizon), then all records ranked together.
Arms: ONE READER (all records, one model, a horizon tag as a feature) | PER-HORIZON, COMBINED | PER-HORIZON, each horizon's top 10% (equal shares).
Reported on held-out documents: AUC overall and within each horizon; precision of the top 10% and the related records it finds per horizon.
Usage: python3 tools/horizon_reader.py <distant.json.gz>   env HR_DOCS (20000)"""
import os, sys, gzip, json, time, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
D = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(D), int(os.environ.get('HR_DOCS', 20000))); D = D[:ND]
PRON = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their'}; LW = {',', 'and', 'or', '(', ')', '-', ';', '/', '"'}
HN = ['H0 adjacent/comma/list', 'H1 short phrase', 'H2 longer phrase', 'H3 far, same sentence', 'H4 next sentence', 'H5 pronoun-led']
recs = []
for di, d in enumerate(D):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {frozenset((l['h'], l['t'])) for l in d['labels']}; S = d['sents']; bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si in bys: bys[si].sort()
    def add(h, ea, eb, words):
        if ea != eb and cn[ea] != cn[eb]: recs.append((di, h, cn[ea], cn[eb], ' '.join(w.lower() for w in words)[:200], frozenset((ea, eb)) in lab))
    for si, ms in bys.items():
        for x in range(len(ms)):
            for z in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[z]
                if a1 > b0: continue
                mid = S[si][a1:b0]; g = b0 - a1
                if g <= 12:
                    h = 0 if (g <= 1 or all(w.lower() in LW for w in mid) or '(' in mid) else (1 if g <= 3 else 2); add(h, ea, eb, mid)
                elif g <= 30: add(3, ea, eb, S[si][a1:a1 + 6] + ['...'] + S[si][b0 - 6:b0])
        if si + 1 in bys:
            for (a0, a1, ea) in ms[-3:]:
                for (b0, b1, eb) in bys[si + 1][:3]:
                    if b0 <= 15: add(4, ea, eb, S[si][a1:a1 + 8] + ['<s>'] + S[si + 1][:b0])
    subj = bys[0][0][2] if 0 in bys and bys[0] else None
    if subj is not None:
        for si, ms in bys.items():
            if si and S[si] and S[si][0].lower() in PRON:
                for (b0, b1, eb) in ms[:4]:
                    if b0 <= 15: add(5, subj, eb, ['<pron>'] + S[si][1:b0])
y = np.array([r[5] for r in recs]); doc = np.array([r[0] for r in recs]); hz = np.array([r[1] for r in recs]); te = doc >= int(ND * 0.8); tr = ~te
key = [frozenset((r[2], r[3])) for r in recs]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k in zip(recs, key): pd[k].add(r[0]); nd[r[2]].add(r[0]); nd[r[3]].add(r[0])
def u_(i):
    r = recs[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[2]]) - 1, len(nd[r[3]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
U = np.full(len(recs), np.nan); I = np.where(tr)[0]; U[I] = [u_(i) for i in I]
P(f"{ND} documents, {len(recs)} records: " + ", ".join(f"{HN[h]} {np.mean(hz == h):.2f} (related {y[hz == h].mean():.3f})" for h in range(6)) + f"  ({time.time() - T0:.0f}s)")
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[4]} HZ_{r[1]}" for r in recs])
def plabels(mask):
    m = mask & tr & ~np.isnan(U)
    if m.sum() < 50: return None, None
    hi, lo = np.quantile(U[m], 0.8), np.quantile(U[m], 0.5); return np.where(m & (U >= hi))[0], np.where(m & (U <= lo))[0]
def fit(ps, ng):
    c = LogisticRegression(C=1.0, max_iter=400, class_weight='balanced'); c.fit(X[np.r_[ps, ng]], np.r_[np.ones(len(ps)), np.zeros(len(ng))]); return c
ps, ng = plabels(np.ones(len(recs), bool)); s1 = fit(ps, ng).decision_function(X)
sH = np.zeros(len(recs)); pH = np.zeros(len(recs)); sh_rank = np.zeros(len(recs))
for h in range(6):
    m = hz == h; p_, n_ = plabels(m)
    if p_ is None: continue
    c = fit(p_, n_); s = c.decision_function(X[m]); sH[m] = s
    pl = LogisticRegression(C=1.0).fit(c.decision_function(X[np.r_[p_, n_]])[:, None], np.r_[np.ones(len(p_)), np.zeros(len(n_))])   # Platt on the pseudo-labels
    prior = len(p_) / (len(p_) + len(n_)); vote = (~np.isnan(U[m & tr])).mean()                 # the teacher's own +1 share in this horizon
    pH[m] = pl.predict_proba(s[:, None])[:, 1] * (len(p_) / max(1, (m & tr & ~np.isnan(U)).sum()) / 0.2)
    o = np.argsort(np.argsort(s)); sh_rank[m] = o / max(1, len(o) - 1)
def rep(nm, s):
    o = np.argsort(-s[te]); top = np.zeros(te.sum(), bool); top[o[:te.sum() // 10]] = True; yy = y[te]; hh = hz[te]
    P(f"   {nm:34s} AUC all {roc_auc_score(yy, s[te]):.3f} | top 10% precision {yy[top].mean():.3f}, related found {yy[top].sum()} | within-horizon AUC " +
      ", ".join(f"H{h} {roc_auc_score(yy[hh == h], s[te][hh == h]):.3f}" for h in range(6) if (hh == h).sum() > 100 and yy[hh == h].any()) +
      " | top 10% share by horizon " + ", ".join(f"H{h} {np.mean(hh[top] == h):.2f} ({yy[top & (hh == h)].mean() if (top & (hh == h)).any() else float('nan'):.2f})" for h in range(6)))
P("held-out documents:")
rep('ONE READER (horizon as a feature)', s1); rep('PER-HORIZON, combined by probability', pH); rep('PER-HORIZON, equal shares (rank)', sh_rank)
P(f"DONE ({time.time() - T0:.0f}s)")
