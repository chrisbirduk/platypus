"""WIDER RECORDS (19 Nov 2026; tools/wide_reader.py) - offline, distant set, LABEL-FREE reader (own-document labels for evaluation only).
Gap: only 34% of labelled pairs ever become a record (two names within 12 words of one sentence); 58% are stated only across sentences.
Record kinds:  NEAR (the current records)  |  FAR (same sentence, 13-30 words apart)  |  NEXT (A in one sentence, B in the next; the words
after A to the sentence end, a boundary mark, the words before B)  |  PRONOUN (a sentence led by he / she / it / they / his / her / its /
their: the ARTICLE'S SUBJECT - the first name of the first sentence - paired with each name in it; the words from the start to the name).
Reports: (1) share of labelled pairs that some record of each set covers; (2) the self-trained reader (usefulness pseudo-labels from all
kinds) per kind: AUC, precision of the top 10% within the kind, base rate; (3) the top 10% overall: its share of each kind and precision.
Usage: python3 tools/wide_reader.py <distant.json.gz>   env WR_DOCS (20000)"""
import os, sys, gzip, json, time, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
D = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(D), int(os.environ.get('WR_DOCS', 20000))); D = D[:ND]
PRON = {'he', 'she', 'it', 'they', 'his', 'her', 'its', 'their'}
recs = []; covered = collections.defaultdict(set); labs = []
for di, d in enumerate(D):
    cn = [collections.Counter(m['name'] for m in vs).most_common(1)[0][0] for vs in d['vertexSet']]
    lab = {frozenset((l['h'], l['t'])) for l in d['labels']}; labs.append(lab); S = d['sents']
    bys = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: bys[m['sent_id']].append((m['pos'][0], m['pos'][1], ei))
    for si in bys: bys[si].sort()
    def add(kind, ea, eb, words):
        if ea == eb or cn[ea] == cn[eb]: return
        recs.append((di, kind, cn[ea], cn[eb], ' '.join(words).lower()[:200], frozenset((ea, eb)) in lab)); covered[kind].add((di, frozenset((ea, eb))))
    for si, ms in bys.items():
        for x in range(len(ms)):
            for z in range(x + 1, len(ms)):
                (a0, a1, ea), (b0, b1, eb) = ms[x], ms[z]
                if a1 > b0: continue
                gap = b0 - a1
                if gap <= 12: add('NEAR', ea, eb, S[si][a1:b0])
                elif gap <= 30: add('FAR', ea, eb, S[si][a1:a1 + 6] + ['...'] + S[si][b0 - 6:b0])
        if si + 1 in bys:
            for (a0, a1, ea) in ms[-3:]:
                for (b0, b1, eb) in bys[si + 1][:3]:
                    if b0 <= 15: add('NEXT', ea, eb, S[si][a1:a1 + 8] + ['<S>'] + S[si + 1][:b0])
    subj = bys[0][0][2] if 0 in bys and bys[0] else None
    if subj is not None:
        for si, ms in bys.items():
            if si == 0 or not S[si] or S[si][0].lower() not in PRON: continue
            for (b0, b1, eb) in ms[:4]:
                if b0 <= 15: add('PRONOUN', subj, eb, ['<PRON>'] + S[si][1:b0])
allab = sum(len(l) for l in labs)
P(f"{ND} documents, {len(recs)} records ({collections.Counter(r[1] for r in recs)}); {allab} labelled pairs  ({time.time() - T0:.0f}s)")
def cov(kinds):
    c = set()
    for k in kinds: c |= {x for x in covered[k] if x[1] in labs[x[0]]}
    return len(c) / allab
P("1. LABELLED PAIRS COVERED by some record: " + " | ".join(f"{'+'.join(k)} {cov(k):.3f}" for k in (['NEAR'], ['NEAR', 'FAR'], ['NEAR', 'NEXT'], ['NEAR', 'PRONOUN'], ['NEAR', 'FAR', 'NEXT', 'PRONOUN'])))
y = np.array([r[5] for r in recs]); doc = np.array([r[0] for r in recs]); kind = np.array([r[1] for r in recs]); key = [frozenset((r[2], r[3])) for r in recs]
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k in zip(recs, key): pd[k].add(r[0]); nd[r[2]].add(r[0]); nd[r[3]].add(r[0])
def u_(i):
    r = recs[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[2]]) - 1, len(nd[r[3]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
tr = np.where(doc < int(ND * 0.8))[0]; te = doc >= int(ND * 0.8); U = np.array([u_(i) for i in tr]); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[4]} KIND_{r[1]}" for r in recs])
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); ps, ng = tr[ok & (U >= hi)], tr[ok & (U <= lo)]
c.fit(X[np.r_[ps, ng]], np.r_[np.ones(len(ps)), np.zeros(len(ng))]); s = c.decision_function(X)
P(f"2. READER (self-trained from usefulness over all kinds: {len(ps)} / {len(ng)} pseudo-labels, kinds of the positives {dict(collections.Counter(kind[ps]))}), held-out documents:")
for k in ('NEAR', 'FAR', 'NEXT', 'PRONOUN'):
    m = te & (kind == k)
    if m.sum() < 100 or y[m].all() or not y[m].any(): continue
    o = np.argsort(-s[m]); P(f"   {k:8s} n {m.sum():7d} base {y[m].mean():.3f} | AUC {roc_auc_score(y[m], s[m]):.3f} | top 10% within the kind {y[m][o[:len(o) // 10]].mean():.3f}")
o = np.argsort(-s[te]); top = np.where(te)[0][o[:te.sum() // 10]]
P(f"3. TOP 10% of all held-out records: precision {y[top].mean():.3f}; kinds " + ", ".join(f"{k} {np.mean(kind[top] == k):.2f} (right {y[top][kind[top] == k].mean() if (kind[top] == k).any() else float('nan'):.2f})" for k in ('NEAR', 'FAR', 'NEXT', 'PRONOUN'))
  + f" | related records found: {y[top].sum()} of {y[te].sum()} ({y[top].sum() / y[te].sum():.3f}); NEAR only would find {y[te & (kind == 'NEAR')].sum()} related at most")
P(f"DONE ({time.time() - T0:.0f}s)")
