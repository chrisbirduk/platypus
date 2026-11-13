"""THE CONSISTENCY TEACHER ON RAW TEXT, AS THE STORE WOULD SEE IT (13 Nov 2026; tools/consistency_raw.py) - offline, no effect on the package.
tools/consistency_teacher.py used the gold MENTIONS of Re-DocRED to find the names in its training text. The store has no gold: its names are
runs of capitalised words. Here the teacher counts over UNANNOTATED text - the Re-DocRED training documents (gold ignored) plus local
Wikipedia (WikiText, ~3x as much), optionally the other domains the model reads - with names found by capitalisation; the learner (same
features) is trained on that teacher and scored on the SAME held-out documents and gold-mention candidates as before (learner with gold
mentions in training: AUC 0.63, top 25% 0.37 related, base 0.24).
  NAMES     runs of capitalised words (a single sentence-initial word only if it is mostly capitalised mid-sentence elsewhere); digits kept as names
  RECORDS   two names in one sentence, 1-12 words apart and <= 60 characters (the words between may be empty: 'Dresden Saxony' is skipped)
  TEACHER   T1 corroboration (same names + wording in another document), T2 functional wording, T3 selective wording (as consistency_teacher.py)
Usage: python3 tools/consistency_raw.py [sources: redocred,wiki (default) | +law,pubmed,arxiv,prose]"""
import os
import re
import sys
import math
import time
import collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = (sys.argv[1] if len(sys.argv) > 1 else 'redocred,wiki').split(',')
_ct = open(os.path.join(HERE, 'consistency_teacher.py')).read()
exec(compile(_ct[:_ct.index('# ---- statistics from TRAINING documents only')], 'consistency_teacher', 'exec'))   # C, TR, TE, y, yt, auc, shape
TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-\.]*[A-Za-z0-9]|[A-Za-z0-9]|[,;:()\"]")
SPL = re.compile(r'(?<=[.!?])\s+(?=[A-Z"(])')
MP = os.environ.get('MINIPILE_DIR', '/tmp/claude-0/sweep/minipile')
texts = []                                                                     # (document id, text)
if 'redocred' in SRC:
    for di in sorted({C[i]['di'] for i in TR}): texts.append((f"r{di}", ' '.join(' '.join(s) for s in docs[di]['sents'])))
for nm_ in ('wiki', 'law', 'pubmed', 'arxiv', 'prose'):
    if nm_ in SRC:
        raw = open(f"{MP}/{nm_}.bin", 'rb').read().decode('utf-8', 'replace').replace('@-@', '-').replace('@,@', ',').replace('@.@', '.')
        parts = re.split(r'\n = [^=\n][^\n]*? = \n', raw) if nm_ == 'wiki' else [raw[i:i + 4000] for i in range(0, len(raw), 4000)]
        texts += [(f"{nm_}{k}", p.replace('\n', ' ')) for k, p in enumerate(parts) if len(p) > 200]
sents = [(d_, TOK.findall(s)) for d_, t in texts for s in SPL.split(t)]
sents = [(d_, s) for d_, s in sents if 4 <= len(s) <= 80]
low = collections.Counter(); capmid = collections.Counter()
for _, s in sents:
    for j, w in enumerate(s):
        if j and w[:1].islower(): low[w.lower()] += 1
        if j and w[:1].isupper(): capmid[w.lower()] += 1
def runs(s):
    out = []; j = 0
    while j < len(s):
        w = s[j]
        if w[:1].isupper() or w[:1].isdigit():
            k = j
            while k + 1 < len(s) and (s[k + 1][:1].isupper() or (s[k + 1] in ('of', 'de', 'the', 'and') and k + 2 < len(s) and s[k + 2][:1].isupper())): k += 1
            if s[k] in ('of', 'de', 'the', 'and'): k -= 1
            if not (j == 0 and k == 0 and capmid[w.lower()] < low[w.lower()]): out.append((j, k + 1))
            j = k + 1
        else: j += 1
    return out
R = []
for d_, s in sents:
    rr = runs(s)
    for x in range(len(rr)):
        for y_ in range(x + 1, len(rr)):
            (a0, a1), (b0, b1) = rr[x], rr[y_]
            if not 1 <= b0 - a1 <= 12 or len(' '.join(s[a0:b1])) > 60: continue
            A, B = s[a0:a1], s[b0:b1]
            if ' '.join(A) == ' '.join(B): continue
            R.append(dict(di=d_, A=A, B=B, mid=s[a1:b0], cA=' '.join(A), cB=' '.join(B), f=' '.join(s[a1:b0]).lower()))
P(f"RAW TEXT ({'+'.join(SRC)}): {len(texts)} documents, {len(sents)} sentences, {sum(len(s) for _, s in sents)} words | names by capitalisation | "
  f"records {len(R)} (gold-mention training candidates: {len(TR)})  ({time.time() - T0:.0f}s)")
trip = collections.defaultdict(set); rps = collections.defaultdict(lambda: collections.defaultdict(list))
shA = collections.defaultdict(collections.Counter); shB = collections.defaultdict(collections.Counter); fc_ = collections.Counter()
for r in R:
    trip[(r['cA'], r['f'], r['cB'])].add(r['di']); rps[r['f']][r['cA']].append(r['cB']); fc_[r['f']] += 1
    shA[r['f']][shape(r['A'])] += 1; shB[r['f']][shape(r['B'])] += 1
def ent(cn_):
    n = sum(cn_.values()); return -sum(v / n * math.log2(v / n) for v in cn_.values()) if n else 0.0
FUNC = {f: float(np.mean([len(set(v)) == 1 for v in d_.values() if len(v) >= 2])) for f, d_ in rps.items() if sum(1 for v in d_.values() if len(v) >= 2) >= 3}
SEL = {f: -(ent(shA[f]) + ent(shB[f])) for f in fc_ if fc_[f] >= 5}
P(f"teacher statistics: wordings {len(fc_)}, functional-scored {len(FUNC)}, selectivity-scored {len(SEL)}  ({time.time() - T0:.0f}s)")
t1 = np.array([float(len(trip[(r['cA'], r['f'], r['cB'])] - {r['di']}) > 0) for r in R])
t2 = np.array([FUNC.get(r['f'], np.nan) for r in R]); t3 = np.array([SEL.get(r['f'], np.nan) for r in R])
rk = lambda x: (lambda z: (np.argsort(np.argsort(z)) / len(z)))(np.where(np.isnan(x), np.nanmedian(x) if np.isfinite(x).any() else 0, x))
lab = rk(t2) + rk(t3) + 2 * t1
fq = collections.Counter([c['f'] for c in C] + [r['f'] for r in R])
HV = HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False, token_pattern=r"[A-Za-z]+|[^\sA-Za-z]", norm='l2')
def feats(items):
    W = HV.transform([' '.join(c['mid']) for c in items])
    ends = HV.transform([('FIRST_' + (c['mid'][0] if c['mid'] else 'none') + ' LAST_' + (c['mid'][-1] if c['mid'] else 'none')).lower() for c in items])
    D = np.array([[len(c['mid']), float(',' in c['mid']), float('(' in c['mid'] or ')' in c['mid']), float(any(w[:1].isdigit() for w in c['A'] + c['B'])),
                   float(any(w[:1].isupper() for w in c['mid'])), np.log1p(fq[c['f']]), float(len(c['mid']) == 0)] for c in items], float)
    return D
from sklearn.preprocessing import StandardScaler
Dr = feats(R); De = feats([C[i] for i in TE]); sc_ = StandardScaler().fit(Dr)
def X(items, D): return hstack([HV.transform([' '.join(c['mid']) for c in items]),
                                HV.transform([('FIRST_' + (c['mid'][0] if c['mid'] else 'none') + ' LAST_' + (c['mid'][-1] if c['mid'] else 'none')).lower() for c in items]),
                                csr_matrix(sc_.transform(D))]).tocsr()
Xr = X(R, Dr); Xe = X([C[i] for i in TE], De)
def rep(tag, sc):
    sc = np.where(np.isnan(sc), np.nanmedian(sc) if np.isfinite(sc).any() else 0.0, sc)
    o = np.argsort(-sc + 1e-9 * np.random.RandomState(5).rand(len(sc))); k1, k2 = len(sc) // 10, len(sc) // 4
    P(f"   {tag:62s} AUC {auc(sc[yt], sc[~yt]):.3f} | top 10%: precision {yt[o[:k1]].mean():.3f}, recall {yt[o[:k1]].sum() / yt.sum():.3f} | "
      f"top 25%: precision {yt[o[:k2]].mean():.3f}, recall {yt[o[:k2]].sum() / yt.sum():.3f}")
for q in (75, 90):
    hi = lab >= np.percentile(lab, q)
    rep(f"LEARNER from the RAW-TEXT teacher (top {100 - q}% as positives)", LogisticRegression(C=1.0, max_iter=3000).fit(Xr, hi).decision_function(Xe))
fsc = np.array([FUNC.get(C[i]['f'], np.nan) for i in TE]); ssc = np.array([SEL.get(C[i]['f'], np.nan) for i in TE])
rep("raw-text teachers applied directly (functional + selective)", rk(fsc) + rk(ssc))
P(f"DONE ({time.time() - T0:.0f}s)")
