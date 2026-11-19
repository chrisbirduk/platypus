"""WHAT THE READER MISSES (19 Nov 2026; tools/miss_reader.py) - error analysis of the self-trained wording reader (usefulness pseudo-labels; no
labels in training) on the distant set, with the labels used to look at its mistakes.
  1. STRUCTURAL: of the labelled pairs in documents, how many ever become a RECORD (two names within 12 words of one sentence)? split by
     where the pair is stated (same sentence near / same sentence far / different sentences only).
  2. RECALL of the reader's top 10%, among related records, by gold relation, by the names' types, by distance, by wording form, by whether
     the pair recurs in other documents.
  3. FALSE POSITIVES in its top 10%: their commonest wordings and type pairs, against the related ones.
Usage: python3 tools/miss_reader.py <distant.json.gz>   env MR_TOP (0.10)"""
import os, sys, gzip, json, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import platypus.inference.premises as pm
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True); TOP = float(os.environ.get('MR_TOP', 0.10))
# ---- 1. structural coverage (raw documents)
D = json.load(gzip.open(sys.argv[1], 'rt')); cov = collections.Counter(); cov_rel = collections.defaultdict(collections.Counter)
for d in D[:20000]:
    near, far, any_s = set(), set(), set(); ments = collections.defaultdict(list)
    for ei, vs in enumerate(d['vertexSet']):
        for m in vs: ments[ei].append((m['sent_id'], m['pos'][0], m['pos'][1]))
    for l in d['labels']:
        h, t = l['h'], l['t']; best = 'different sentences only'
        for (sa, a0, a1) in ments[h]:
            for (sb, b0, b1) in ments[t]:
                if sa != sb: continue
                gap = (b0 - a1) if a1 <= b0 else (a0 - b1) if b1 <= a0 else -1
                if 0 <= gap <= 12: best = 'same sentence, within 12 words (a RECORD)'
                elif best != 'same sentence, within 12 words (a RECORD)': best = 'same sentence, more than 12 words apart'
        cov[best] += 1; cov_rel[l['r']][best] += 1
tot = sum(cov.values()); del D
P(f"1. STRUCTURAL: {tot} labelled pairs in 20,000 documents: " + " | ".join(f"{k} {v / tot:.3f}" for k, v in cov.most_common()))
rc = sorted(cov_rel.items(), key=lambda kv: -sum(kv[1].values()))[:10]
P("   by relation (share that ever becomes a record): " + ", ".join(f"{r} {c['same sentence, within 12 words (a RECORD)'] / sum(c.values()):.2f} (n {sum(c.values())})" for r, c in rc))
# ---- the reader
pm._load_distant(types.SimpleNamespace(inf_premises_path=sys.argv[1], inf_rret_list2=1))
R, ND, PRD, TY = pm._PM['R'], pm._PM['ND'], pm._PM['PRD'], pm._PM['TY']; R = [r for r in R if r[0] < 50000]; ND = min(ND, 50000)
y = np.array([bool(r[6]) for r in R]); doc = np.array([r[0] for r in R]); key = [frozenset((r[1], r[2])) for r in R]
pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k in zip(R, key): pd[k].add(r[0]); nd[r[1]].add(r[0]); nd[r[2]].add(r[0])
seen = np.array([len(pd[k]) > 1 for k in key])
def u_(i):
    r = R[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r[1]]) - 1, len(nd[r[2]]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
tr = np.where(doc < int(ND * 0.8))[0]; U = np.array([u_(i) for i in tr]); ok = ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
X = hv.transform([f"{r[3]} LAST_{r[4] or 'none'} DIST_{min(r[5], 12) // 3}" for r in R])
c = LogisticRegression(C=1.0, max_iter=300, class_weight='balanced'); c.fit(X[np.r_[tr[ok & (U >= hi)], tr[ok & (U <= lo)]]], np.r_[np.ones((ok & (U >= hi)).sum()), np.zeros((ok & (U <= lo)).sum())])
s = c.decision_function(X); kept = s >= np.quantile(s, 1 - TOP)
P(f"2. RECALL of the reader's top {TOP:.0%} (precision {y[kept].mean():.3f}; recall over all related records {kept[y].mean():.3f}; base {y.mean():.3f})")
def tab(nm, lab, n=12, minn=300):
    by = collections.defaultdict(list)
    for l, k, yy in zip(lab, kept, y):
        if yy: by[l].append(k)
    rows = sorted(((len(v), l, np.mean(v)) for l, v in by.items() if len(v) >= minn), reverse=True)[:n]
    P(f"   {nm}: " + " | ".join(f"{l} recall {r:.2f} (n {n_})" for n_, l, r in rows))
gold = [(sorted(PRD.get((r[1], r[2])) or [])[:1] or [(sorted(PRD.get((r[2], r[1])) or [])[:1] or ['-'])[0] + '^-1'])[0] for r in R]
tab('by gold relation', gold)
tab('by type pair (first, second)', [f"{TY.get(r[1], '?')}-{TY.get(r[2], '?')}" for r in R])
tab('by words between', ['0' if r[5] == 0 else '1' if r[5] == 1 else '2-3' if r[5] <= 3 else '4-8' if r[5] <= 8 else '9-12' for r in R], minn=100)
lst = np.array([pm._rret_listlike(r) for r in R])
tab('by wording form', ['list-like' if l else ('parenthesis' if '(' in r[3] else 'comma only' if r[3].strip() in (',', '') else 'verb or preposition') for r, l in zip(R, lst)], minn=100)
tab('by recurrence', ['pair seen in another document' if v else 'pair seen once' for v in seen])
P("3. FALSE POSITIVES in the top slice vs RELATED ones there - commonest words between:")
fp = collections.Counter(r[3] for r, k, yy in zip(R, kept, y) if k and not yy); tp = collections.Counter(r[3] for r, k, yy in zip(R, kept, y) if k and yy)
P("   unrelated: " + ", ".join(f"'{w}' {n}" for w, n in fp.most_common(15)))
P("   related:   " + ", ".join(f"'{w}' {n}" for w, n in tp.most_common(15)))
mw = collections.Counter(r[3] for r, k, yy in zip(R, kept, y) if (not k) and yy)
P("   MISSED related (not in the top slice), commonest words between: " + ", ".join(f"'{w}' {n}" for w, n in mw.most_common(20)))
ft = collections.Counter(f"{TY.get(r[1], '?')}-{TY.get(r[2], '?')}" for r, k, yy in zip(R, kept, y) if k and not yy)
P("   unrelated in the top slice by type pair: " + ", ".join(f"{t} {n}" for t, n in ft.most_common(8)))
P(f"DONE ({time.time() - T0:.0f}s)")
