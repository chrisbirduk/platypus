"""WHERE THE READER FAILS: TEACHER OR FEATURES? (19 Nov 2026; tools/reader_ceiling.py) - offline, distant set. The labels are used to DIAGNOSE.
The self-trained reader (usefulness pseudo-labels -> words between the names) reaches about 0.65 AUC and misses relations stated by phrases.
Two possible causes, separable by training the same model on TRUE labels:
  - if the true-label reader does well on phrase records, the FEATURES can carry the relation and the TEACHER is the problem;
  - if it also fails there, the FEATURES (bag of words between the names) are the problem.
Reports, on held-out documents, by wording form (comma / bracket / list or adjacent | short phrase 1-3 words | longer phrase 4-12):
  1. AUC of the self-trained reader and of the true-label reader (same features), and the gap;
  2. the TEACHER's quality by form: precision of usefulness +1 and -1 votes, and how many it gives;
  3. a learning curve of the true-label reader on phrase records (how many labels the features need);
  4. extra FEATURES with true labels: + the names' types (the dataset's entity types: an upper bound for learned types), + the words
     around the pair (3 before, 3 after), + the names themselves (memorising pairs) - which ones lift the phrase records.
Usage: python3 tools/reader_ceiling.py <distant_50k.json.gz>   env RC_DOCS (30000)"""
import os, sys, gzip, json, time, types, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
T0 = time.time(); P = lambda s: print(s, flush=True)
D = json.load(gzip.open(sys.argv[1], 'rt')); ND = min(len(D), int(os.environ.get('RC_DOCS', 30000))); D = D[:ND]
LW = {',', 'and', 'or', '(', ')', '-', ';', '/', '"'}
recs = []
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
                form = 0 if (len(mid) <= 1 or all(w in LW for w in mid) or '(' in mid) else (1 if len(mid) <= 3 else 2)
                recs.append(dict(d=di, A=cn[ea], B=cn[eb], mid=' '.join(mid), last=mid[-1] if mid else '', n=len(mid), form=form, ta=ty[ea], tb=ty[eb],
                                 ctx=' '.join(w.lower() for w in S[max(0, a0 - 3):a0]) + ' | ' + ' '.join(w.lower() for w in S[b1:b1 + 3]), y=frozenset((ea, eb)) in lab))
y = np.array([r['y'] for r in recs]); doc = np.array([r['d'] for r in recs]); form = np.array([r['form'] for r in recs]); te = doc >= int(ND * 0.8); tr = ~te
key = [frozenset((r['A'], r['B'])) for r in recs]; pd, nd = collections.defaultdict(set), collections.defaultdict(set)
for r, k in zip(recs, key): pd[k].add(r['d']); nd[r['A']].add(r['d']); nd[r['B']].add(r['d'])
def u_(i):
    r = recs[i]; a_ = len(pd[key[i]]) - 1; nA, nB = len(nd[r['A']]) - 1, len(nd[r['B']]) - 1
    if not a_: return np.nan
    hA, hB = a_ / (nA + 2), a_ / (nB + 2); return max(hA * np.log(ND / (nB + 2)) - (1 - hA) * 0.5, hB * np.log(ND / (nA + 2)) - (1 - hB) * 0.5)
U = np.full(len(recs), np.nan); I = np.where(tr)[0]; U[I] = [u_(i) for i in I]; ok = tr & ~np.isnan(U); hi, lo = np.quantile(U[ok], 0.8), np.quantile(U[ok], 0.5)
ps, ng = np.where(ok & (U >= hi))[0], np.where(ok & (U <= lo))[0]
FN = ['comma / bracket / list', 'short phrase (1-3 words)', 'longer phrase (4-12 words)']
P(f"{ND} documents, {len(recs)} records; forms {[f'{FN[f]} {np.mean(form == f):.2f} (related {y[form == f].mean():.3f})' for f in range(3)]}  ({time.time() - T0:.0f}s)")
hv = HashingVectorizer(n_features=2 ** 20, ngram_range=(1, 2), alternate_sign=False, norm='l2', token_pattern=r"[^\s]+")
txt = {'words': lambda r: f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3}",
       '+ types': lambda r: f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3} TP_{r['ta']}_{r['tb']} TA_{r['ta']} TB_{r['tb']}",
       '+ context': lambda r: f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3} " + ' '.join('C_' + w for w in r['ctx'].split()),
       '+ types + context': lambda r: f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3} TP_{r['ta']}_{r['tb']} " + ' '.join('C_' + w for w in r['ctx'].split()),
       '+ the names': lambda r: f"{r['mid']} LAST_{r['last'] or 'none'} DIST_{min(r['n'], 12) // 3} NA_{r['A'].replace(' ', '_')} NB_{r['B'].replace(' ', '_')}"}
XS = {k: hv.transform([f(r) for r in recs]) for k, f in txt.items()}
def fit(X, ix, yy, C=1.0):
    c = LogisticRegression(C=C, max_iter=400, class_weight='balanced'); c.fit(X[ix], yy); return c.decision_function(X)
def aucf(s, m):
    return roc_auc_score(y[m], s[m]) if y[m].any() and (~y[m]).any() else float('nan')
sS = fit(XS['words'], np.r_[ps, ng], np.r_[np.ones(len(ps)), np.zeros(len(ng))])
trS = np.where(tr)[0][np.random.RandomState(0).permutation(tr.sum())[:200000]]
sT = fit(XS['words'], trS, y[trS])
P("1. SAME FEATURES (words between), held-out AUC by form:   self-trained (usefulness)  |  TRUE labels  |  gap")
for f in range(3):
    m = te & (form == f); P(f"   {FN[f]:28s} n {m.sum():7d}  {aucf(sS, m):.3f}  |  {aucf(sT, m):.3f}  |  {aucf(sT, m) - aucf(sS, m):+.3f}")
m = te; P(f"   {'ALL':28s} n {m.sum():7d}  {aucf(sS, m):.3f}  |  {aucf(sT, m):.3f}  |  {aucf(sT, m) - aucf(sS, m):+.3f}")
P("2. THE TEACHER by form (training documents): usefulness +1 precision (n) | -1 precision of 'unrelated' (n) | records it votes on")
for f in range(3):
    mf = tr & (form == f); pp = np.intersect1d(ps, np.where(mf)[0]); nn = np.intersect1d(ng, np.where(mf)[0])
    P(f"   {FN[f]:28s} +1 {y[pp].mean():.3f} ({len(pp)}) | -1 {(~y[nn]).mean():.3f} ({len(nn)}) | votes on {(len(pp) + len(nn)) / max(1, mf.sum()):.3f} of {mf.sum()} (base related {y[mf].mean():.3f})")
P("3. LEARNING CURVE, true labels, phrase records only (forms 1-2), held-out phrase AUC:")
ph_tr = np.where(tr & (form > 0))[0]; mph = te & (form > 0)
for n in (1000, 5000, 20000, 100000):
    if n > len(ph_tr): continue
    ix = ph_tr[np.random.RandomState(1).permutation(len(ph_tr))[:n]]; P(f"   {n:6d} labels: {aucf(fit(XS['words'], ix, y[ix]), mph):.3f}")
P("4. EXTRA FEATURES, true labels (200k training records), held-out AUC: all | comma-type | short phrase | longer phrase")
for k in txt:
    s = sT if k == 'words' else fit(XS[k], trS, y[trS]); P(f"   {k:20s} {aucf(s, te):.3f} | " + " | ".join(f"{aucf(s, te & (form == f)):.3f}" for f in range(3)))
s2 = fit(XS['+ types + context'], np.r_[ps, ng], np.r_[np.ones(len(ps)), np.zeros(len(ng))])
P(f"   SELF-TRAINED with + types + context (label-free except the types): {aucf(s2, te):.3f} | " + " | ".join(f"{aucf(s2, te & (form == f)):.3f}" for f in range(3)))
P(f"DONE ({time.time() - T0:.0f}s)")
